"""Helpers to collect, aggregate, and cache activations in PyTorch.

This module provides classes for efficiently collecting, aggregating, and
caching neural network activations during inference. It is designed to be
flexible and robust, supporting custom aggregation logic and providing an
object-oriented API for saving and loading the cache.

The main classes are `ActMax`, which stores the top-k activations for a single
layer, and `ActMaxCache`, which manages the process of hooking into a model and
populating `ActMax` instances for multiple layers.

Workflow Example
----------------
>>> import torch
>>> from torch.utils.data import TensorDataset
>>> from torchvision.models import resnet18
>>> from pathlib import Path
>>> from semanticlens.component_visualization import aggregators
>>> from semanticlens.component_visualization.activation_caching import ActMaxCache
>>>
>>> model = resnet18()
>>> layer_names = ["layer4.1.conv2"]
>>> dataset = TensorDataset(torch.randn(100, 3, 224, 224))
>>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
>>>
>>> # 1. Instantiate the cache.
>>> cache = ActMaxCache(
...     layer_names=layer_names,
...     aggregation_fn=aggregators.aggregate_conv_mean,
...     n_collect=10
... )
>>>
>>> # 2. Run the model within the hook context to collect activations.
>>> with cache.hook_context(model):
...     for batch in dataloader:
...         model(batch[0])
>>>
>>> # 3. Save the populated cache to disk.
>>> cache.store(Path("./my_cache"))
"""

from __future__ import annotations

import inspect
import logging
from collections import Counter, OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import safetensors
import safetensors.torch
import torch

from . import aggregators

logger = logging.getLogger(__name__)


DEFAULT_AGGREGATION_FUNCTION_MAP = {name: func for name, func in inspect.getmembers(aggregators, inspect.isfunction)}

# Configure device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class ActMax:
    """
    Tool for collecting and storing the top-k maximal activations.

    This class can be initialized with all dimensions known or can infer the
    number of latent dimensions from the first batch of data it receives.

    Parameters
    ----------
    n_collect : int
        Number of top activations to collect and store.
    n_latents : int, optional
        Number of latent dimensions (e.g., channels, neurons). If None,
        will be inferred from the first batch of data.

    Attributes
    ----------
    n_collect : int
        Number of top activations being collected.
    n_latents : int or None
        Number of latent dimensions.
    is_setup : bool
        Whether the internal tensors have been initialized.
    activations : torch.Tensor
        Tensor storing the top activation values.
    sample_ids : torch.Tensor
        Tensor storing the sample IDs corresponding to top activations.
    """

    def __init__(self, n_collect: int, n_latents: int | None = None):
        self.n_collect = n_collect
        self.n_latents = n_latents
        self.is_setup = False

        if n_latents is not None:
            self._setup_tensors()

    def _setup_tensors(self):
        """
        Initialize the tensors for storing activations and sample IDs.

        Creates tensors for storing the top activations and their corresponding
        sample IDs based on the number of latents and collection size.
        """
        self.activations = -torch.zeros(self.n_latents, self.n_collect, dtype=torch.bfloat16)
        self.sample_ids = -torch.ones(self.n_latents, self.n_collect, dtype=torch.int64)
        self.is_setup = True

    def update(self, acts: torch.Tensor, sample_ids: torch.Tensor):
        """
        Update activations with a new batch, setting up tensors on first call if needed.

        Parameters
        ----------
        acts : torch.Tensor
            Activation tensor with shape (batch_size, n_latents).
        sample_ids : torch.Tensor
            Sample IDs corresponding to the activations.

        Notes
        -----
        If this is the first call and tensors haven't been set up, the number
        of latents will be inferred from the activation tensor shape.
        """
        assert acts.ndim == 2
        if not self.is_setup:
            self.n_latents = acts.shape[1]
            self._setup_tensors()

        batch_acts = acts.T.to(self.activations.dtype)
        batch_sample_ids = sample_ids.repeat(self.n_latents, 1)

        # Find top-k among existing and new activations
        all_acts = torch.cat([self.activations, batch_acts], dim=1)
        all_ids = torch.cat([self.sample_ids, batch_sample_ids], dim=1)

        self.activations, indices = torch.topk(all_acts, k=self.n_collect, dim=1)
        self.sample_ids = torch.gather(all_ids, dim=1, index=indices)

    @property
    def alive_latents(self) -> torch.Tensor:
        """
        Return indices of latents with any non-zero activation.

        Returns
        -------
        torch.Tensor
            Indices of latent dimensions that have non-zero activations.
            Empty tensor if the instance is not set up.
        """
        if not self.is_setup:
            return torch.tensor([], dtype=torch.int64)
        return torch.where(self.activations.abs().sum(dim=1) > 0)[0]

    def store(self, file_path: str | Path, metadata: dict[str, str] | None = None):
        """
        Store tensors and metadata to a safetensors file.

        Parameters
        ----------
        file_path : str or Path
            Path where the data should be saved.
        metadata : dict[str, str], optional
            Additional metadata to store with the tensors.

        Notes
        -----
        If the instance is not set up, the method will log a warning and skip storage.
        """
        if not self.is_setup:
            logger.warning("Attempted to store an un-initialized ActMax instance; skipping.")
            return

        tensors = {
            "activations": self.activations,
            "sample_ids": self.sample_ids,
        }
        safetensors.torch.save_file(tensors, file_path, metadata=metadata)
        logger.debug(f"Stored ActMax data to {file_path}")

    @classmethod
    def load(cls, file_path: str | Path) -> ActMax:
        """
        Load an ActMax instance from a safetensors file.

        Parameters
        ----------
        file_path : str or Path
            Path to the safetensors file to load.

        Returns
        -------
        ActMax
            Loaded ActMax instance with data from the file.

        Raises
        ------
        ValueError
            If the file is missing required metadata for loading.
        """
        with safetensors.safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
            if metadata is None:
                raise ValueError(f"File {file_path} is missing required metadata for loading.")
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        n_collect = int(metadata["n_collect"])
        n_latents = int(metadata["n_latents"])

        instance = cls(n_collect=n_collect, n_latents=n_latents)
        instance.activations = tensors["activations"]
        instance.sample_ids = tensors["sample_ids"]
        return instance

class ActCache:
    """
    Base class to collect raw activations from model layers via forward hooks.

    This class provides the fundamental infrastructure for registering forward hooks
    on specified model layers and collecting their activations during inference.

    Parameters
    ----------
    layer_names : list of str
        Names of the model layers to hook and collect activations from.

    Attributes
    ----------
    layer_names : list of str
        Names of the layers being monitored.
    cache : dict[str, Any]
        Dictionary storing collected activations by layer name.
    handles : list[torch.utils.hooks.RemovableHandle]
        List of hook handles for proper cleanup.
    """

    def __init__(self, layer_names: list[str]):
        self.layer_names = layer_names
        self.cache: dict[str, Any] = OrderedDict()
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def _get_hook(self, name: str) -> Callable:
        """
        Create the hook function for a given layer name.

        Parameters
        ----------
        name : str
            Name of the layer to create a hook for.

        Returns
        -------
        callable
            Hook function that stores layer outputs in the cache.
        """

        def hook_fn(module, ins, outs):
            self.cache[name] = outs.detach().cpu()

        return hook_fn

    def _register_hooks(self, model: torch.nn.Module):
        """
        Register forward hooks on the specified model layers.

        Parameters
        ----------
        model : torch.nn.Module
            The model to register hooks on.
        """
        for name, module in model.named_modules():
            if name in self.layer_names:
                self.handles.append(module.register_forward_hook(self._get_hook(name)))

    def _finalize(self):
        """
        Perform post-processing after the context exits.

        This method can be overridden by subclasses to perform cleanup
        or additional processing when the hook context is exited.
        """
        pass

    @contextmanager
    def hook_context(self, model: torch.nn.Module):
        """
        Context manager to automatically register and remove hooks.

        Parameters
        ----------
        model : torch.nn.Module
            The model to register hooks on.

        Yields
        ------
        None
            Context for running model inference with hooks active.

        Notes
        -----
        This context manager ensures that hooks are properly cleaned up
        even if an exception occurs during inference.
        """
        self._register_hooks(model)
        try:
            yield
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self._finalize()

class CustomActMax(ActMax):
    """
        Tool for collecting and storing the top-k maximal activations.

        This class can be initialized with all dimensions known or can infer the
        number of latent dimensions from the first batch of data it receives.

        Parameters
        ----------
        n_collect : int
            Number of top activations to collect and store.
        n_latents : int, optional
            Number of latent dimensions (e.g., channels, neurons). If None,
            will be inferred from the first batch of data.

        Attributes
        ----------
        n_collect : int
            Number of top activations being collected.
        n_latents : int or None
            Number of latent dimensions.
        is_setup : bool
            Whether the internal tensors have been initialized.
        activations : torch.Tensor
            Tensor storing the top activation values.
        sample_ids : torch.Tensor
            Tensor storing the sample IDs corresponding to top activations.
        """

    def __init__(self, model_name, n_collect: int, n_latents: int | None = None, n_all_latents: list | None = None):
        self.model_name = model_name
        self.n_all_latents = n_all_latents

        super().__init__(n_collect, n_latents)

        if n_all_latents is not None:
            self._setup_tensors()

    # TODO: separately for each model ?
    # add self.activations_not_aggr
    def _setup_tensors(self):
        """
        Initialize the tensors for storing activations and sample IDs.

        Creates tensors for storing the top activations and their corresponding
        sample IDs based on the number of latents and collection size.
        """
        self.activations = -torch.zeros(self.n_latents, self.n_collect, dtype=torch.bfloat16)
        self.sample_ids = -torch.ones(self.n_latents, self.n_collect, dtype=torch.int64)

        if self.model_name == "resnet" or self.model_name == "vgg16" or self.model_name == "resnet_50":
            acts_not_aggr_shape = [self.n_collect] + list(self.n_all_latents) # --> [n_images, n_neurons, *h, *w]
            acts_not_aggr_shape = [acts_not_aggr_shape[1], acts_not_aggr_shape[0]] + acts_not_aggr_shape[2:]
        else:
            acts_not_aggr_shape = [self.n_collect] + list(self.n_all_latents)  # --> [n_images, hid_d, n_neurons]
            acts_not_aggr_shape = [acts_not_aggr_shape[2], acts_not_aggr_shape[0], acts_not_aggr_shape[1]]

        self.activations_not_aggr = -torch.zeros(acts_not_aggr_shape, dtype=torch.bfloat16).to(device) # 4D or 3D
        self.is_setup = True


    # TODO: separate for each  model/ more general
    # update self.activations_not_aggr
    def update(self, acts: torch.Tensor, acts_not_aggr: torch.Tensor, sample_ids: torch.Tensor):
        """
        Update activations with a new batch, setting up tensors on first call if needed.

        Parameters
        ----------
        acts : torch.Tensor
            Aggregated activation tensor with shape (batch_size, n_latents).
        acts_not_aggr: torch.Tensor
            Tensor with not aggregated activations. Size differs depending on the model
        sample_ids : torch.Tensor
            Sample IDs corresponding to the activations.

        Notes
        -----
        If this is the first call and tensors haven't been set up, the number
        of latents will be inferred from the activation tensor shape.
        """
        if isinstance(acts_not_aggr, tuple):
          acts_not_aggr = acts_not_aggr[0]

        assert acts.ndim == 2
        if not self.is_setup:
            self.n_latents = acts.shape[1]

            if self.model_name == "qwen":
                # in qwen, batch size and hidden dimensions are flattened into one dim
                self.n_all_latents = [256, acts_not_aggr.shape[-1]]
            else:
                self.n_all_latents = list(acts_not_aggr.shape[1:])  # the first dim is n of images
            self._setup_tensors()

        batch_acts = acts.T.to(self.activations.dtype)  # --> shape: [n_neurons, batch]
        batch_sample_ids = sample_ids.repeat(self.n_latents, 1)

        if self.model_name == "resnet" or self.model_name == "vgg16" or self.model_name == "resnet_50":
            # [imgs, n_neurons, h, w] --> [n_neurons, imgs h, w]
            batch_acts_not_aggr = acts_not_aggr.transpose(1, 0).to(self.activations_not_aggr.dtype).to(device)
            #print(f"batch_acts_not_aggr: {batch_acts_not_aggr.shape}")

        elif self.model_name == "qwen":
            acts_not_aggr = acts_not_aggr.view(-1, 256, acts_not_aggr.shape[-1])  # b*hid, 1280 --> b, hid, n_neurons
            batch_acts_not_aggr = acts_not_aggr.permute(2, 0, 1).to(self.activations_not_aggr.dtype).to(device)

        else:
            # [imgs, hidden, n_neurons] --> [n_neurons, imgs, hidden]
            batch_acts_not_aggr = acts_not_aggr.permute(2, 0, 1).to(self.activations_not_aggr.dtype).to(device)


        # Find top-k among existing and new activations
        all_acts = torch.cat([self.activations, batch_acts], dim=1)
        all_ids = torch.cat([self.sample_ids, batch_sample_ids], dim=1)

        #print(f"activations_not_aggr: {self.activations_not_aggr.shape}")
        #print(f"batch_acts_not_aggr: {batch_acts_not_aggr.shape}")

        all_acts_not_aggr = torch.cat([self.activations_not_aggr, batch_acts_not_aggr], dim=1)

        self.activations, indices = torch.topk(all_acts, k=self.n_collect, dim=1)
        self.sample_ids = torch.gather(all_ids, dim=1, index=indices)

        # indices and input tensor should have the same shape
        if self.model_name == "resnet" or self.model_name == "vgg16" or self.model_name == "resnet_50":
            h, w = self.n_all_latents[-2],  self.n_all_latents[-2]
            indices_reshaped = indices.unsqueeze(-1).unsqueeze(-1).to(device)  # resnet: [1024, 10, 1, 1]
            indices_reshaped = indices_reshaped.expand(-1, -1, h, w)  # resnet [1024, 10, 14, 14]
        else:
            hid = self.n_all_latents[0] # n_all_latents = [hid, n_neurons]
            indices_reshaped = indices.unsqueeze(-1).to(device)  # [n_nueorns, 10, 1]
            indices_reshaped = indices_reshaped.expand(-1, -1, hid)


        self.activations_not_aggr = torch.gather(all_acts_not_aggr, dim=1, index=indices_reshaped).to(device)


    # store also activations_not_aggr
    def store(self, file_path: str | Path, metadata: dict[str, str] | None = None):
        """
        Store tensors and metadata to a safetensors file.

        Parameters
        ----------
        file_path : str or Path
            Path where the data should be saved.
        metadata : dict[str, str], optional
            Additional metadata to store with the tensors.

        Notes
        -----
        If the instance is not set up, the method will log a warning and skip storage.
        """
        if not self.is_setup:
            logger.warning("Attempted to store an un-initialized ActMax instance; skipping.")
            return

        tensors = {
            "activations": self.activations,
            "sample_ids": self.sample_ids,
            "not_aggr_activations": self.activations_not_aggr
        }
        safetensors.torch.save_file(tensors, file_path, metadata=metadata)
        logger.debug(f"Stored ActMax data to {file_path}")

    # change, so n_all_latent is also loaded
    @classmethod
    def load(cls, file_path: str | Path) -> ActMax:
        """
        Load an ActMax instance from a safetensors file.

        Parameters
        ----------
        file_path : str or Path
            Path to the safetensors file to load.

        Returns
        -------
        CustomActMax
            Loaded ActMax instance with data from the file.

        Raises
        ------
        ValueError
            If the file is missing required metadata for loading.
        """
        with safetensors.safe_open(file_path, framework="pt") as f:
            metadata = f.metadata()
            if metadata is None:
                raise ValueError(f"File {file_path} is missing required metadata for loading.")
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        import ast

        n_collect = int(metadata["n_collect"])
        n_latents = int(metadata["n_latents"])
        n_all_latents = ast.literal_eval(metadata["n_all_latents"])
        model_name = str(metadata["model_name"])

        instance = cls(model_name=model_name, n_collect=n_collect, n_latents=n_latents, n_all_latents=n_all_latents)

        instance.activations = tensors["activations"]
        instance.sample_ids = tensors["sample_ids"]
        instance.activations_not_aggr = tensors["not_aggr_activations"]

        return instance

class ActMaxCache(ActCache):
    """
    Collects, aggregates, and finds the top-k activations for specified layers.

    This class extends ActCache to not only collect activations but also aggregate
    them using a specified function and maintain only the top-k activating samples
    for each component in each layer.

    Parameters
    ----------
    layer_names : list of str
        Names of the model layers to analyze.
    aggregation_fn : callable
        Function to aggregate activations (e.g., mean over spatial dimensions).
        Must be a named function, not a lambda.
    n_collect : int
        Number of top activating samples to collect per component.

    Attributes
    ----------
    aggregation_fn : callable
        The aggregation function being used.
    n_collect : int
        Number of samples collected per component.
    sample_idx_counter : int
        Counter for tracking sample indices during data processing.
    agg_fn_name : str
        Name of the aggregation function for metadata.
    cache : dict[str, ActMax]
        Dictionary mapping layer names to ActMax instances.

    Raises
    ------
    ValueError
        If the aggregation function is a lambda or has no name.
    """

    def __init__(self, layer_names: list[str], aggregation_fn: Callable, n_collect: int):
        super().__init__(layer_names)
        self.aggregation_fn = aggregation_fn
        self.n_collect = n_collect
        self.sample_idx_counter = Counter()

        agg_fn_name = getattr(self.aggregation_fn, "__name__", None)
        if agg_fn_name is None or agg_fn_name == "<lambda>":
            raise ValueError("Aggregation function must be a defined function, not a lambda.")
        self.agg_fn_name = agg_fn_name

        self.cache: dict[str, ActMax] = {name: ActMax(n_collect=n_collect) for name in layer_names}

    def __getitem__(self, layer_name: str) -> ActMax:
        """
        Get the ActMax instance for a specific layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer to retrieve.

        Returns
        -------
        ActMax
            The ActMax instance associated with the specified layer.
        """
        return self.cache[layer_name]

    def __iter__(self):
        """Return an iterator over the ActMax instances in the cache."""
        return iter(self.cache.values())

    def _get_hook(self, layer_name: str) -> Callable:
        """
        Create a hook that aggregates activations and updates the ActMax instance.

        Parameters
        ----------
        name : str
            Name of the layer to create a hook for.

        Returns
        -------
        callable
            Hook function that processes layer outputs and updates the cache.
        """

        def hook_fn(module, ins, outs):
            aggregated_acts = self.aggregation_fn(outs)
            batch_size = aggregated_acts.shape[0]

            assert aggregated_acts.ndim == 2, "Something is wrong with the aggregation_fn"

            # Generate sample IDs for this batch
            sample_ids = torch.arange(
                self.sample_idx_counter[layer_name], self.sample_idx_counter[layer_name] + batch_size
            )
            self.sample_idx_counter[layer_name] += batch_size

            # Update the corresponding ActMax instance
            self.cache[layer_name].update(aggregated_acts, sample_ids)

        return hook_fn

    def __repr__(self) -> str:
        """Return string representation of the Actmaxcache instance."""
        agg_name = getattr(self.aggregation_fn, "__name__", "custom_function")
        return f"Actmaxcache(layers={list(self.layer_names)}, aggregation_fn='{agg_name}', n_collect={self.n_collect})"

    @property
    def metadata(self) -> dict[str, str]:
        """Returns metadata about the cache instance."""
        return dict(
            aggregation_fn_name=self.agg_fn_name,
            n_collect=str(self.n_collect),
            layer_names=str(list(self.cache.keys())),
        )

    def store(self, directory: Path | str):
        """
        Save the cache to a directory.

        Each layer's data is saved to a separate `.safetensors` file named
        `{layer_name}.{aggregation_function_name}.safetensors`.

        Parameters
        ----------
        directory : Path or str
            Directory where cache files will be saved.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for layer_name, act_max_instance in self.cache.items():
            if not act_max_instance.is_setup:
                logger.warning(f"Skipping layer '{layer_name}' as it has no data.")
                continue

            metadata = {
                "aggregation_fn_name": self.agg_fn_name,
                "n_collect": str(self.n_collect),
                "n_latents": str(act_max_instance.n_latents),
                "layer_name": layer_name,
            }
            # CHANGED: Filename now includes the aggregation function name to prevent overwrites.
            fname = "-".join([str(v) for k, v in metadata.items() if k not in ["n_latents"]]) + ".safetensors"
            filepath = directory / fname
            act_max_instance.store(filepath, metadata=metadata)

        logger.info(f"Cache saved successfully to {directory}")

    def load(self, directory: Path | str):
        """
        Load data into this cache instance from a directory.

        This method will only load files that match the instance's configured
        aggregation function and `n_collect` value.

        Parameters
        ----------
        directory : Path or str
            The directory containing the `.safetensors` files.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        ValueError
            If no matching cache files are found or if file metadata doesn't match.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Cache directory not found: {directory}")

        expected_agg_fn_name = self.aggregation_fn.__name__
        logger.info(f"Loading cache for aggregation fn: '{expected_agg_fn_name}'")

        loaded_count = 0
        for layer_name in self.layer_names:
            metadata = {
                "aggregation_fn_name": self.agg_fn_name,
                "n_collect": str(self.n_collect),
                "layer_name": layer_name,
            }
            # CHANGED: Filename now includes the aggregation function name to prevent overwrites.
            fname = "-".join([str(v) for k, v in metadata.items()]) + ".safetensors"

            fpath = directory / fname

            if not fpath.exists():
                logger.warning(f"File not found for layer '{layer_name}': {fpath}")
                raise FileNotFoundError(f"Expected file not found: {fpath}")

            try:
                with safetensors.safe_open(fpath, framework="pt") as f:
                    metadata = f.metadata()

                    if metadata.get("aggregation_fn_name") != expected_agg_fn_name:
                        raise ValueError(
                            f"Mismatch in aggregation function for layer '{layer_name}'. "
                            f"Expected '{expected_agg_fn_name}', but file has '{metadata.get('aggregation_fn_name')}'."
                        )
                    if int(metadata.get("n_collect")) != self.n_collect:
                        raise ValueError(
                            f"Mismatch in n_collect for layer '{layer_name}'. "
                            f"Expected '{self.n_collect}', but file has '{metadata.get('n_collect')}'."
                        )
            except ValueError as e:
                logger.warning(f"Validation failed for layer '{layer_name}': {e}")
                raise FileNotFoundError(f"Expected file not found: {fpath}")

            # If validation passes, load the data.
            self.cache[layer_name] = ActMax.load(fpath)
            loaded_count += 1

        if loaded_count == 0:
            logger.warning(f"No matching cache files were found and loaded from {directory}")
        else:
            logger.info(f"Successfully loaded data for {loaded_count} layer(s) from {directory}")

class CustomActMaxCache(ActMaxCache):
    """
    Collects, aggregates, and finds the top-k activations for specified layers.

    This class extends ActCache to not only collect activations but also aggregate
    them using a specified function and maintain only the top-k activating samples
    for each component in each layer.

    Parameters
    ----------
    layer_names : list of str
        Names of the model layers to analyze.
    aggregation_fn : callable
        Function to aggregate activations (e.g., mean over spatial dimensions).
        Must be a named function, not a lambda.
    n_collect : int
        Number of top activating samples to collect per component.

    Attributes
    ----------
    aggregation_fn : callable
        The aggregation function being used.
    n_collect : int
        Number of samples collected per component.
    sample_idx_counter : int
        Counter for tracking sample indices during data processing.
    agg_fn_name : str
        Name of the aggregation function for metadata.
    cache : dict[str, CustomActMax]
        Dictionary mapping layer names to ActMax instances.

    Raises
    ------
    ValueError
        If the aggregation function is a lambda or has no name.
    """

    # set cache values to CustomActMax instances
    def __init__(self, model_name, layer_names: list[str], aggregation_fn, n_collect: int):
        super().__init__(layer_names, aggregation_fn, n_collect)
        self.model_name = model_name

        self.cache: dict[str, CustomActMax] = {name: CustomActMax(n_collect=n_collect, model_name=self.model_name) for name in layer_names}


    # also add not aggregated activations to cache
    def _get_hook(self, layer_name: str):
        """
        Create a hook that aggregates activations and updates the ActMax instance.

        Parameters
        ----------
        name : str
            Name of the layer to create a hook for.

        Returns
        -------
        callable
            Hook function that processes layer outputs and updates the cache.
        """

        def hook_fn(module, ins, outs):
            aggregated_acts = self.aggregation_fn(outs)
            batch_size = aggregated_acts.shape[0]

            assert aggregated_acts.ndim == 2, "Something is wrong with the aggregation_fn"

            # Generate sample IDs for this batch
            sample_ids = torch.arange(
                self.sample_idx_counter[layer_name], self.sample_idx_counter[layer_name] + batch_size
            )
            self.sample_idx_counter[layer_name] += batch_size

            # Update the corresponding ActMax instance
            self.cache[layer_name].update(aggregated_acts, outs, sample_ids)

        return hook_fn

    # add n_all_latents to metadata
    def store(self, directory: Path | str):
        """
        Save the cache to a directory.

        Each layer's data is saved to a separate `.safetensors` file named
        `{layer_name}.{aggregation_function_name}.safetensors`.

        Parameters
        ----------
        directory : Path or str
            Directory where cache files will be saved.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for layer_name, act_max_instance in self.cache.items():
            if not act_max_instance.is_setup:
                logger.warning(f"Skipping layer '{layer_name}' as it has no data.")
                continue

            metadata = {
                "model_name": self.model_name,
                "aggregation_fn_name": self.agg_fn_name,
                "n_collect": str(self.n_collect),
                "n_latents": str(act_max_instance.n_latents),
                "n_all_latents": str(act_max_instance.n_all_latents),
                "layer_name": layer_name,
            }
            # CHANGED: Filename now includes the aggregation function name to prevent overwrites.
            fname = "-".join([str(v) for k, v in metadata.items() if k not in ["model_name", "n_latents", "n_all_latents"]]) + ".safetensors"
            filepath = directory / fname
            act_max_instance.store(filepath, metadata=metadata)

        logger.info(f"Cache saved successfully to {directory}")

    # change cache content to CustomActMax instance
    def load(self, directory: Path | str):
        """
        Load data into this cache instance from a directory.

        This method will only load files that match the instance's configured
        aggregation function and `n_collect` value.

        Parameters
        ----------
        directory : Path or str
            The directory containing the `.safetensors` files.

        Raises
        ------
        FileNotFoundError
            If the directory does not exist.
        ValueError
            If no matching cache files are found or if file metadata doesn't match.
        """
        # TODO: add a possible dir ActMax cache. Make sure, the info is compatible. Write a special reader for it
        directory = Path(directory)

        if not directory.is_dir():
            raise FileNotFoundError(f"Cache directory not found: {directory}")

        expected_agg_fn_name = self.aggregation_fn.__name__
        logger.info(f"Loading cache for aggregation fn: '{expected_agg_fn_name}'")

        loaded_count = 0
        for layer_name in self.layer_names:
            metadata = {
                "model_name": self.model_name,
                "aggregation_fn_name": self.agg_fn_name,
                "n_collect": str(self.n_collect),
                "layer_name": layer_name
            }
            # CHANGED: Filename now includes the aggregation function name to prevent overwrites.
            fname = "-".join([str(v) for k, v in metadata.items() if k not in ["model_name", "n_latents", "n_all_latents"]]) + ".safetensors"

            #fname = "-".join([str(v) for k, v in metadata.items()]) + ".safetensors"

            fpath = directory / fname

            if not fpath.exists():
                logger.warning(f"File not found for layer '{layer_name}': {fpath}")
                raise FileNotFoundError(f"Expected file not found: {fpath}")

            try:
                with safetensors.safe_open(fpath, framework="pt") as f:
                    metadata = f.metadata()

                    if metadata.get("aggregation_fn_name") != expected_agg_fn_name:
                        raise ValueError(
                            f"Mismatch in aggregation function for layer '{layer_name}'. "
                            f"Expected '{expected_agg_fn_name}', but file has '{metadata.get('aggregation_fn_name')}'."
                        )
                    if int(metadata.get("n_collect")) != self.n_collect:
                        raise ValueError(
                            f"Mismatch in n_collect for layer '{layer_name}'. "
                            f"Expected '{self.n_collect}', but file has '{metadata.get('n_collect')}'."
                        )
            except ValueError as e:
                logger.warning(f"Validation failed for layer '{layer_name}': {e}")
                raise FileNotFoundError(f"Expected file not found: {fpath}")

            # If validation passes, load the data.
            self.cache[layer_name] = CustomActMax.load(fpath)
            loaded_count += 1

        if loaded_count == 0:
            logger.warning(f"No matching cache files were found and loaded from {directory}")
        else:
            logger.info(f"Successfully loaded data for {loaded_count} layer(s) from {directory}")
