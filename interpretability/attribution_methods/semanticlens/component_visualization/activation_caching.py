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
import ast
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
import os
import re

from interpretability.adapters.semanticlens.base import PreparedBatch

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
    Store top-k aggregated activations and their corresponding
    canonical unaggregated activations.

    Canonical activation layout:

        [batch, neurons, *spatial]

    Internal cache layout:

        activations:
            [neurons, n_collect]

        sample_ids:
            [neurons, n_collect]

        activations_not_aggr:
            [neurons, n_collect, *spatial]

        spatial_grids:
            [neurons, n_collect, 2]

        The final dimension of spatial_grids contains:
            [grid_h, grid_w]
    """

    def __init__(
        self,
        model_name,
        n_collect: int,
        n_latents: int | None = None,
        n_all_latents: list | None = None,
        cache_device: torch.device | str | None = None,
    ):
        # Do not call ActMax.__init__, because it invokes
        # _setup_tensors() using the parent method signature.
        self.model_name = model_name
        self.n_collect = n_collect
        self.n_latents = n_latents
        self.n_all_latents = n_all_latents
        self.cache_device = (
            torch.device(cache_device)
            if cache_device is not None
            else None
        )

        self.is_setup = False
        self.activations = None
        self.sample_ids = None
        self.activations_not_aggr = None

    def _setup_tensors(
            self,
            n_latents: int,
            spatial_shape: tuple[int, ...],
            device: torch.device | str,
    ):
        self.n_latents = n_latents
        self.n_all_latents = list(spatial_shape)

        # Preserve previous initialization and dtype.
        self.activations = torch.zeros(
            (
                self.n_latents,
                self.n_collect,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

        self.sample_ids = torch.full(
            (
                self.n_latents,
                self.n_collect,
            ),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )

        self.activations_not_aggr = torch.zeros(
            (
                self.n_latents,
                self.n_collect,
                *spatial_shape,
            ),
            dtype=torch.bfloat16,
            device=device,
        )

        self.is_setup = True

    def update(
            self,
            aggregated_acts: torch.Tensor,
            canonical_acts: torch.Tensor,
            sample_ids: torch.Tensor,
    ):
        """
        aggregated_acts:
            [batch, neurons]

        canonical_acts:
            [batch, neurons, *spatial]
        """
        if aggregated_acts.ndim != 2:
            raise ValueError(
                "aggregated_acts must have shape [B, C], "
                f"got {tuple(aggregated_acts.shape)}"
            )

        if canonical_acts.ndim < 3:
            raise ValueError(
                "canonical_acts must have shape "
                "[B, C, *spatial], "
                f"got {tuple(canonical_acts.shape)}"
            )

        batch_size, n_latents = aggregated_acts.shape

        if canonical_acts.shape[:2] != (
                batch_size,
                n_latents,
        ):
            raise ValueError(
                "Aggregated and canonical activation shapes "
                "are incompatible: "
                f"{tuple(aggregated_acts.shape)} versus "
                f"{tuple(canonical_acts.shape)}"
            )

        spatial_shape = tuple(
            canonical_acts.shape[2:]
        )
        cache_device = (
            self.cache_device
            if self.cache_device is not None
            else canonical_acts.device
        )

        if not self.is_setup:
            self._setup_tensors(
                n_latents=n_latents,
                spatial_shape=spatial_shape,
                device=cache_device,
            )
        else:
            # Checkpoints are loaded on CPU. Move them once to the selected
            # cache device instead of copying the complete cache to and from
            # the model device for every batch.
            if self.activations.device != cache_device:
                self.activations = self.activations.to(cache_device)
                self.sample_ids = self.sample_ids.to(cache_device)
                self.activations_not_aggr = (
                    self.activations_not_aggr.to(cache_device)
                )

            cached_spatial_shape = tuple(
                self.activations_not_aggr.shape[2:]
            )

            if spatial_shape != cached_spatial_shape:
                raise ValueError(
                    "Activation shape changed between batches: "
                    f"expected {cached_spatial_shape}, "
                    f"got {spatial_shape}"
                )

        batch_aggregated = (
            aggregated_acts
            .transpose(0, 1)
            .to(
                device=cache_device,
                dtype=self.activations.dtype,
            )
        )

        batch_unaggregated = (
            canonical_acts
            .permute(
                1,
                0,
                *range(2, canonical_acts.ndim),
            )
            .to(
                device=cache_device,
                dtype=self.activations_not_aggr.dtype,
            )
        )

        batch_sample_ids = (
            sample_ids
            .to(cache_device)
            .unsqueeze(0)
            .expand(n_latents, -1)
        )

        all_aggregated = torch.cat(
            [
                self.activations,
                batch_aggregated,
            ],
            dim=1,
        )

        all_sample_ids = torch.cat(
            [
                self.sample_ids,
                batch_sample_ids,
            ],
            dim=1,
        )

        all_unaggregated = torch.cat(
            [
                self.activations_not_aggr,
                batch_unaggregated,
            ],
            dim=1,
        )

        top_activations, indices = torch.topk(
            all_aggregated,
            k=self.n_collect,
            dim=1,
        )

        index_shape = (
                list(indices.shape)
                + [1] * len(spatial_shape)
        )
        expanded_shape = (
                list(indices.shape)
                + list(spatial_shape)
        )

        gather_indices = (
            indices
            .reshape(index_shape)
            .expand(expanded_shape)
        )

        self.activations = top_activations.detach()

        self.sample_ids = torch.gather(
            all_sample_ids,
            dim=1,
            index=indices,
        ).detach()

        self.activations_not_aggr = torch.gather(
            all_unaggregated,
            dim=1,
            index=gather_indices,
        ).detach()

    def store(
        self,
        file_path: str | Path,
        metadata: dict[str, str] | None = None,
    ):
        if not self.is_setup:
            logger.warning(
                "Attempted to store an uninitialized "
                "CustomActMax instance; skipping."
            )
            return

        tensors = {
            "activations": (
                self.activations.detach().cpu().contiguous()
            ),
            "sample_ids": (
                self.sample_ids.detach().cpu().contiguous()
            ),
            "not_aggr_activations": (
                self.activations_not_aggr
                .detach()
                .cpu()
                .contiguous()
            ),
        }

        safetensors.torch.save_file(
            tensors,
            file_path,
            metadata=metadata,
        )

    @classmethod
    def load(
            cls,
            file_path: str | Path,
    ) -> CustomActMax:
        import ast

        with safetensors.safe_open(
                file_path,
                framework="pt",
        ) as file:
            metadata = file.metadata()

            if metadata is None:
                raise ValueError(
                    f"File {file_path} is missing metadata"
                )

            tensors = {
                key: file.get_tensor(key)
                for key in file.keys()
            }

        instance = cls(
            model_name=str(metadata["model_name"]),
            n_collect=int(metadata["n_collect"]),
            n_latents=int(metadata["n_latents"]),
            n_all_latents=ast.literal_eval(
                metadata["n_all_latents"]
            ),
        )

        instance.activations = tensors[
            "activations"
        ]
        instance.sample_ids = tensors[
            "sample_ids"
        ]
        instance.activations_not_aggr = tensors[
            "not_aggr_activations"
        ]

        instance.is_setup = True

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
    def __init__(
        self,
        adapter,
        model_name: str,
        layer_names: list[str],
        aggregation_fn,
        n_collect: int,
        cache_device: torch.device | str | None = None,
    ):
        if adapter is None:
            raise ValueError(
                "A SemanticLens adapter is required"
            )

        super().__init__(
            layer_names,
            aggregation_fn,
            n_collect,
        )

        self.adapter = adapter
        self.model_name = model_name
        self.cache_device = cache_device
        self.current_batch = None

        self.cache: dict[str, CustomActMax] = {
            layer_name: CustomActMax(
                model_name=model_name,
                n_collect=n_collect,
                cache_device=cache_device,
            )
            for layer_name in layer_names
        }

    @staticmethod
    def _safe_layer_name(layer_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", layer_name)

    def _store_tensors_atomic(
        self,
        tensors: dict[str, torch.Tensor],
        file_path: Path,
        metadata: dict[str, str],
    ) -> None:
        """
        Write a complete temporary file and then atomically replace
        the previous checkpoint.
        """
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        tmp_path = file_path.with_name(
            file_path.name + ".tmp"
        )

        cpu_tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in tensors.items()
        }

        try:
            safetensors.torch.save_file(
                cpu_tensors,
                tmp_path,
                metadata=metadata,
            )

            os.replace(
                tmp_path,
                file_path,
            )
        finally:
            # Clean up if save_file() or os.replace() failed.
            if tmp_path.exists():
                tmp_path.unlink()

    def store_checkpoint(
        self,
        directory: Path | str,
        *,
        neuron_chunk_size: int | None = None,
    ) -> None:
        """
        Store the current CustomActMax cache.

        Parameters
        ----------
        directory:
            Destination checkpoint directory.

        neuron_chunk_size:
            None writes one file per layer.

            A positive integer, for example 100, writes one file per
            group of that many neurons.
        """
        if (
            neuron_chunk_size is not None
            and neuron_chunk_size <= 0
        ):
            raise ValueError(
                "neuron_chunk_size must be positive or None"
            )

        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for layer_name, act_max in self.cache.items():
            if not act_max.is_setup:
                logger.warning(
                    "Skipping checkpoint for layer '%s' because "
                    "it has no data.",
                    layer_name,
                )
                continue

            safe_name = self._safe_layer_name(
                layer_name
            )
            n_latents = int(act_max.n_latents)

            base_metadata = {
                "model_name": self.model_name,
                "aggregation_fn_name": self.agg_fn_name,
                "n_collect": str(self.n_collect),
                "n_latents": str(n_latents),
                "n_all_latents": str(
                    act_max.n_all_latents
                ),
                "layer_name": layer_name,
                "samples_seen": str(
                    self.sample_idx_counter[layer_name]
                ),
                "cache_type": "CustomActMax",
            }

            if neuron_chunk_size is None:
                file_path = directory / (
                    f"{safe_name}."
                    f"{self.agg_fn_name}."
                    f"safetensors"
                )

                self._store_tensors_atomic(
                    tensors={
                        "activations": (
                            act_max.activations
                        ),
                        "sample_ids": (
                            act_max.sample_ids
                        ),
                        "not_aggr_activations": (
                            act_max.activations_not_aggr
                        ),
                    },
                    file_path=file_path,
                    metadata={
                        **base_metadata,
                        "neuron_start": "0",
                        "neuron_end": str(n_latents),
                    },
                )

                continue

            for start in range(
                0,
                n_latents,
                neuron_chunk_size,
            ):
                end = min(
                    start + neuron_chunk_size,
                    n_latents,
                )

                metadata = {
                    **base_metadata,
                    "neuron_start": str(start),
                    "neuron_end": str(end),
                }

                file_path = directory / (
                    f"{safe_name}."
                    f"{self.agg_fn_name}."
                    f"neurons-{start:06d}-{end:06d}."
                    f"safetensors"
                )

                self._store_tensors_atomic(
                    tensors={
                        "activations": (
                            act_max.activations[start:end]
                        ),
                        "sample_ids": (
                            act_max.sample_ids[start:end]
                        ),
                        "not_aggr_activations": (
                            act_max
                            .activations_not_aggr[start:end]
                        ),
                    },
                    file_path=file_path,
                    metadata=metadata,
                )

        logger.info(
            "Checkpoint saved successfully to %s",
            directory,
        )

    def set_current_batch( self, prepared_batch: PreparedBatch):
        self.current_batch = prepared_batch

    def _get_hook(self, layer_name):
        def hook_fn(module, inputs, output):
            if self.current_batch is None:
                raise RuntimeError(
                    "No prepared batch is associated with "
                    "the current forward pass"
                )

            canonical_acts = (
                self.adapter.canonicalize_activations(
                    output,
                    self.current_batch,
                )
            )

            aggregate_for_cache = getattr(
                self.adapter,
                "aggregate_for_cache",
                None,
            )

            if aggregate_for_cache is None:
                # Backward compatibility for adapters created before the
                # canonical-aggregation extension was introduced.
                aggregated_acts = self.aggregation_fn(output)
            else:
                aggregated_acts = aggregate_for_cache(
                    raw_activations=output,
                    canonical_activations=canonical_acts,
                    aggregation_fn=self.aggregation_fn,
                )

            batch_size = canonical_acts.shape[0]

            sample_ids = torch.arange(
                self.sample_idx_counter[layer_name],
                self.sample_idx_counter[layer_name]
                + batch_size,
                dtype=torch.int64,
            )

            self.sample_idx_counter[
                layer_name
            ] += batch_size

            self.cache[layer_name].update(
                aggregated_acts=aggregated_acts,
                canonical_acts=canonical_acts,
                sample_ids=sample_ids,
            )

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
    def load(
            self,
            directory: Path | str,
    ) -> None:
        """
        Load CustomActMax data from checkpoint files.

        Supported layouts
        -----------------
        Unsharded checkpoint:
            {layer}.{aggregation_fn}.safetensors

        Sharded checkpoint:
            {layer}.{aggregation_fn}.
            neurons-{start}-{end}.safetensors

        Legacy file:
            {aggregation_fn}-{n_collect}-{layer}.safetensors

        Sharded tensors are concatenated along the neuron dimension.
        The sample counter is restored from the `samples_seen`
        checkpoint metadata.
        """
        directory = Path(directory)

        if not directory.is_dir():
            raise FileNotFoundError(
                f"Cache directory not found: {directory}"
            )

        loaded_count = 0

        for layer_name in self.layer_names:
            safe_layer_name = self._safe_layer_name(
                layer_name
            )

            unsharded_path = directory / (
                f"{safe_layer_name}."
                f"{self.agg_fn_name}."
                f"safetensors"
            )

            shard_pattern = (
                f"{safe_layer_name}."
                f"{self.agg_fn_name}."
                f"neurons-*-*.safetensors"
            )

            shard_paths = list(
                directory.glob(shard_pattern)
            )

            # Preserve compatibility with files written by the
            # existing CustomActMaxCache.store() method.
            legacy_path = directory / (
                f"{self.agg_fn_name}-"
                f"{self.n_collect}-"
                f"{layer_name}."
                f"safetensors"
            )

            if unsharded_path.exists():
                checkpoint_paths = [
                    unsharded_path
                ]
                is_sharded = False

            elif shard_paths:
                checkpoint_paths = shard_paths
                is_sharded = True

            elif legacy_path.exists():
                checkpoint_paths = [
                    legacy_path
                ]
                is_sharded = False

            else:
                raise FileNotFoundError(
                    "No checkpoint found for layer "
                    f"'{layer_name}' in {directory}. "
                    f"Expected '{unsharded_path.name}' or "
                    f"files matching '{shard_pattern}'."
                )

            shard_records = []

            for checkpoint_path in checkpoint_paths:
                with safetensors.safe_open(
                        checkpoint_path,
                        framework="pt",
                ) as file:
                    metadata = file.metadata()

                    if metadata is None:
                        raise ValueError(
                            "Checkpoint has no metadata: "
                            f"{checkpoint_path}"
                        )

                    tensor_names = set(file.keys())

                    required_tensors = {
                        "activations",
                        "sample_ids",
                        "not_aggr_activations",
                    }

                    missing_tensors = (
                            required_tensors - tensor_names
                    )

                    if missing_tensors:
                        raise ValueError(
                            f"Checkpoint {checkpoint_path} "
                            "is missing tensors: "
                            f"{sorted(missing_tensors)}"
                        )

                    aggregation_name = metadata.get(
                        "aggregation_fn_name"
                    )

                    if (
                            aggregation_name
                            != self.agg_fn_name
                    ):
                        raise ValueError(
                            "Aggregation function mismatch in "
                            f"{checkpoint_path}: expected "
                            f"'{self.agg_fn_name}', got "
                            f"'{aggregation_name}'."
                        )

                    checkpoint_n_collect = int(
                        metadata["n_collect"]
                    )

                    if (
                            checkpoint_n_collect
                            != self.n_collect
                    ):
                        raise ValueError(
                            f"n_collect mismatch in "
                            f"{checkpoint_path}: expected "
                            f"{self.n_collect}, got "
                            f"{checkpoint_n_collect}."
                        )

                    checkpoint_layer = metadata.get(
                        "layer_name"
                    )

                    if (
                            checkpoint_layer is not None
                            and checkpoint_layer != layer_name
                    ):
                        raise ValueError(
                            f"Layer mismatch in "
                            f"{checkpoint_path}: expected "
                            f"'{layer_name}', got "
                            f"'{checkpoint_layer}'."
                        )

                    checkpoint_model = metadata.get(
                        "model_name"
                    )

                    if (
                            checkpoint_model is not None
                            and checkpoint_model
                            != self.model_name
                    ):
                        raise ValueError(
                            f"Model mismatch in "
                            f"{checkpoint_path}: expected "
                            f"'{self.model_name}', got "
                            f"'{checkpoint_model}'."
                        )

                    activations = file.get_tensor(
                        "activations"
                    )

                    sample_ids = file.get_tensor(
                        "sample_ids"
                    )

                    activations_not_aggr = (
                        file.get_tensor(
                            "not_aggr_activations"
                        )
                    )

                    # New checkpoints contain explicit ranges.
                    # Legacy files do not, so infer their complete
                    # range from the stored tensor.
                    neuron_start = int(
                        metadata.get(
                            "neuron_start",
                            0,
                        )
                    )

                    neuron_end = int(
                        metadata.get(
                            "neuron_end",
                            neuron_start
                            + activations.shape[0],
                        )
                    )

                    total_n_latents = int(
                        metadata["n_latents"]
                    )

                    raw_n_all_latents = metadata.get(
                        "n_all_latents"
                    )

                    if raw_n_all_latents is None:
                        n_all_latents = list(
                            activations_not_aggr.shape[2:]
                        )
                    else:
                        n_all_latents = ast.literal_eval(
                            raw_n_all_latents
                        )

                    samples_seen_raw = metadata.get(
                        "samples_seen"
                    )

                    samples_seen = (
                        int(samples_seen_raw)
                        if samples_seen_raw is not None
                        else None
                    )

                expected_shard_size = (
                        neuron_end - neuron_start
                )

                if expected_shard_size <= 0:
                    raise ValueError(
                        "Invalid neuron range in "
                        f"{checkpoint_path}: "
                        f"[{neuron_start}, {neuron_end})."
                    )

                if (
                        activations.shape[0]
                        != expected_shard_size
                ):
                    raise ValueError(
                        f"Neuron range in {checkpoint_path} "
                        f"has size {expected_shard_size}, but "
                        f"the activation tensor contains "
                        f"{activations.shape[0]} neurons."
                    )

                if (
                        sample_ids.shape[:2]
                        != activations.shape[:2]
                ):
                    raise ValueError(
                        "Activation and sample-ID shapes are "
                        f"incompatible in {checkpoint_path}: "
                        f"{tuple(activations.shape)} versus "
                        f"{tuple(sample_ids.shape)}."
                    )

                if (
                        activations_not_aggr.shape[:2]
                        != activations.shape[:2]
                ):
                    raise ValueError(
                        "Aggregated and unaggregated activation "
                        f"shapes are incompatible in "
                        f"{checkpoint_path}: "
                        f"{tuple(activations.shape)} versus "
                        f"{tuple(activations_not_aggr.shape)}."
                    )

                shard_records.append(
                    {
                        "path": checkpoint_path,
                        "start": neuron_start,
                        "end": neuron_end,
                        "total_n_latents": (
                            total_n_latents
                        ),
                        "n_all_latents": (
                            list(n_all_latents)
                        ),
                        "samples_seen": samples_seen,
                        "activations": activations,
                        "sample_ids": sample_ids,
                        "activations_not_aggr": (
                            activations_not_aggr
                        ),
                    }
                )

            # Filename sorting is normally sufficient because the
            # range is zero-padded, but metadata sorting is safer.
            shard_records.sort(
                key=lambda record: record["start"]
            )

            first_record = shard_records[0]

            expected_n_latents = first_record[
                "total_n_latents"
            ]
            expected_n_all_latents = first_record[
                "n_all_latents"
            ]
            expected_samples_seen = first_record[
                "samples_seen"
            ]

            expected_start = 0

            for record in shard_records:
                if record["start"] != expected_start:
                    raise ValueError(
                        "Missing or overlapping neuron shards "
                        f"for layer '{layer_name}': expected "
                        f"the next shard to start at "
                        f"{expected_start}, but "
                        f"{record['path'].name} starts at "
                        f"{record['start']}."
                    )

                if (
                        record["total_n_latents"]
                        != expected_n_latents
                ):
                    raise ValueError(
                        "Inconsistent n_latents metadata for "
                        f"layer '{layer_name}'."
                    )

                if (
                        record["n_all_latents"]
                        != expected_n_all_latents
                ):
                    raise ValueError(
                        "Inconsistent n_all_latents metadata "
                        f"for layer '{layer_name}'."
                    )

                if (
                        expected_samples_seen is not None
                        and record["samples_seen"]
                        != expected_samples_seen
                ):
                    raise ValueError(
                        "Checkpoint shards for layer "
                        f"'{layer_name}' were written at "
                        "different processing positions."
                    )

                expected_start = record["end"]

            if expected_start != expected_n_latents:
                raise ValueError(
                    f"Incomplete checkpoint for layer "
                    f"'{layer_name}': loaded neurons "
                    f"[0, {expected_start}), but expected "
                    f"[0, {expected_n_latents})."
                )

            activations = torch.cat(
                [
                    record["activations"]
                    for record in shard_records
                ],
                dim=0,
            )

            sample_ids = torch.cat(
                [
                    record["sample_ids"]
                    for record in shard_records
                ],
                dim=0,
            )

            activations_not_aggr = torch.cat(
                [
                    record["activations_not_aggr"]
                    for record in shard_records
                ],
                dim=0,
            )

            instance = CustomActMax(
                model_name=self.model_name,
                n_collect=self.n_collect,
                n_latents=expected_n_latents,
                n_all_latents=expected_n_all_latents,
                cache_device=self.cache_device,
            )

            instance.activations = (
                activations.contiguous()
            )
            instance.sample_ids = (
                sample_ids.contiguous()
            )
            instance.activations_not_aggr = (
                activations_not_aggr.contiguous()
            )
            instance.is_setup = True

            self.cache[layer_name] = instance

            if expected_samples_seen is not None:
                self.sample_idx_counter[
                    layer_name
                ] = expected_samples_seen
            else:
                # Legacy checkpoints have no samples_seen
                # metadata. This is only an approximation because
                # the top-k cache may not contain the latest ID.
                valid_ids = sample_ids[
                    sample_ids >= 0
                    ]

                self.sample_idx_counter[
                    layer_name
                ] = (
                    int(valid_ids.max().item()) + 1
                    if valid_ids.numel() > 0
                    else 0
                )

                logger.warning(
                    "Checkpoint for layer '%s' has no "
                    "'samples_seen' metadata. The restored "
                    "sample counter was inferred from retained "
                    "top-k IDs and may be too small.",
                    layer_name,
                )

            loaded_count += 1

            logger.info(
                "Loaded layer '%s' from %d checkpoint "
                "file(s)%s.",
                layer_name,
                len(shard_records),
                " using neuron shards"
                if is_sharded
                else "",
            )

        logger.info(
            "Successfully loaded checkpoints for %d layer(s) "
            "from %s.",
            loaded_count,
            directory,
        )
