"""Activation-based component visualization for neural network analysis.

This module provides the `ActivationComponentVisualizer`, a tool for identifying
the concept examples that most strongly activate specific components (e.g., neurons,
channels) of a neural network. It works by performing a forward pass over a dataset,
caching the activations for specified layers, and identifying the top-k activating
input samples for each component.
For component visualization the full act-max samples are returned.

"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import copy
import cv2
import os

import torch
from torch.utils.data import DataLoader
from torch import nn

from configs.general_cofig import *

from transformers import AutoProcessor, ViltProcessor

from . import aggregators
from .activation_caching import CustomActMaxCache
from .base import AbstractComponentVisualizer
from ..utils.helper import (
    get_fallback_name,
    restore_crop,
)
from ..scores import polysemanticity_score

logger = logging.getLogger(__name__)


class MissingNameWarning(UserWarning):
    """
    Warning raised when a model or dataset is missing a `.name` attribute.

    This attribute is crucial for the caching mechanism to create a stable and predictable cache location.
    Without it, a fallback name is generated.
    """

    pass


class ActivationComponentVisualizer(AbstractComponentVisualizer):
    """Finds and visualizes concepts based on activation maximization.

    This class implements the activation-based approach to component
    visualization. It processes a dataset to find the input examples that
    produce the highest activation values for each component within specified
    layers of a neural network.

    The results, including the indices of the top-activating samples, are
    cached to disk for efficient re-use in subsequent analyses.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to analyze. It is recommended that the model
        has a `.name` attribute for reliable caching.
    dataset_model : torch.utils.data.Dataset
        The dataset used for model inference to find top-activating samples.
        It should be preprocessed as required by the model. It is recommended
        that the dataset has a `.name` attribute for reliable caching.
    dataset_fm : torch.utils.data.Dataset
        The dataset preprocessed for the foundation model. This version should
        yield raw data (e.g., PIL Images) that the foundation model's own
        preprocessor can handle.
    layer_names : list[str]
        A list of names of the layers to analyze (e.g., `['layer4.1.conv2']`).
    num_samples : int
        The number of top-activating samples to collect for each component.
    device : torch.device or str, optional
        The device on which to perform computations. If None, the model's
        current device is used.
    aggregate_fn : callable, optional
        A function to aggregate the spatial or temporal dimensions of the layer
        activations into a single value per component. If None, defaults to
        taking the mean over spatial dimensions for convolutional layers.
        (A selection of aggregation functions are provided in
        `semanticlens.component_visualization.aggregators`.)
    cache_dir : str or None, optional
        The root directory for caching results. If None, caching is disabled.

    Attributes
    ----------
    actmax_cache : ActMaxCache
        An object that manages the collection and caching of top activations.

    Raises
    ------
    ValueError
        If any layer name in `layer_names` is not found in the model.

    Examples
    --------
    >>> import torch
    >>> from torchvision.models import resnet18
    >>> from torch.utils.data import TensorDataset
    >>> from semanticlens.component_visualization import ActivationComponentVisualizer
    >>>
    >>> # 1. Prepare model and dataset
    >>> model = resnet18(weights=...)
    >>> model.name = "resnet18"
    >>> dummy_data = TensorDataset(torch.randn(100, 3, 224, 224))
    >>> dummy_data.name = "dummy_data"
    >>>
    >>> # 2. Initialize the visualizer
    >>> visualizer = ActivationComponentVisualizer(
    ...     model=model,
    ...     dataset_model=dummy_data,
    ...     dataset_fm=dummy_data, # Using same dataset for simplicity here
    ...     layer_names=["layer4.1.conv2"],
    ...     num_samples=10,
    ...     cache_dir="./cache"
    ... )
    >>>
    >>> # 3. Run the analysis to find top-activating samples
    >>> # This will process the dataset and save the results to the cache.
    >>> # visualizer.run(batch_size=32)
    """

    AGGREGATION_DEFAULTS = {
        "mean": aggregators.aggregate_conv_mean,
        "max": aggregators.aggregate_conv_max,
    }

    def __init__(
        self,
        model: nn.Module,
        dataset_model,
        dataset_fm,
        layer_names: list[str],
        num_samples: int,
        device=None,
        adapter=None,
        aggregate_fn=None,
        cache_dir: str | None = None,
    ):
        """
        Initialize the ActivationComponentVisualizer.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to analyze.
        dataset_model : torch.utils.data.Dataset
            Dataset for model inference and activation collection.
        dataset_fm : torch.utils.data.Dataset
            Dataset preprocessed for foundation model encoding.
        layer_names : list of str
            Names of the layers to analyze for component visualization.
        num_samples : int
            Number of top activating samples to collect per component.
        device : torch.device or str, optional
            Device for computations. If None, uses model's device.
        aggregate_fn : callable, optional
            Function for aggregating activations. If None, uses default conv mean aggregation.
        cache_dir : str or None, optional
            Directory for caching results. If None, results will not be cached.

        Raises
        ------
        ValueError
            If any layer in layer_names is not found in the model.
        """
        self.model = model
        self.adapter = adapter
        self.dataset = dataset_model
        self.dataset_fm = dataset_fm
        self._init_cache_dir(cache_dir)
        self._validate_args()

        self.layer_names = layer_names
        self._check_layers()

        device = device or next(model.parameters()).device
        self.model.to(device)

        if aggregate_fn is None:
            logger.warning(f"No aggregation_fn provided using default: {aggregators.aggregate_conv_mean.__name__}")
            aggregate_fn = aggregators.aggregate_conv_mean

        self.actmax_cache = CustomActMaxCache(
            adapter=adapter,
            model_name=self.model.name,
            layer_names=self.layer_names,
            aggregation_fn=aggregate_fn,
            n_collect=num_samples,
        )

        if self.caching:
            try:
                self.actmax_cache.load(self.storage_dir)
                logger.info(f"Results loaded from {self.storage_dir}")
            except FileNotFoundError:
                logger.info(f"Results will be stored in {self.storage_dir}")

    def _validate_args(self):
        """For caching we need names for model and dataset.
        They are supposed to be provided as instance-attributes.
        If they are missing we use a fallback that is a combination of their class-name and a hash of their printable representation (`repr()`).
        """
        if not hasattr(self.model, "name"):
            model_name = get_fallback_name(self.model)
            if self.caching:
                message = (
                    f"Model does not have a name attribute, which is required for reliable caching.\n"
                    f"Using a fallback name: {model_name}."
                )
                warnings.warn(message, MissingNameWarning, stacklevel=2)
            self.model.name = model_name
        if not hasattr(self.dataset, "name"):
            dataset_name = get_fallback_name(self.dataset)
            if self.caching:
                message = (
                    f"Dataset does not have a name attribute, which is required for reliable caching.\n"
                    f"Using a fallback name: {dataset_name}."
                )
                warnings.warn(message, MissingNameWarning, stacklevel=2)
            self.dataset.name = dataset_name

        if len(self.dataset) != len(self.dataset_fm):
            raise ValueError(
                "Model and foundation model my_datasets should have the same length.",
                (len(self.dataset), len(self.dataset_fm)),
            )

    def _check_layers(self):
        """
        Validate that all specified layers exist in the model.

        Raises
        ------
        ValueError
            If any layer in self.layer_names is not found in the model.
        """
        for layer in self.layer_names:
            if layer not in dict(self.model.named_modules()):
                raise ValueError(f"Layer '{layer}' not found in model.")

    def _init_cache_dir(self, cache_dir):
        """
        Initialize the cache directory for storing results.

        Parameters
        ----------
        cache_dir : str or None
            Directory path for caching. If None, caching is disabled.
        """
        if cache_dir is None:
            logger.warning("No cache dir provided. Results will not be cached!")
            self._cache_root = None
        else:
            self._cache_root = Path(cache_dir)
            self._cache_root.mkdir(parents=True, exist_ok=True)

    @property
    def device(self):
        """
        Get the device of the model.

        Returns
        -------
        torch.device
            The device where the model parameters are located.
        """
        return next(self.model.parameters()).device

    def to(self, device):
        """
        Move the model to the specified device.

        Parameters
        ----------
        device : torch.device or str
            The target device to move the model to.

        Returns
        -------
        torch.nn.Module
            The model after being moved to the specified device.
        """
        return self.model.to(device)

    @property
    def caching(self) -> bool:
        """Check if caching is enabled."""
        return self._cache_root is not None

    @property
    def storage_dir(self):
        """
        Get the directory for storing concept visualization cache.

        Returns
        -------
        pathlib.Path
            Path to the storage directory for this visualizer instance.

        Raises
        ------
        AssertionError
            If no cache directory was provided during initialization.
        """
        assert self._cache_root, "No cache dir provided"
        return self._cache_root / self.__class__.__name__ / self.dataset.name / self.model.name

    @property
    def metadata(self) -> dict[str, str]:
        """
        Get metadata about the visualization instance.

        Returns
        -------
        dict[str, str]
            Dictionary containing metadata about the cache, dataset, and model.
        """
        return {**self.actmax_cache.metadata, "dataset": self.dataset.name, "model": self.model.name}

    def run(self, batch_size=32, num_workers=0):
        """Run the activation maximization analysis on the dataset.

        This method processes the entire `dataset_model` to find the maximally
        activating input examples for each component in the specified layers.
        If a valid cache is found, the results are loaded directly from disk,
        skipping the computation. Otherwise, the computation is performed and
        the results are saved to the cache.

        Parameters
        ----------
        batch_size : int, default=32
            The batch size to use for processing the dataset.
        num_workers : int, default=0
            The number of worker processes for the data loader.

        Returns
        -------
        dict
            A dictionary mapping layer names to `ActMax` instances, which
            contain the top activating samples for each component.
        """
        if self._cache_root is None:
            logger.debug("No cache root provided, running computation...")
            return self._run(batch_size=batch_size, num_workers=num_workers)
        try:
            self.actmax_cache.load(self.storage_dir)
            return self.actmax_cache.cache
        except FileNotFoundError:
            logger.debug(f"Activation maximization cache not found at {self.storage_dir}. Running computation...")
            return self._run(batch_size=batch_size, num_workers=num_workers)

    @torch.no_grad()
    def _run(self, batch_size: int = 64, num_workers: int = 0):
        """Actual ActMax-Cache computation/population and caching."""
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        with self.actmax_cache.hook_context(self.model):
            for i, (images, _) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Collecting ActMax"):
                print("fixed forward run")
                _ = self.model(images.to(self.device)).cpu()

        if self._cache_root:
            self.actmax_cache.store(self.storage_dir)
            logger.debug(f"Stored activation maximization cache at {self.storage_dir}")

        return self.actmax_cache.cache

    @torch.no_grad()
    def _compute_concept_db(self, fm, batch_size=32, **kwargs):
        """
        Compute the concept database for the given fm (foundation model).

        This is called from the Lens class following the Inversion of Control pattern.
        The method processes the dataset to find maximally activating samples and then
        embeds those samples using the foundation model.

        Parameters
        ----------
        fm : FoundationModel
            The foundation model used for embedding the maximally activating samples.
        batch_size : int, default=32
            Batch size for processing.
        **kwargs
            Additional keyword arguments passed to run() and embedding methods.

        Returns
        -------
        dict
            Dictionary mapping layer names to embedded concept representations.
        """
        self.run(batch_size=batch_size, **kwargs)

        embeds = self._embed_vision_dataset(fm, batch_size, **kwargs)

        concept_db = dict()
        for layer_name in self.layer_names:
            concept_db[layer_name] = embeds[self.get_max_reference(layer_name)]
        return concept_db

    def _embed_vision_dataset(self, fm, batch_size, **kwargs):
        """
        Embed the vision dataset using the provided foundation model.

        Parameters
        ----------
        fm : FoundationModel
            Foundation model with encode_image method for embedding images.
        batch_size : int
            Batch size for processing the dataset.
        **kwargs
            Additional keyword arguments passed to DataLoader.

        Returns
        -------
        torch.Tensor
            Tensor of shape (dataset_size, embedding_dim) containing embeddings
            for all samples in the dataset.

        """
        fm.to(self.device)

        def pil_list_collate(batch):
            """We apply the FM transformation (via fm.preprocess) lazy, thus the dataset_fm returns PILs and a special collate implementation is needed."""
            if isinstance(batch[0], (tuple, list)):
                return [item[0] for item in batch]
            return list(batch)

        pil_dataloader = torch.utils.data.DataLoader(
            self.dataset_fm, batch_size=batch_size, shuffle=False, collate_fn=pil_list_collate, **kwargs
        )
        embeds = []
        with tqdm(total=len(self.dataset), desc="Embedding Dataset") as pbar_dataset:
            for pil_list in pil_dataloader:
                inputs = fm.preprocess(pil_list)
                fm_out = fm.encode_image(inputs).cpu()
                embeds.append(fm_out)
                pbar_dataset.update(batch_size)
        embeds = torch.cat(embeds)

        assert embeds.shape[0] == len(self.dataset_fm), "Number of embeddings does not match number of ids!"
        return embeds

    def get_max_reference(self, layer_name) -> torch.Tensor:
        """
        Get sample IDs of maximally activating samples for a layer.

        Parameters
        ----------
        layer_name : str
            Name of the layer to get sample IDs for.

        Returns
        -------
        torch.Tensor
            Tensor of shape (n_components, n_samples) containing the dataset
            indices of maximally activating samples for each component.
        """
        self._check_layer_name(layer_name)
        return self.actmax_cache.cache[layer_name].sample_ids

    def visualize_components(
        self,
        component_ids: torch.Tensor,
        layer_name: str,
        n_samples: int = 9,
        nrows: int = 3,
        fname=None,
        denormalization_fn=None,
    ):
        """
        Visualize specific components by displaying their top activating samples.

        A good place to put it here since we need access to the PIL-dataset and actmax cache to implement this. However we should call a stateless function in here that abstracts complexity and can be used by other versions of the concept visualizer as well.

        Parameters
        ----------
        component_ids : torch.Tensor
            IDs of the components to visualize.
        layer_name : str
            Name of the layer containing the components.
        n_samples : int, default=9
            Number of top activating samples to display per component.
        nrows : int, default=3
            Number of rows in the grid layout for each component.
        denormalization_fn : callable, optional
            Function to denormalize the images before visualization.
        """
        self._check_layer_name(layer_name)
        import matplotlib.pyplot as plt
        from torchvision.utils import make_grid

        if hasattr(self.dataset, "denormalization_fn"):
            post_process = self.dataset.denormalization_fn
        elif denormalization_fn is not None:
            post_process = denormalization_fn
        else:
            logger.debug("Dataset does not have denormalization_fn method.")

            def post_process(x):
                return x

        pics = []
        for component_id in component_ids:
            ids = self.get_max_reference(layer_name=layer_name)[component_id][:n_samples]
            pics.append(
                make_grid(
                    [post_process(self.dataset[i][0]) for i in ids],
                    nrow=nrows,
                )
                .permute(1, 2, 0)
                .numpy()
            )
        n_pics = len(pics)

        n_cols = int(n_pics**0.5)
        n_rows = (n_pics + n_cols - 1) // n_cols

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))

        if n_pics == 1:
            axs = [axs]
        elif n_rows == 1 or n_cols == 1:
            axs = axs.flatten()
        else:
            axs = axs.flatten()

        for i, pic in enumerate(pics):
            axs[i].imshow(pic)
            axs[i].set_title(f"Neuron {component_ids[i]}")
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        for i in range(n_pics, len(axs)):
            axs[i].axis("off")

        plt.suptitle((f"{fname:.15} " if fname else "") + f"{self.model.name:>.10} {layer_name:<.15}", fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        if self.caching:
            component_id_str = "-".join(map(str, component_ids.tolist()))
            fdir = self.storage_dir / "plots"
            fdir.mkdir(parents=True, exist_ok=True)
            fpath = fdir / ((fname + "_" if fname else "") + f"{layer_name}_{component_id_str}.png")
            plt.savefig(fpath)
            plt.close(fig)
            print(f"Saved visualization to {fpath}")
        elif fname:
            logger.warning(
                "Failed to save visualization. Caching is not enabled in the ComponentVisualizer (`cv.caching: False`)"
            )

    def _check_layer_name(self, layer_name):
        """
        Validate that a layer name exists in the configured layers.

        Parameters
        ----------
        layer_name : str
            Name of the layer to validate.

        Raises
        ------
        ValueError
            If the layer name is not found in self.layer_names.
        """
        if layer_name not in self.layer_names:
            raise ValueError(f"Layer '{layer_name}' not found in model layers: {self.layer_names}")

class CustomActivationComponentVisualizer(ActivationComponentVisualizer):
    @torch.no_grad()
    def __init__(
            self,
            model,
            dataset_model,
            dataset_fm,
            layer_names,
            num_samples,
            device=None,
            adapter=None,
            aggregate_fn=None,
            cache_dir=None,
            model_name=None,
    ):
        if adapter is None:
            raise ValueError(
                "CustomActivationComponentVisualizer "
                "requires an adapter"
            )

        if model_name is not None:
            model.name = model_name

        super().__init__(
            model=model,
            dataset_model=dataset_model,
            dataset_fm=dataset_fm,
            layer_names=layer_names,
            num_samples=num_samples,
            device=device,
            adapter=adapter,
            aggregate_fn=aggregate_fn,
            cache_dir=cache_dir,
        )

        self.model_name = self.model.name
        self.component_ids = None

        self.top_k_per_layer = None
        self.top_k_max_samples = None

        self.crop_info = None
        self.crop_db = None
        self.neurons_crops_mapping = None
        self.embeds = None
        self.concept_db = None


    def _init_cache_dir(self, cache_dir):
        """
        Initialize the cache directory for storing results.

        Parameters
        ----------
        cache_dir : str or None
            Directory path for caching. If None, caching is disabled.
        """
        if cache_dir is None:
            print(f"Provided cache dir: {cache_dir}")
            logger.warning("No cache dir provided. Results will not be cached!")
            self._cache_root = None
        else:
            self._cache_root = Path(cache_dir)
            self._cache_root.mkdir(parents=True, exist_ok=True)


    @torch.no_grad()
    # override to pass the inputs in the form needed for ViLT and Qwen-VL
    @torch.no_grad()
    def _run(
            self,
            batch_size: int = 64,
            num_workers: int = 0,
    ):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        with self.actmax_cache.hook_context(
                self.model
        ):
            for images, _ in tqdm(
                    dataloader,
                    total=len(dataloader),
                    desc="Collecting ActMax",
            ):
                prepared_batch = (
                    self.adapter.prepare_batch(
                        images,
                        self.device,
                    )
                )

                self.actmax_cache.set_current_batch(
                    prepared_batch
                )

                # Exactly one forward pass.
                self.adapter.forward(
                    self.model,
                    prepared_batch,
                )

        self.actmax_cache.current_batch = None

        if self._cache_root:
            self.actmax_cache.store(
                self.storage_dir
            )

        return self.actmax_cache.cache


    # add the possibility to download cache for max act from a given path
    def run(self, batch_size=32, num_workers=0):
        """Run the activation maximization analysis on the dataset.

        This method processes the entire `dataset_model` to find the maximally
        activating input examples for each component in the specified layers.
        If a valid cache is found, the results are loaded directly from disk,
        skipping the computation. Otherwise, the computation is performed and
        the results are saved to the cache.

        Parameters
        ----------
        batch_size : int, default=32
            The batch size to use for processing the dataset.
        num_workers : int, default=0
            The number of worker processes for the data loader.

        Returns
        -------
        dict
            A dictionary mapping layer names to `ActMax` instances, which
            contain the top activating samples for each component.
        """
        if self._cache_root is None:
            logger.debug("No cache root provided, running computation...")
            return self._run(batch_size=batch_size, num_workers=num_workers)
        try:
            self.actmax_cache.load(self.storage_dir)
            return self.actmax_cache.cache
        except FileNotFoundError:
            '''try:
                # TODO: change path if needed
                self.actmax_cache.load("semanticlens_cache/CustomActivationComponentVisualizer/Subset-13079351281979546919510671421896698979616609685017560695586993513236229725530/Qwen2-VL-2B-Instruct")
                print("Loaded act max from cache")
                print("shape of actmax_cache", self.actmax_cache.cache['model.visual.blocks.12'].activations.shape)
                return self.actmax_cache.cache
            except:'''
            logger.debug(f"Activation maximization cache not found at {self.storage_dir}")

            return self._run(batch_size=batch_size, num_workers=num_workers)

    # TODO: general. Also adapt for relevance scores?
    def find_top_act_neurons(
            self,
            topk_neurons: int,
    ) -> tuple[dict, dict]:
        """
        Find the most activating neurons in a model.

        Returns
        -------
        top_k_per_layer
            Dictionary mapping each layer name to its selected neurons
            and their mean activations.

        top_k_max_samples
            Dictionary mapping each layer name and neuron ID to the
            IDs of its maximally activating reference samples.
        """
        print("Finding top-k neurons")

        topk_n_path = (
            f"{OUTPUTS}/top_k_act_neurons/"
            f"top_{topk_neurons}_per_layer_{self.model_name}.pt"
        )
        top_samples_path = (
            f"{OUTPUTS}/top_k_act_neurons/"
            f"top_{topk_neurons}_max_samples_{self.model_name}.pt"
        )

        # Load existing files only if their layer names are compatible
        # with the layers used by the current visualizer.
        if (
                os.path.exists(topk_n_path)
                and os.path.exists(top_samples_path)
        ):
            print("Loading top-k neurons from existing files")

            top_k_per_layer = torch.load(
                topk_n_path,
                map_location="cpu",
                weights_only=False,
            )
            top_k_max_samples = torch.load(
                top_samples_path,
                map_location="cpu",
                weights_only=False,
            )

            cached_layers = (
                    set(top_k_per_layer.keys())
                    | set(top_k_max_samples.keys())
            )
            current_layers = set(self.layer_names)

            unknown_cached_layers = (
                    cached_layers - current_layers
            )

            if not unknown_cached_layers:
                self.top_k_per_layer = top_k_per_layer
                self.top_k_per_layer_filtered = (
                    top_k_per_layer
                )
                self.top_k_max_samples = (
                    top_k_max_samples
                )

                return (
                    top_k_per_layer,
                    top_k_max_samples,
                )

            print(
                "The existing top-k cache is incompatible "
                "with the current layer names."
            )
            print(
                "Unknown cached layers:",
                unknown_cached_layers,
            )
            print("Recomputing the top-k neurons...")

        def _find_neurons_of_layer(
                layer_name,
                start,
                end,
        ):
            for idx, neuron_idx in enumerate(top_indices):
                neuron_idx = neuron_idx.item()

                if start <= neuron_idx < end:
                    local_idx = neuron_idx - start

                    top_k_per_layer[layer_name][
                        local_idx
                    ] = float(top_values[idx])

        act_max_cache = self.actmax_cache

        # Collect activations from the different layers so that
        # the neurons can be sorted globally.
        activations_all_layers = []

        # Store the number of neurons in each layer. This is later
        # used to map global neuron indices back to local indices.
        layers_lens = dict.fromkeys(
            act_max_cache.layer_names
        )

        for layer_name in act_max_cache.layer_names:
            activations_all_layers.append(
                act_max_cache[layer_name].activations
            )

            layers_lens[layer_name] = len(
                act_max_cache[layer_name].activations
            )

        activations_all_layers = torch.cat(
            activations_all_layers,
            dim=0,
        )

        # compute the mean activation of each neuron over its maximally activating samples.
        activations_mean = torch.mean(
            activations_all_layers,
            dim=1,
        )

        # choose the globally most active neurons.
        top_k = torch.topk(
            activations_mean,
            k=topk_neurons,
            dim=0,
        )

        k_actual = (
                top_k.values > 0
        ).sum().item()

        print(
            f"{k_actual} neurons have activation over 0. "
            f"Analysing {k_actual} neurons further."
        )

        # Only keep neurons whose mean activation is above zero.
        top_values = top_k.values[:k_actual]
        top_indices = top_k.indices[:k_actual]

        # Map the global neuron indices back to their respective
        # layers and local neuron indices.
        top_k_per_layer = {
            layer_name: {}
            for layer_name in act_max_cache.layer_names
        }

        start_idx = 0

        for layer_name in act_max_cache.layer_names:
            end_idx = (
                    start_idx
                    + layers_lens[layer_name]
            )

            _find_neurons_of_layer(
                layer_name,
                start_idx,
                end_idx,
            )

            start_idx = end_idx

        top_k_per_layer_filtered = {}

        # Preserve the original filtering behavior: layers with
        # only one selected neuron are excluded.
        for layer, content in top_k_per_layer.items():
            if len(content) > 1:
                top_k_per_layer_filtered[layer] = (
                    content
                )

        print(
            "Top-k neurons per layer:",
            top_k_per_layer_filtered,
        )

        top_k_max_samples = {
            layer_name: {}
            for layer_name
            in top_k_per_layer_filtered
        }

        for (
                layer_name,
                neurons,
        ) in top_k_per_layer_filtered.items():
            for neuron_id in neurons:
                top_k_max_samples[layer_name][
                    neuron_id
                ] = self.get_max_reference(
                    layer_name
                )[neuron_id]

        torch.save(
            top_k_per_layer_filtered,
            topk_n_path,
        )
        torch.save(
            top_k_max_samples,
            top_samples_path,
        )

        self.top_k_per_layer = (
            top_k_per_layer_filtered
        )
        self.top_k_per_layer_filtered = (
            top_k_per_layer_filtered
        )
        self.top_k_max_samples = (
            top_k_max_samples
        )

        return (
            top_k_per_layer_filtered,
            top_k_max_samples,
        )


    def get_mask_and_crop(self, sample, layer, neuron_id, grid_h=None, grid_w=None, img_h= None, img_w = None, text_tokens_n=None):
        """
        Returns information necessary to create crops: original img path and bounding box.
        :param sample:
        :param layer:
        :param neuron_id:
        :param grid_h:
        :param grid_w:
        :param img_h:
        :param img_w:
        :param text_tokens_n:
        :return:
        """

        # find the position of the sample's id in the cache.sample_ids (shape=[n_neurons, n_samples])
        img_pos = torch.where(self.actmax_cache.cache[layer].sample_ids[neuron_id] == sample) # in all dims

        img_pos = int(img_pos[0]) # only in 0-th dim

        # Load raw image
        image_path = self.dataset.dataset.samples[sample][0]  # image file path
        try:
                        orig_img = cv2.imread(image_path)
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                        h_orig, w_orig, _ = orig_img.shape
        except Exception:
                        # Fallback to dummy image if file doesn't exist
                        print(f"Could not load {image_path}. Generating dummy image for testing.")
                        orig_img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
                        h_orig, w_orig, _ = orig_img.shape

        cached_activation = (
            self.actmax_cache.cache[layer]
            .activations_not_aggr[
                neuron_id,
                img_pos,
            ]
        )

        heatmap_2d = self.adapter.create_heatmap(
            cached_activation
        )

        if heatmap_2d.ndim != 2:
            raise ValueError(
                "Adapter must return a two-dimensional heatmap, "
                f"got shape {heatmap_2d.shape}"
            )

        grid_h, grid_w = heatmap_2d.shape
        if grid_h <= 0 or grid_w <= 0:
            raise ValueError(
                f"Invalid spatial grid {grid_h}x{grid_w} "
                f"for layer={layer}, neuron={neuron_id}, "
                f"sample={sample}"
            )

        # use magnitude (keeps negative activations)
        heatmap_2d = np.maximum(heatmap_2d, 0)

        # normalize
        max_val = heatmap_2d.max()
        if max_val > 0:
            heatmap_2d = heatmap_2d / max_val
        # -------------------------------
        # Threshold on patch grid
        # -------------------------------

        threshold = THRESHOLD_ACT
        #threshold = np.percentile(heatmap_2d, 50)

        binary_patch_mask = (heatmap_2d >= threshold).astype(np.uint8)

        # Safety: make sure at least one patch exists
        if binary_patch_mask.sum() == 0:
                            top_k = 5

                            flat_idx = np.argsort(heatmap_2d.flatten())[-top_k:]

                            binary_patch_mask = np.zeros_like(heatmap_2d, dtype=np.uint8)

                            for idx in flat_idx:
                                y, x = np.unravel_index(idx, heatmap_2d.shape)
                                binary_patch_mask[y, x] = 1

        # -------------------------------
        # Connected components on patches
        # -------------------------------

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                            binary_patch_mask,
                            connectivity=8
                    )

        if num_labels > 1:

                            # largest non-background component
                            largest_label = (
                                    1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                            )

                            component_mask = (
                                    labels == largest_label
                            ).astype(np.uint8)

        else:

                            component_mask = binary_patch_mask

        # -------------------------------
        # Patch coordinates -> image coordinates
        # -------------------------------

        ys, xs = np.where(component_mask)

        # safety check
        if len(xs) == 0:

                            # fallback to full image
                            xmin, ymin = 0, 0
                            xmax, ymax = w_orig, h_orig

        else:
                            xmin_patch = xs.min()
                            xmax_patch = xs.max()

                            ymin_patch = ys.min()
                            ymax_patch = ys.max()

                            patch_w = w_orig / grid_w
                            patch_h = h_orig / grid_h

                            xmin = int(xmin_patch * patch_w)
                            xmax = int((xmax_patch + 1) * patch_w)

                            ymin = int(ymin_patch * patch_h)
                            ymax = int((ymax_patch + 1) * patch_h)


        # -------------------------------
        # Full-resolution heatmap for display
        # -------------------------------

        heatmap_resized = cv2.resize(
            heatmap_2d,
            (w_orig, h_orig),
            interpolation=cv2.INTER_LINEAR
        )

        heatmaps_dir = f"{OUTPUTS}/act_heatmaps/heatmaps_{self.model_name}"
        os.makedirs(heatmaps_dir, exist_ok=True)
        heatmap_uint8 = (np.clip(heatmap_resized, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(heatmap_uint8).convert("RGB").save(f"{heatmaps_dir}/{layer}_n{neuron_id}_s{sample}.jpeg")

        return image_path, (ymin, ymax, xmin, xmax)

    # TODO: only for my application. Helper / in embedding file
    def create_crop_db(self) -> list:
        '''
        Collect unique crops into one df for embedding
        :param top_k_refs: dict(layer_name:str, dict(neuron_id:int, refs_ids:tensor))
        :return:  [{ image_id: tensor,
                     image_path: str,
                     bbox: tuple,
                     img_array: np.ndarray
                   },
                   ...]
        '''
        crop_db_path = f"{OUTPUTS}/embeddings/semanticlens/{self.actmax_cache.n_collect}_refs_per_neuron/crop_db/crop_db_{self.model_name}.pt"
        if self.top_k_max_samples is None:
            self.find_top_act_neurons(K_ACT_NEURONS)

        if os.path.exists(crop_db_path):
            crop_db = torch.load(crop_db_path)

            self.crop_db = crop_db

            print(f"Crop_db was loaded from an existing file. It contains {len(crop_db)} crops.")

            self.crop_info = {
                layer: {}
                for layer in self.layer_names
            }

            for layer, info in self.top_k_max_samples.items():
                if layer not in self.crop_info:
                    continue

                print(f"creating crop_db for layer {layer}")

                for channel_id, ref_samples in info.items():
                    for sample in ref_samples:
                        if sample == -1:
                            break

                        # get crop coordinates and original image path/ id
                        img_path, bbox = self.get_mask_and_crop(
                            sample,
                            layer,
                            channel_id
                        )

                        # add crop to crop_info
                        if channel_id not in self.crop_info[layer]:
                            self.crop_info[layer][channel_id] = [(img_path, bbox)]
                        else:
                            self.crop_info[layer][channel_id].append((img_path, bbox))

            return crop_db

        print("Creating crop db...")

        crop_db = []
        self.crop_info = {
            layer: {}
            for layer in self.top_k_max_samples.keys()
        }
        # add image info and bbox into crop_db
        seen = set()

        for layer, info in self.top_k_max_samples.items():
            for channel_id, ref_samples in info.items():
                for sample in ref_samples:
                    if sample == -1:
                        break

                    # get crop coordinates and original image path/ id
                    img_path, bbox = self.get_mask_and_crop(
                        sample,
                        layer,
                        channel_id
                    )

                    key = (sample, img_path, bbox)

                    if key not in seen:
                        seen.add(key)

                        crop_db.append({
                            "image_id": sample,
                            "image_path": img_path,
                            "bbox": bbox
                        })

                    # add crop to crop_info
                    if channel_id not in self.crop_info[layer]:
                        self.crop_info[layer][channel_id] = [(img_path, bbox)]
                    else:
                        self.crop_info[layer][channel_id].append((img_path, bbox))

        # add crop_id key to elements of crop_db
        for i, el in enumerate(crop_db):
            el["crop_id"] = i

        crop_db_dir = f"{OUTPUTS}/embeddings/semanticlens/{self.actmax_cache.n_collect}_refs_per_neuron/crop_db"
        os.makedirs(crop_db_dir, exist_ok=True)
        torch.save(crop_db, crop_db_path)

        self.crop_db = crop_db
        print(f"Crop_db was successfully created. It contains {len(crop_db)} crops.")

        return crop_db


    def create_neuron_crops_mapping(self):
        if self.crop_db is None:
            self.crop_db = self.create_crop_db()

        crop_lookup = {
            (item['image_path'], tuple(item['bbox'])): item['crop_id']
            for item in self.crop_db
        }

        self.neurons_crops_mapping = {layer: {} for layer in self.top_k_max_samples}
        for layer, info in self.crop_info.items():  # crop_info: neuron_id: [(img_path, bbox), ...]
            for neuron_id, crops_info in info.items():


                for crop_info in crops_info:
                    if neuron_id not in self.neurons_crops_mapping[layer]:
                        try:
                            self.neurons_crops_mapping[layer][neuron_id] = [crop_lookup[crop_info]]
                        except KeyError: # means that cache contains either different images path or crops
                            print("Recomputing crop_db, since it contains incompatible information.")
                            crop_db_path = f"{OUTPUTS}/embeddings/semanticlens/{self.actmax_cache.n_collect}_refs_per_neuron/crop_db/crop_db_{self.model_name}.pt"
                            os.remove(crop_db_path)
                            self.crop_db = self.create_crop_db()
                            crop_lookup = {
                                (item['image_path'], tuple(item['bbox'])): item['crop_id']
                                for item in self.crop_db
                            }
                            self.neurons_crops_mapping[layer][neuron_id] = [crop_lookup[crop_info]]

                    else:
                        self.neurons_crops_mapping[layer][neuron_id].append(crop_lookup[crop_info])

        return self.neurons_crops_mapping


    def _create_crops_list(self):
        if self.crop_db is None:
            self.crop_db = self.create_crop_db()
        return [restore_crop(el["image_path"], el["bbox"]) for el in self.crop_db]



    def _embed_vision_dataset(self, fm, batch_size, top_k_refs, **kwargs):
        def pil_list_collate(batch):
            return list(batch)

        class CropDataset(torch.utils.data.Dataset):
            def __init__(self, crops):
                self.crops = crops

            def __len__(self):
                return len(self.crops)

            def __getitem__(self, idx):
                return Image.fromarray(self.crops[idx])

        crops = self._create_crops_list()
        loader = DataLoader(
            CropDataset(crops),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=pil_list_collate
        )

        embeds = []

        for pil_list in loader:
            inputs = fm.preprocess(pil_list)
            fm_out = fm.encode_image(inputs)

            embeds.append(fm_out)

        embeds = torch.cat(embeds)

        assert len(embeds) == len(crops)

        return embeds


    # override it to save embeddings when computed and load them from a file, if it already exists
    def _compute_concept_db(self, fm, top_k_refs, batch_size=32, **kwargs):
        """
        Compute or read from a file the concept database for the given fm (foundation model).
        Returns
        -------
        dict
            Dictionary mapping layer names to embedded concept representations.
        """
        concept_db_path = f"{OUTPUTS}/embeddings/semanticlens/{N_SAMPLES}_refs_per_neuron/{EMBEDDING_MODEL}/concept_db/concept_db_{self.model_name}.pt"
        if os.path.exists(concept_db_path):
            concept_db = torch.load(concept_db_path, map_location=torch.device(self.device))

            cached_layers = (
                set(concept_db.keys())
            )
            current_layers = set(self.layer_names)

            unknown_cached_layers = (
                    cached_layers - current_layers
            )

            if not unknown_cached_layers:
                self.concept_db = concept_db

                return concept_db

            print(
                "The existing concept_db cache is incompatible "
                "with the current layer names."
            )
            print(
                "Unknown cached layers:",
                unknown_cached_layers,
            )
            print("Recomputing concept_db...")

            self.concept_db = concept_db
            print("Loaded concept_db from an existing file")

            self.crop_info = {
                layer: {}
                for layer in self.top_k_max_samples.keys()
            }

            for layer, info in self.top_k_max_samples.items():
                for channel_id, ref_samples in info.items():
                    for sample in ref_samples:
                        if sample == -1:
                            break

                        # get crop coordinates and original image path/ id
                        img_path, bbox = self.get_mask_and_crop(
                            sample,
                            layer,
                            channel_id
                        )

                        # add crop to crop_info
                        if channel_id not in self.crop_info[layer]:
                            self.crop_info[layer][channel_id] = [(img_path, bbox)]
                        else:
                            self.crop_info[layer][channel_id].append((img_path, bbox))

            return concept_db

        if self.neurons_crops_mapping is None:
            self.neurons_crops_mapping = self.create_neuron_crops_mapping()

        '''
        embeds file contains a dict:
            key(s): "embeds_*model name*"
            values: embeddings of the images in the reference dataset

            shape: dataset size x latent embeddings
        '''
        embeds_path = f"{OUTPUTS}/embeddings/semanticlens/{N_SAMPLES}_refs_per_neuron/{EMBEDDING_MODEL}/embeds/embeds_{self.model_name}.pt"
        if os.path.exists(embeds_path):
            print(f"Load embeddings from existing file")
            embeds = torch.load(embeds_path, map_location="cpu")
            embeds = embeds[f"embeds_{self.model_name}"]
        else:
            print(f"Embedding the dataset")
            embeds = self._embed_vision_dataset(fm, batch_size, top_k_refs, **kwargs)
            embeds_dir = f"{OUTPUTS}/embeddings/semanticlens/{N_SAMPLES}_refs_per_neuron/{EMBEDDING_MODEL}/embeds"
            os.makedirs(embeds_dir, exist_ok=True)
            torch.save({f"embeds_{self.model_name}": embeds}, embeds_path)

        self.embeds = embeds

        concept_db = {layer: {} for layer in self.top_k_max_samples.keys()}
        print("Building concept db...")

        for layer_name, info in self.top_k_max_samples.items():
            for neuron_id, _ in info.items():
                concept_db[layer_name][neuron_id] = embeds[[self.neurons_crops_mapping[layer_name][neuron_id]]]

        self.concept_db = concept_db
        concept_db_dir = f"{OUTPUTS}/embeddings/semanticlens/{N_SAMPLES}_refs_per_neuron/{EMBEDDING_MODEL}/concept_db"
        os.makedirs(concept_db_dir, exist_ok=True)
        torch.save(concept_db, concept_db_path)

        return concept_db
