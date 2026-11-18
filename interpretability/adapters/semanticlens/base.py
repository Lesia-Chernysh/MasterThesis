from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class SpatialInfo:
    """
    Spatial metadata for one sample.

    grid_h/grid_w describe the visual-token grid.
    Additional information can be stored in metadata.
    """

    grid_h: int
    grid_w: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class PreparedBatch:
    """
    Model input and spatial information produced by an adapter.
    """

    args: tuple = ()
    kwargs: dict[str, Any] = field(
        default_factory=dict
    )
    spatial_info: list[SpatialInfo] = field(
        default_factory=list
    )


class SemanticLensAdapter(ABC):
    """
    Interface between SemanticLens and a particular model family.
    """

    def collate_fn(self, batch):
        """
        Default PyTorch collation for models whose datasets return
        tensors that can be stacked.
        """
        return default_collate(batch)

    @abstractmethod
    def prepare_batch(
        self,
        images,
        device: torch.device | str,
    ) -> PreparedBatch:
        """
        Construct the arguments required by model.forward().
        """

    def prepare_loader_batch(
            self,
            batch,
            device: torch.device | str,
    ) -> PreparedBatch:
        """
        Default handling for datasets returning:

            images
            (images, labels)
            (images, other_metadata)

        Model-specific adapters can override this method when they need
        additional dataset fields.
        """

        if isinstance(batch, dict):
            images = batch["image"]
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        return self.prepare_batch(
            images=images,
            device=device,
        )

    
    def forward(
        self,
        model,
        prepared_batch: PreparedBatch,
    ):
        """
        Execute exactly one model forward pass.
        """

        return model(
            *prepared_batch.args,
            **prepared_batch.kwargs,
        )

    @abstractmethod
    def canonicalize_activations(
        self,
        activations: torch.Tensor,
        prepared_batch: PreparedBatch,
    ) -> torch.Tensor:
        """
        Convert hook output to:

            [batch, neurons, *spatial_dimensions]
        """

    @abstractmethod
    def create_heatmap(
        self,
        neuron_activation: torch.Tensor,
    ) -> np.ndarray:
        """
        Convert one neuron's unaggregated activation into
        a two-dimensional visual heatmap.
        """

    def aggregate(
        self,
        canonical_activations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Default aggregation: mean across every spatial dimension.

        Input:
            [B, C, *spatial]

        Output:
            [B, C]
        """

        if canonical_activations.ndim < 3:
            raise ValueError(
                "Canonical activations must have at least "
                "three dimensions: [batch, neurons, spatial]"
            )

        return canonical_activations.flatten(
            start_dim=2
        ).mean(dim=2)