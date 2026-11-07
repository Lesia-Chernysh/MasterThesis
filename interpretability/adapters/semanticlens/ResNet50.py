import numpy as np
import torch

from base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo,
)


class ResNetAdapter(SemanticLensAdapter):
    def prepare_batch(
        self,
        images,
        device,
    ) -> PreparedBatch:
        images = images.to(device)

        return PreparedBatch(
            args=(images,),
        )

    def canonicalize_activations(
        self,
        activations: torch.Tensor,
        prepared_batch: PreparedBatch,
    ) -> torch.Tensor:
        """
        ResNet hook output already has the canonical shape:

            [batch, channels, height, width]
        """

        if activations.ndim != 4:
            raise ValueError(
                "Expected ResNet activations with shape "
                f"[B, C, H, W], got {tuple(activations.shape)}"
            )

        return activations

    def create_heatmap(
            self,
            neuron_activation,
    ):
        if neuron_activation.ndim != 2:
            raise ValueError(
                "Expected ResNet activation [H, W], "
                f"got {tuple(neuron_activation.shape)}"
            )

        return (
            neuron_activation
            .detach()
            .cpu()
            .float()
            .numpy()
        )