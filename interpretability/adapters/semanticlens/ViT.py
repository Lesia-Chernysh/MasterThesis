import math

import numpy as np
import torch

from base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo,
)


class ViTAdapter(SemanticLensAdapter):
    def __init__(
        self,
        image_size: int = 224,
        has_cls_token: bool = True,
    ):
        self.image_size = image_size
        self.has_cls_token = has_cls_token

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
        Convert:

            [B, tokens, neurons]

        to:

            [B, neurons, tokens]
        """

        if activations.ndim != 3:
            raise ValueError(
                "Expected ViT activations with shape "
                f"[B, T, C], got {tuple(activations.shape)}"
            )

        return activations.transpose(1, 2)

    def create_heatmap(
            self,
            neuron_activation,
    ):
        # Remove the CLS token.
        patch_activations = neuron_activation[1:]

        n_patches = patch_activations.numel()
        grid_size = int(n_patches ** 0.5)

        if grid_size * grid_size != n_patches:
            raise ValueError(
                f"Cannot infer a square ViT grid from "
                f"{n_patches} patch activations"
            )

        return (
            patch_activations
            .reshape(grid_size, grid_size)
            .detach()
            .cpu()
            .float()
            .numpy()
        )