import copy

import numpy as np
import torch
from transformers import AutoProcessor

from base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo,
)

class QwenAdapter(SemanticLensAdapter):
    def __init__(
        self,
        processor=None,
        prompt: str = (
            "Answer in ONE word. "
            "What object is depicted in the image?"
        ),
        grid_h=16,
        grid_w=16,
    ):
        self.processor = (
            processor
            if processor is not None
            else AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct"
            )
        )
        self.prompt = prompt
        self.grid_h = grid_h
        self.grid_w = grid_w

    def prepare_batch(
        self,
        images,
        device,
    ) -> PreparedBatch:
        template = {
            "role": "user",
            "content": [
                { "type": "image", "image": None, },
                {"type": "text", "text": self.prompt, },
            ],
        }

        conversations = []

        for image in images:
            conversation = copy.deepcopy(template)
            conversation["content"][0]["image"] = image
            conversations.append([conversation])

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        # BatchEncoding supports .to(device).
        inputs = inputs.to(device)

        grid_thw = inputs[
            "image_grid_thw"
        ].detach().cpu()

        spatial_info = [
            SpatialInfo(
                grid_h=int(grid[1]),
                grid_w=int(grid[2]),
                metadata={
                    "temporal_grid": int(grid[0]),
                },
            )
            for grid in grid_thw
        ]

        return PreparedBatch(
            kwargs=dict(inputs),
            spatial_info=spatial_info,
        )

    def canonicalize_activations(
        self,
        activations: torch.Tensor,
        prepared_batch: PreparedBatch,
    ) -> torch.Tensor:
        """
        Convert flattened visual tokens:

            [batch * visual_tokens, neurons]

        to:

            [batch, neurons, visual_tokens]

        This implementation requires all samples in the batch
        to have the same visual-token grid.
        """

        token_counts = [
            info.grid_h * info.grid_w
            for info in prepared_batch.spatial_info
        ]

        if not token_counts:
            raise ValueError(
                "Qwen spatial information is missing"
            )

        if len(set(token_counts)) != 1:
            raise ValueError(
                "Qwen samples have different visual-token "
                f"counts: {token_counts}. A fixed-size cache "
                "cannot store this batch without padding or "
                "ragged tensors."
            )

        batch_size = len(token_counts)
        tokens_per_image = token_counts[0]

        expected_rows = (
            batch_size * tokens_per_image
        )

        if activations.shape[0] != expected_rows:
            raise ValueError(
                "Qwen activation shape does not match its "
                "visual grid: expected "
                f"{expected_rows} rows, got "
                f"{activations.shape[0]}"
            )

        activations = activations.reshape(
            batch_size,
            tokens_per_image,
            activations.shape[-1],
        )

        return activations.permute(0, 2, 1)

    def create_heatmap(
        self,
        neuron_activation,
    ):
        expected_tokens = (
            self.grid_h * self.grid_w
        )

        if neuron_activation.numel() != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} Qwen visual "
                f"tokens, got {neuron_activation.numel()}"
            )

        return (
            neuron_activation
            .reshape(
                self.grid_h,
                self.grid_w,
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )

        if neuron_activation.numel() != expected_tokens:
            raise ValueError(
                f"Cannot reshape {neuron_activation.numel()} "
                f"Qwen activations into "
                f"{spatial_info.grid_h}x"
                f"{spatial_info.grid_w}"
            )

        return (
            neuron_activation
            .reshape(
                spatial_info.grid_h,
                spatial_info.grid_w,
            )
            .detach()
            .cpu()
            .float()
            .numpy()
        )