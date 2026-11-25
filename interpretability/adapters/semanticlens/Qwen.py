import copy

import numpy as np
import torch
from transformers import AutoProcessor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from interpretability.adapters.semanticlens.base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo
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
            questions=None,
    ) -> PreparedBatch:
        images = list(images)

        # Resize the actual objects that will be put into the
        # conversations.
        prepared_images = [
            resize(
                image,
                size=[224, 224],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            for image in images
        ]

        if questions is None:
            questions = [
                self.prompt
                for _ in prepared_images
            ]
        else:
            questions = list(questions)

        if len(questions) != len(prepared_images):
            raise ValueError(
                f"Received {len(prepared_images)} images but "
                f"{len(questions)} questions"
            )

        conversations = []

        for image, question in zip(
                prepared_images,
                questions,
        ):
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",

                                # Important: use the resized image.
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ]
            )

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = inputs.to(device)

        grid_thw = inputs["image_grid_thw"]

        expected = torch.tensor(
            [1, 16, 16],
            dtype=grid_thw.dtype,
            device=grid_thw.device,
        )

        valid_grids = torch.all(
            grid_thw == expected.unsqueeze(0),
            dim=1,
        )

        if not torch.all(valid_grids):
            raise ValueError(
                "Unexpected Qwen visual grids: "
                f"{grid_thw.detach().cpu().tolist()}; "
                "expected [1, 16, 16] for every image"
            )

        return PreparedBatch(
            kwargs=dict(inputs),
            spatial_info=[
                SpatialInfo(
                    grid_h=16,
                    grid_w=16,
                )
                for _ in prepared_images
            ],
        )

    def prepare_loader_batch(
            self,
            batch,
            device,
    ):
        questions = None

        if isinstance(batch, dict):
            images = batch["image"]
            questions = batch.get("question")

        elif isinstance(batch, (tuple, list)):
            images = batch[0]

            if len(batch) >= 2:
                possible_questions = batch[1]

                if (
                        isinstance(possible_questions, (tuple, list))
                        and possible_questions
                        and all(
                    isinstance(question, str)
                    for question in possible_questions
                )
                ):
                    questions = list(possible_questions)
        else:
            images = batch

        return self.prepare_batch(
            images=images,
            device=device,
            questions=questions,
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


    # Don't include the question, since visual encoder encodes the image before it sees the qustion
    def forward(
            self,
            model,
            prepared_batch,
    ):
        pixel_values = prepared_batch.kwargs[
            "pixel_values"
        ]

        image_grid_thw = prepared_batch.kwargs[
            "image_grid_thw"
        ]

        # Transformers versions expose the vision encoder in one
        # of these two locations.
        vision_model = getattr(
            model,
            "visual",
            None,
        )

        if vision_model is None:
            vision_model = model.model.visual

        return vision_model(
            pixel_values,
            grid_thw=image_grid_thw,
        )