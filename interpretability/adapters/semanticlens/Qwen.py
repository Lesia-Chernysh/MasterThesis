import torch
from transformers import AutoProcessor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from interpretability.adapters.semanticlens.base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo,
)


class QwenAdapter(SemanticLensAdapter):
    """SemanticLens adapter for Qwen2-VL's vision encoder."""

    def __init__(
        self,
        processor=None,
        prompt: str = (
            "Answer in ONE word. "
            "What object is depicted in the image?"
        ),
        input_size: int = 224,
        patch_size: int = 14,
        vision_only: bool = True,
    ):
        if input_size % (2 * patch_size) != 0:
            raise ValueError(
                "Qwen input_size must be divisible by the 28-pixel "
                f"spatial merge factor, got {input_size}."
            )

        self.processor = (
            processor
            if processor is not None
            else AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-2B-Instruct"
            )
        )
        self.prompt = prompt
        self.input_size = input_size
        self.patch_size = patch_size
        self.grid_h = input_size // patch_size
        self.grid_w = input_size // patch_size
        self.vision_only = vision_only

    def _prepare_images(self, images):
        return [
            resize(
                image,
                size=[self.input_size, self.input_size],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
            for image in list(images)
        ]

    def _validate_grid(self, grid_thw):
        expected = torch.tensor(
            [1, self.grid_h, self.grid_w],
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
                f"{grid_thw.detach().cpu().tolist()}; expected "
                f"[1, {self.grid_h}, {self.grid_w}] for every image."
            )

    def _spatial_info(self, batch_size):
        return [
            SpatialInfo(
                grid_h=self.grid_h,
                grid_w=self.grid_w,
                metadata={"temporal_grid": 1},
            )
            for _ in range(batch_size)
        ]

    def prepare_batch(
        self,
        images,
        device,
        questions=None,
    ) -> PreparedBatch:
        prepared_images = self._prepare_images(images)

        if self.vision_only:
            # Questions cannot affect visual.blocks.*, so do not spend time
            # constructing and tokenizing conversations during this stage.
            inputs = self.processor.image_processor(
                images=prepared_images,
                return_tensors="pt",
            ).to(device)
        else:
            if questions is None:
                questions = [self.prompt] * len(prepared_images)
            else:
                questions = list(questions)

            if len(questions) != len(prepared_images):
                raise ValueError(
                    f"Received {len(prepared_images)} images but "
                    f"{len(questions)} questions."
                )

            conversations = [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": question},
                        ],
                    }
                ]
                for image, question in zip(prepared_images, questions)
            ]

            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

        self._validate_grid(inputs["image_grid_thw"])

        model_kwargs = dict(inputs)
        if not self.vision_only:
            model_kwargs["use_cache"] = False

        return PreparedBatch(
            kwargs=model_kwargs,
            spatial_info=self._spatial_info(len(prepared_images)),
        )

    def prepare_loader_batch(self, batch, device):
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
        if isinstance(activations, (tuple, list)):
            activations = activations[0]

        if activations.ndim != 2:
            raise ValueError(
                "Expected flattened Qwen visual activations with shape "
                f"[B*T, D], got {tuple(activations.shape)}."
            )

        token_counts = [
            info.grid_h * info.grid_w
            for info in prepared_batch.spatial_info
        ]
        if not token_counts:
            raise ValueError("Qwen spatial information is missing.")
        if len(set(token_counts)) != 1:
            raise ValueError(
                "Qwen samples have different visual-token counts: "
                f"{token_counts}."
            )

        batch_size = len(token_counts)
        tokens_per_image = token_counts[0]
        expected_rows = batch_size * tokens_per_image

        if activations.shape[0] != expected_rows:
            raise ValueError(
                "Qwen activation shape does not match its visual grid: "
                f"expected {expected_rows} rows, got "
                f"{activations.shape[0]}."
            )

        return activations.reshape(
            batch_size,
            tokens_per_image,
            activations.shape[-1],
        ).permute(0, 2, 1)

    def create_heatmap(self, neuron_activation):
        expected_tokens = self.grid_h * self.grid_w
        if neuron_activation.numel() != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} Qwen visual tokens, got "
                f"{neuron_activation.numel()}."
            )

        return (
            neuron_activation
            .reshape(self.grid_h, self.grid_w)
            .detach()
            .cpu()
            .float()
            .numpy()
        )

    def aggregate_for_cache(
        self,
        raw_activations,
        canonical_activations,
        aggregation_fn,
    ):
        # Qwen vision hooks flatten B and T into one dimension. The
        # canonical representation restores B, so aggregation remains valid
        # for any configured square input size instead of assuming T == 256.
        return canonical_activations.mean(dim=2)

    def forward(self, model, prepared_batch):
        if not self.vision_only:
            return super().forward(model, prepared_batch)

        vision_model = getattr(model, "visual", None)
        if vision_model is None:
            vision_model = model.model.visual

        return vision_model(
            prepared_batch.kwargs["pixel_values"],
            grid_thw=prepared_batch.kwargs["image_grid_thw"],
        )
