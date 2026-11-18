from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import torch
from transformers import ViltProcessor
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

from interpretability.adapters.semanticlens.base import (
    PreparedBatch,
    SemanticLensAdapter,
    SpatialInfo
)


class ViLTAdapter(SemanticLensAdapter):
    def __init__(
        self,
        processor: ViltProcessor | None = None,
        model_name: str = (
            "dandelin/vilt-b32-finetuned-vqa"
        ),
        default_question: str = (
            "what is on the image?"
        ),
    ):
        self.processor = (
            processor
            if processor is not None
            else ViltProcessor.from_pretrained(
                model_name
            )
        )

        self.default_question = default_question

    def collate_fn(self, batch):
        """
        Preserve images as a list.

        This allows ViltProcessor to receive PIL images of different
        dimensions instead of asking PyTorch to stack them.
        """

        if not batch:
            raise ValueError("Cannot collate an empty batch")

        first_sample = batch[0]

        if isinstance(first_sample, dict):
            images = [
                sample["image"]
                for sample in batch
            ]

            questions = [
                sample.get("question")
                for sample in batch
            ]

            return images, questions

        if isinstance(first_sample, (tuple, list)):
            images = [
                sample[0]
                for sample in batch
            ]

            additional_values = [
                sample[1]
                for sample in batch
            ]

            return images, additional_values

        # Dataset returns images without labels/questions.
        return list(batch)

    def prepare_loader_batch(
        self,
        batch,
        device: torch.device | str,
    ) -> PreparedBatch:
        """
        Handle either:

            (images, VQA questions)
            (images, numeric labels)
            images only

        A second batch element is interpreted as questions only if it
        contains strings. Numeric class labels are ignored.
        """

        questions = None

        if isinstance(batch, dict):
            images = batch["image"]

            questions = batch.get(
                "question",
                batch.get("questions"),
            )

        elif isinstance(batch, (tuple, list)):
            if len(batch) == 0:
                raise ValueError(
                    "Received an empty dataloader batch"
                )

            images = batch[0]

            if len(batch) >= 2:
                candidate_questions = batch[1]

                if self._contains_questions(
                    candidate_questions
                ):
                    questions = candidate_questions
        else:
            images = batch

        return self.prepare_batch(
            images=images,
            device=device,
            questions=questions,
        )

    @staticmethod
    def _contains_questions(value) -> bool:
        """
        Return True for a string or a sequence of strings.

        Consequently, tensors containing ImageNet labels are not
        interpreted as questions.
        """

        if isinstance(value, str):
            return True

        if isinstance(value, Sequence):
            return (
                len(value) > 0
                and all(
                    isinstance(element, str)
                    for element in value
                )
            )

        return False

    def prepare_batch(
        self,
        images,
        device: torch.device | str,
        questions=None,
    ) -> PreparedBatch:
        processor_images = self._prepare_images(images)

        batch_size = len(processor_images)

        if batch_size == 0:
            raise ValueError(
                "Cannot prepare an empty ViLT batch"
            )

        questions = self._prepare_questions(
            questions=questions,
            batch_size=batch_size,
        )

        encoding = self.processor(
            images=processor_images,
            text=questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        encoding = encoding.to(device)

        return PreparedBatch(
            kwargs=dict(encoding),
        )

    @staticmethod
    def _prepare_images(self, images) -> list:
        """
        Resize every image to a fixed square.

        This guarantees a constant ViLT patch grid:
            384 / 32 = 12
            -> 12 x 12 patches
        """

        if isinstance(images, torch.Tensor):
            if images.ndim == 3:
                images = images.unsqueeze(0)

            if images.ndim != 4:
                raise ValueError(
                    "Expected [B, C, H, W], got "
                    f"{tuple(images.shape)}"
                )

            images = [
                image.detach().cpu()
                for image in images
            ]

        elif isinstance(images, Sequence):
            images = list(images)

        else:
            raise TypeError(
                "Expected a batched tensor or a sequence "
                f"of images, got {type(images)!r}"
            )

        prepared_images = []

        for image in images:
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu()

                # ViltProcessor expects float tensors in [0, 1].
                if image.is_floating_point():
                    min_value = float(image.min())
                    max_value = float(image.max())

                    if min_value < 0 or max_value > 1:
                        raise ValueError(
                            "ViLT received an already-normalized "
                            "image tensor with range "
                            f"[{min_value}, {max_value}]. "
                            "Remove Normalize from the dataset "
                            "transform."
                        )

            image = resize(
                image,
                size=[
                    self.image_size,
                    self.image_size,
                ],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )

            prepared_images.append(image)

        return prepared_images

    def _prepare_questions(
        self,
        questions,
        batch_size: int,
    ) -> list[str]:
        if questions is None:
            return [
                self.default_question
                for _ in range(batch_size)
            ]

        if isinstance(questions, str):
            questions = [questions]

        questions = list(questions)

        if len(questions) != batch_size:
            raise ValueError(
                "The number of ViLT questions must equal the "
                f"number of images: {len(questions)} questions "
                f"for {batch_size} images"
            )

        if not all(
            isinstance(question, str)
            for question in questions
        ):
            raise TypeError(
                "Every ViLT question must be a string"
            )

        return questions

    # Keep the following methods from the previous ViLT adapter:
    #
    # def forward(...)
    # def canonicalize_activations(...)
    # def create_heatmap(...)

    def forward(
        self,
        model,
        prepared_batch: PreparedBatch,
    ):
        """
        Execute one ViLT forward pass and capture the mapping between visual
        tokens and their original patch-grid positions.

        ViLT's visual embedding implementation may shuffle or subsample image
        patches. Consequently, taking the last N activations and reshaping
        them directly is not reliable.
        """

        spatial_metadata_captured = False

        def capture_spatial_information(
            module,
            inputs,
            output,
        ):
            nonlocal spatial_metadata_captured

            if not isinstance(output, (tuple, list)):
                raise TypeError(
                    "Expected ViLT embeddings to return a tuple, "
                    f"got {type(output)!r}"
                )

            if len(output) < 3:
                raise ValueError(
                    "Expected ViLT embeddings to return "
                    "(embeddings, masks, patch_information)"
                )

            _, combined_mask, patch_information = output

            if (
                not isinstance(patch_information, (tuple, list))
                or len(patch_information) != 2
            ):
                raise ValueError(
                    "Unexpected ViLT patch-information format"
                )

            patch_indices, grid_shape = patch_information

            if (
                not isinstance(grid_shape, (tuple, list))
                or len(grid_shape) != 2
            ):
                raise ValueError(
                    "Unexpected ViLT grid-shape format"
                )

            grid_h = int(grid_shape[0])
            grid_w = int(grid_shape[1])

            if patch_indices.ndim != 3:
                raise ValueError(
                    "Expected ViLT patch indices with shape "
                    "[B, N, 2], got "
                    f"{tuple(patch_indices.shape)}"
                )

            if patch_indices.shape[-1] != 2:
                raise ValueError(
                    "The last patch-index dimension must contain "
                    "(row, column)"
                )

            input_ids = prepared_batch.kwargs.get("input_ids")

            if input_ids is None:
                raise ValueError(
                    "ViLT prepared batch does not contain input_ids"
                )

            text_token_count = input_ids.shape[1]

            # combined_mask consists of:
            #
            #   [text tokens, image CLS token, image patch tokens]
            image_mask = combined_mask[:, text_token_count:]

            expected_image_tokens = patch_indices.shape[1] + 1

            if image_mask.shape[1] != expected_image_tokens:
                raise ValueError(
                    "ViLT image-mask and patch-index sizes are "
                    "incompatible: "
                    f"{image_mask.shape[1]} image tokens versus "
                    f"{patch_indices.shape[1]} patches plus CLS"
                )

            prepared_batch.spatial_info = [
                SpatialInfo(
                    grid_h=grid_h,
                    grid_w=grid_w,
                    metadata={
                        "patch_indices": patch_indices[
                            batch_index
                        ],
                        # Remove the image CLS entry from the mask.
                        "patch_mask": image_mask[
                            batch_index,
                            1:,
                        ].bool(),
                        "text_token_count": int(
                            text_token_count
                        ),
                    },
                )
                for batch_index in range(
                    patch_indices.shape[0]
                )
            ]

            spatial_metadata_captured = True

        try:
            embeddings_module = model.vilt.embeddings
        except AttributeError as error:
            raise AttributeError(
                "The supplied model does not expose "
                "model.vilt.embeddings and therefore does not "
                "look like ViltForQuestionAnswering"
            ) from error

        handle = embeddings_module.register_forward_hook(
            capture_spatial_information
        )

        try:
            result = model(
                *prepared_batch.args,
                **prepared_batch.kwargs,
            )
        finally:
            handle.remove()

        if not spatial_metadata_captured:
            raise RuntimeError(
                "The ViLT embeddings hook did not capture spatial "
                "patch information"
            )

        return result

    def canonicalize_activations(
        self,
        activations: torch.Tensor,
        prepared_batch: PreparedBatch,
    ) -> torch.Tensor:
        """
        Convert:

            [B, text + image_cls + image_patches, neurons]

        into:

            [B, neurons, grid_h, grid_w]

        Patch indices captured during the forward pass are used to undo
        ViLT's patch shuffling.
        """

        if isinstance(activations, (tuple, list)):
            if len(activations) == 0:
                raise ValueError(
                    "The hooked ViLT layer returned an empty tuple"
                )

            activations = activations[0]

        if not isinstance(activations, torch.Tensor):
            raise TypeError(
                "Expected hooked ViLT activations to be a tensor, "
                f"got {type(activations)!r}"
            )

        if activations.ndim != 3:
            raise ValueError(
                "Expected ViLT activations with shape "
                "[B, tokens, neurons], got "
                f"{tuple(activations.shape)}"
            )

        batch_size, sequence_length, n_neurons = (
            activations.shape
        )

        if len(prepared_batch.spatial_info) != batch_size:
            raise ValueError(
                "Missing ViLT spatial information: expected "
                f"{batch_size} entries, got "
                f"{len(prepared_batch.spatial_info)}"
            )

        grid_shapes = {
            (
                info.grid_h,
                info.grid_w,
            )
            for info in prepared_batch.spatial_info
        }

        if len(grid_shapes) != 1:
            raise ValueError(
                "All samples in one activation-cache batch must "
                "have the same padded visual grid, got "
                f"{sorted(grid_shapes)}"
            )

        grid_h, grid_w = next(iter(grid_shapes))

        canonical = activations.new_zeros(
            (
                batch_size,
                n_neurons,
                grid_h,
                grid_w,
            )
        )

        for batch_index, info in enumerate(
            prepared_batch.spatial_info
        ):
            text_token_count = int(
                info.metadata["text_token_count"]
            )

            # Sequence layout:
            #
            #   text tokens
            #   image CLS token
            #   selected image patch tokens
            patch_activations = activations[
                batch_index,
                text_token_count + 1:,
                :,
            ]

            patch_indices = info.metadata[
                "patch_indices"
            ].to(activations.device)

            patch_mask = info.metadata[
                "patch_mask"
            ].to(activations.device)

            if patch_activations.shape[0] != (
                patch_indices.shape[0]
            ):
                raise ValueError(
                    "The number of ViLT patch activations does "
                    "not match the number of patch indices: "
                    f"{patch_activations.shape[0]} versus "
                    f"{patch_indices.shape[0]}"
                )

            if patch_mask.shape[0] != (
                patch_activations.shape[0]
            ):
                raise ValueError(
                    "The ViLT patch mask has an incompatible "
                    "length"
                )

            valid_indices = patch_indices[patch_mask]
            valid_activations = patch_activations[patch_mask]

            if valid_indices.numel() == 0:
                continue

            rows = valid_indices[:, 0].long()
            columns = valid_indices[:, 1].long()

            if (
                torch.any(rows < 0)
                or torch.any(rows >= grid_h)
                or torch.any(columns < 0)
                or torch.any(columns >= grid_w)
            ):
                raise ValueError(
                    "ViLT returned patch coordinates outside "
                    f"the {grid_h}x{grid_w} grid"
                )

            flat_positions = rows * grid_w + columns

            # [neurons, H * W]
            flat_grid = canonical[
                batch_index
            ].reshape(n_neurons, -1)

            # valid_activations:
            #     [valid_patches, neurons]
            flat_grid[:, flat_positions] = (
                valid_activations.transpose(0, 1)
            )

        return canonical

    def create_heatmap(
        self,
        neuron_activation: torch.Tensor,
    ) -> np.ndarray:
        """
        Convert one cached ViLT neuron activation into a heatmap.

        canonicalize_activations() has already restored the patch layout,
        so no token slicing or reshaping is needed here.
        """

        if neuron_activation.ndim != 2:
            raise ValueError(
                "Expected a canonical ViLT activation with shape "
                "[grid_h, grid_w], got "
                f"{tuple(neuron_activation.shape)}"
            )

        return (
            neuron_activation
            .detach()
            .cpu()
            .float()
            .numpy()
        )