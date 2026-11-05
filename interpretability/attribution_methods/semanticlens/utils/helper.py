"""Helper functions for SemanticLens."""

from __future__ import annotations

import cv2

import hashlib

import torch
from torchvision import transforms

from semanticlens.scores import polysemanticity_score

def restore_crop(img_path, bbox):
    """
    Creates an array of a crop based on the original image and bounding box.
    :param img_path:
    :param bbox:
    :return:
    """
    ymin, ymax, xmin, xmax = bbox
    orig_img = cv2.imread(img_path)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    return orig_img[ymin:ymax, xmin:xmax]

def get_polysem_score(concept_db: dict) -> dict:
    print("Computing polysemanticity scores per neuron in each layer...")
    polysem_scores = {layer: {} for layer in concept_db}


    for layer, neuron_info in concept_db.items():
        for neuron_id, embeddings in neuron_info.items():
            current_shape = concept_db[layer][neuron_id].shape # --> 10, 512
            polysem_scores[layer][neuron_id] = polysemanticity_score(concept_db[layer][neuron_id].view(-1, current_shape[0], current_shape[1]))

    return polysem_scores

def _string_hash(s: str) -> int:
    """Generate a stable hash for a given string."""
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def get_fallback_name(obj):
    """Function to get a fallback name for an object."""
    return obj.__class__.__name__ + "-" + str(_string_hash(str(obj)))


def to_transforms_compose(instance: transforms._presets.ImageClassification) -> transforms.Compose:
    """Helper function to convert an ImageClassification instance to a Compose transform."""
    return transforms.Compose(
        [
            transforms.Resize(
                instance.resize_size,
                interpolation=instance.interpolation,
                antialias=instance.antialias,
            ),
            transforms.CenterCrop(instance.crop_size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=instance.mean, std=instance.std),
        ]
    )


def get_denormalization_transform(
    mean: torch.Tensor | list[float] = torch.tensor([0.485, 0.456, 0.406]),
    std: torch.Tensor | list[float] = torch.tensor([0.229, 0.224, 0.225]),
) -> torch.nn.Module:
    """Return a transform to undo the normalization of an image tensor.

    This function creates a composition of transforms that reverses standard
    ImageNet normalization, which is useful for visualization purposes.

    Parameters
    ----------
    mean : torch.Tensor, optional
        Mean values used in the original normalization. Default is ImageNet means
        [0.485, 0.456, 0.406] for RGB channels.
    std : torch.Tensor, optional
        Standard deviation values used in the original normalization. Default is
        ImageNet stds [0.229, 0.224, 0.225] for RGB channels.

    Returns
    -------
    torch.nn.Module
        A composed transform that reverses the normalization when applied to a tensor.
    """
    from torchvision import transforms

    if isinstance(mean, list):
        mean = torch.tensor(mean, dtype=torch.float32)

    if isinstance(std, list):
        std = torch.tensor(std, dtype=torch.float32)

    return transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )
