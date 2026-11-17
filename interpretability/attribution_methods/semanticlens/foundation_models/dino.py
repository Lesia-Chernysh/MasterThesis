import timm
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel



from abc import ABC, abstractmethod

import torch
from PIL import Image


class AbstractVisionModel(ABC):
    """
    Abstract base class for DINO-family vision models.

    Provides a common interface for preprocessing images and extracting
    image embeddings.

    Parameters
    ----------
    url : str
        Model identifier or checkpoint location.
    device : str, optional
        Device on which to run the model, by default "cpu".
    """

    def __init__(self, url: str, device: str = "cpu"):
        self.url = url
        self._device = torch.device(device)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"url='{self.url}', "
            f"model={self.model.__class__.__name__})"
        )

    @property
    def device(self) -> torch.device:
        """Return the device on which the model is located."""
        return next(self.model.parameters()).device

    def to(self, device: str | torch.device):
        """Move the model to the specified device."""
        self.model.to(device)
        self._device = torch.device(device)
        return self

    @abstractmethod
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode preprocessed images into feature vectors.

        Parameters
        ----------
        img : torch.Tensor
            Preprocessed image tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Image embeddings of shape (B, D).
        """
        pass

    @abstractmethod
    def preprocess(
        self,
        img: Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """
        Preprocess one or more PIL images.

        Returns
        -------
        torch.Tensor
            Batched image tensor on the model's device.
        """
        pass


class DINOv2(AbstractVisionModel):
    """
    DINOv2 vision foundation model implementation.

    Uses Hugging Face Transformers to extract image embeddings
    from pretrained DINOv2 models.

    Parameters
    ----------
    url : str, optional
        Hugging Face model identifier, by default "facebook/dinov2-base".
    device : str, optional
        Device on which to run the model, by default "cpu".
    pooling : {"cls", "mean"}, optional
        Embedding pooling strategy, by default "mean".
    normalize : bool, optional
        Whether to L2-normalize embeddings, by default True.
    **kwargs
        Additional keyword arguments passed to AutoModel.from_pretrained.

    Attributes
    ----------
    model : torch.nn.Module
        Loaded DINOv2 model.
    preprocessor : callable
        Hugging Face image processor.
    pooling : str
        Pooling strategy used for image embeddings.
    normalize : bool
        Whether embeddings are L2-normalized.
    """

    def __init__(
        self,
        url: str = "facebook/dinov2-base",
        device: str = "cpu",
        pooling: str = "mean",
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(url=url, device=device)

        if pooling not in ("cls", "mean"):
            raise ValueError(
                f"Unsupported pooling strategy: {pooling}"
            )

        self.model = AutoModel.from_pretrained(
            url, **kwargs
        ).eval().to(device)

        self.preprocessor = AutoImageProcessor.from_pretrained(url)

        self.pooling = pooling
        self.normalize = normalize

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor) -> torch.Tensor:
        """
        Encode images into DINOv2 feature vectors.

        Parameters
        ----------
        img : torch.Tensor
            Preprocessed image tensor of shape (B, 3, H, W).

        Returns
        -------
        torch.Tensor
            Image embeddings of shape (B, D).
        """
        outputs = self.model(pixel_values=img)

        hidden_states = outputs.last_hidden_state

        if self.pooling == "cls":
            embeddings = hidden_states[:, 0, :]

        elif self.pooling == "mean":
            # Exclude the CLS token and average patch tokens.
            embeddings = hidden_states[:, 1:, :].mean(dim=1)

        if self.normalize:
            embeddings = F.normalize(
                embeddings, p=2, dim=-1
            )

        return embeddings

    def preprocess(
        self,
        img: Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """
        Preprocess PIL images for DINOv2.

        Parameters
        ----------
        img : Image.Image or list[Image.Image]
            Single image or list of images.

        Returns
        -------
        torch.Tensor
            Preprocessed images of shape (B, 3, H, W),
            moved to the model's device.
        """
        if not isinstance(img, list):
            img = [img]

        img = [image.convert("RGB") for image in img]

        inputs = self.preprocessor(
            images=img,
            return_tensors="pt",
        )

        return inputs["pixel_values"].to(self.device)


class ConvNeXt(AbstractVisionModel):
    """
    ConvNeXt vision foundation model implementation.

    Extracts image embeddings by pooling and concatenating
    intermediate convolutional feature maps.

    Parameters
    ----------
    url : str
        Model identifier from the timm model registry.
    device : str, optional
        Device on which to run the model.
    stages : tuple[int], optional
        Indices of feature extraction stages (zero-based).
    pooling : str, optional
        Pooling strategy: "mean", "mean_std", or "spatial".
    grid_size : int, optional
        Spatial grid size used for spatial pooling.
    normalize : bool, optional
        Whether to L2-normalize embeddings.
    **kwargs
        Additional arguments passed to timm.create_model.
    """

    def __init__(
        self,
        url="convnext_tiny.in12k",
        device="cpu",
        stages=(0, 1, 2),
        pooling="mean_std",
        grid_size=2,
        normalize=True,
        **kwargs,
    ):
        super().__init__(url=url, device=device)

        if pooling not in ("mean", "mean_std", "spatial"):
            raise ValueError(f"Unsupported pooling: {pooling}")

        self.model = timm.create_model(
            url,
            pretrained=True,
            features_only=True,
            out_indices=stages,
            **kwargs,
        ).eval().to(device)

        config = timm.data.resolve_model_data_config(self.model)

        self.preprocessor = timm.data.create_transform(
            **config,
            is_training=False,
        )

        self.stages = stages
        self.pooling = pooling
        self.grid_size = grid_size
        self.normalize = normalize

    @property
    def embedding_dim(self):
        """Return the number of embedding dimensions."""
        channels = sum(self.model.feature_info.channels())

        if self.pooling == "mean_std":
            return channels * 2

        if self.pooling == "spatial":
            return channels * self.grid_size ** 2

        return channels

    def pool_features(self, feature: torch.Tensor):
        """Convert a feature map into an embedding."""

        if self.pooling == "mean":
            return feature.mean(dim=(2, 3))

        if self.pooling == "mean_std":
            mean = feature.mean(dim=(2, 3))
            std = feature.std(dim=(2, 3), unbiased=False)

            return torch.cat([mean, std], dim=-1)

        if self.pooling == "spatial":
            pooled = F.adaptive_avg_pool2d(
                feature,
                (self.grid_size, self.grid_size),
            )
            return pooled.flatten(start_dim=1)

    @torch.no_grad()
    def encode_image(self, img: torch.Tensor):
        """
        Encode preprocessed images into feature vectors.

        Input shape: (B, C, H, W).
        Output shape: (B, D).
        """
        features = self.model(img)

        embeddings = []

        for feature in features:
            embedding = self.pool_features(feature)

            if self.normalize:
                embedding = F.normalize(embedding, dim=-1)

            embeddings.append(embedding)

        embedding = torch.cat(embeddings, dim=-1)

        if self.normalize:
            embedding = F.normalize(embedding, dim=-1)

        return embedding

    def preprocess(
        self,
        img: Image.Image | list[Image.Image],
    ) -> torch.Tensor:
        """Preprocess images and move them to the model device."""

        if not isinstance(img, list):
            img = [img]

        inputs = torch.stack([
            self.preprocessor(image.convert("RGB"))
            for image in img
        ])

        return inputs.to(self.device)
