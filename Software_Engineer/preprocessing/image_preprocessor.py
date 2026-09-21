"""
Production-grade image preprocessing pipeline for
cross-spectral thermal-to-visible image translation.

Supported modalities:
    - thermal
    - night_vision

Pipeline:
    Image
      -> Load
      -> Validate
      -> Channel handling
      -> Resize
      -> Normalize
      -> PyTorch tensor
      -> Batch dimension
      -> Model-ready tensor [B, C, H, W]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image


PathLike = Union[str, Path]


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration for image preprocessing.

    Attributes:
        image_size:
            Target spatial resolution (height, width).

        thermal_channels:
            Number of channels expected by the thermal pipeline.

        night_vision_channels:
            Number of channels expected by the night-vision pipeline.

        normalize_min:
            Minimum intensity after normalization.

        normalize_max:
            Maximum intensity after normalization.

        output_dtype:
            PyTorch tensor dtype used by the model.
    """

    image_size: tuple[int, int] = (256, 256)

    thermal_channels: int = 1
    night_vision_channels: int = 3

    normalize_min: float = -1.0
    normalize_max: float = 1.0

    output_dtype: torch.dtype = torch.float32


class ImagePreprocessor:
    """
    Converts raw thermal/night-vision images into model-ready tensors.

    Output format:
        [B, C, H, W]

    where:
        B = batch size
        C = number of channels
        H = target height
        W = target width
    """

    SUPPORTED_MODALITIES = {
        "thermal",
        "night_vision",
    }

    SUPPORTED_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:

        self.config = config or PreprocessingConfig()

        self._validate_config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(
        self,
        image: Image.Image | PathLike,
        modality: str,
    ) -> torch.Tensor:
        """
        Complete preprocessing pipeline.

        Args:
            image:
                PIL image or path to an image file.

            modality:
                Either "thermal" or "night_vision".

        Returns:
            Model-ready tensor with shape [1, C, H, W].

        Raises:
            ValueError:
                If modality or image format is invalid.
        """

        modality = self._normalize_modality(modality)

        pil_image = self._load_image(image)

        pil_image = self._convert_channels(
            pil_image,
            modality,
        )

        pil_image = self._resize(
            pil_image,
        )

        array = self._to_numpy(
            pil_image,
        )

        array = self._normalize(
            array,
        )

        tensor = self._to_tensor(
            array,
            modality,
        )

        tensor = self._add_batch_dimension(
            tensor,
        )

        self._validate_output(
            tensor,
            modality,
        )

        return tensor

    # ------------------------------------------------------------------
    # Modality
    # ------------------------------------------------------------------

    def _normalize_modality(
        self,
        modality: str,
    ) -> str:

        if not isinstance(modality, str):
            raise TypeError(
                "modality must be a string."
            )

        normalized = modality.strip().lower()

        aliases = {
            "ir": "thermal",
            "infrared": "thermal",
            "thermal_image": "thermal",
            "thermal/infrared": "thermal",
            "nightvision": "night_vision",
            "night vision": "night_vision",
            "night-vision": "night_vision",
        }

        normalized = aliases.get(
            normalized,
            normalized,
        )

        if normalized not in self.SUPPORTED_MODALITIES:
            raise ValueError(
                f"Unsupported modality '{modality}'. "
                f"Supported modalities: "
                f"{sorted(self.SUPPORTED_MODALITIES)}"
            )

        return normalized

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _load_image(
        self,
        image: Image.Image | PathLike,
    ) -> Image.Image:

        if isinstance(image, Image.Image):

            return image.copy()

        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Expected an image file, got: {image_path}"
            )

        if image_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: "
                f"{image_path.suffix}"
            )

        try:
            with Image.open(image_path) as img:
                return img.copy()

        except Exception as exc:
            raise ValueError(
                f"Unable to read image: {image_path}"
            ) from exc

    # ------------------------------------------------------------------
    # Channel conversion
    # ------------------------------------------------------------------

    def _convert_channels(
        self,
        image: Image.Image,
        modality: str,
    ) -> Image.Image:

        if modality == "thermal":

            # Thermal images are represented as one-channel
            # intensity information.
            return image.convert("L")

        if modality == "night_vision":

            # Night-vision images entering the current
            # multimodal pipeline are represented as RGB.
            return image.convert("RGB")

        raise ValueError(
            f"Unsupported modality: {modality}"
        )

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------

    def _resize(
        self,
        image: Image.Image,
    ) -> Image.Image:

        width, height = self.config.image_size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Image dimensions must be positive."
            )

        return image.resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )

    # ------------------------------------------------------------------
    # NumPy conversion
    # ------------------------------------------------------------------

    def _to_numpy(
        self,
        image: Image.Image,
    ) -> np.ndarray:

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if array.ndim == 2:
            array = array[..., np.newaxis]

        if array.ndim != 3:
            raise ValueError(
                f"Expected image array with 3 dimensions "
                f"(H, W, C), got shape {array.shape}"
            )

        return array

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _normalize(
        self,
        array: np.ndarray,
    ) -> np.ndarray:

        # Input image values are expected to be in [0, 255].
        array = array / 255.0

        # Convert [0, 1] -> [-1, 1].
        minimum = self.config.normalize_min
        maximum = self.config.normalize_max

        if minimum == -1.0 and maximum == 1.0:
            array = (array * 2.0) - 1.0

        else:
            array = (
                array * (maximum - minimum)
            ) + minimum

        return array.astype(
            np.float32,
            copy=False,
        )

    # ------------------------------------------------------------------
    # Tensor conversion
    # ------------------------------------------------------------------

    def _to_tensor(
        self,
        array: np.ndarray,
        modality: str,
    ) -> torch.Tensor:

        # HWC -> CHW
        tensor = torch.from_numpy(
            array
        ).permute(
            2,
            0,
            1,
        )

        tensor = tensor.to(
            dtype=self.config.output_dtype
        )

        expected_channels = (
            self.config.thermal_channels
            if modality == "thermal"
            else self.config.night_vision_channels
        )

        if tensor.shape[0] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels "
                f"for modality '{modality}', "
                f"got {tensor.shape[0]}."
            )

        return tensor

    # ------------------------------------------------------------------
    # Batch dimension
    # ------------------------------------------------------------------

    def _add_batch_dimension(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:

        return tensor.unsqueeze(0)

    # ------------------------------------------------------------------
    # Output validation
    # ------------------------------------------------------------------

    def _validate_output(
        self,
        tensor: torch.Tensor,
        modality: str,
    ) -> None:

        expected_channels = (
            self.config.thermal_channels
            if modality == "thermal"
            else self.config.night_vision_channels
        )

        expected_height = self.config.image_size[1]
        expected_width = self.config.image_size[0]

        expected_shape = (
            1,
            expected_channels,
            expected_height,
            expected_width,
        )

        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                "Preprocessing produced an unexpected "
                f"tensor shape. Expected "
                f"{expected_shape}, got "
                f"{tuple(tensor.shape)}."
            )

        if not torch.isfinite(tensor).all():
            raise ValueError(
                "Preprocessed tensor contains "
                "NaN or infinite values."
            )

        minimum = self.config.normalize_min
        maximum = self.config.normalize_max

        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()

        tolerance = 1e-5

        if tensor_min < minimum - tolerance:
            raise ValueError(
                f"Tensor minimum {tensor_min} is below "
                f"the expected minimum {minimum}."
            )

        if tensor_max > maximum + tolerance:
            raise ValueError(
                f"Tensor maximum {tensor_max} is above "
                f"the expected maximum {maximum}."
            )

    # ------------------------------------------------------------------
    # Configuration validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:

        width, height = self.config.image_size

        if width <= 0 or height <= 0:
            raise ValueError(
                "image_size values must be positive."
            )

        if self.config.thermal_channels != 1:
            raise ValueError(
                "Current thermal preprocessing expects "
                "one channel."
            )

        if self.config.night_vision_channels != 3:
            raise ValueError(
                "Current night-vision preprocessing expects "
                "three channels."
            )

        if (
            self.config.normalize_min
            >= self.config.normalize_max
        ):
            raise ValueError(
                "normalize_min must be smaller than "
                "normalize_max."
            )