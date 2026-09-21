from pathlib import Path

from src.config import SUPPORTED_IMAGE_FORMATS


# Supported input modalities
SUPPORTED_MODALITIES = {
    "thermal",
    "night_vision",
}


def is_supported_image(filename: str) -> bool:
    """
    Check whether the uploaded file has a supported image extension.

    Supported formats are defined in src.config.
    """
    suffix = Path(filename).suffix.lower()

    return suffix in SUPPORTED_IMAGE_FORMATS


def is_supported_modality(modality: str) -> bool:
    """
    Check whether the selected input modality is supported.

    Supported modalities:
        - thermal
        - night_vision
    """
    return modality in SUPPORTED_MODALITIES