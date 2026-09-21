from dataclasses import dataclass

from src.validation import (
    is_supported_image,
    is_supported_modality,
)


@dataclass
class InputImage:
    """
    Represents a validated image input to the AI pipeline.
    """

    image: object
    modality: str
    filename: str


def create_input(
    image: object,
    modality: str,
    filename: str,
) -> InputImage:
    """
    Validate and create a structured image input.

    Parameters
    ----------
    image:
        Uploaded image object.

    modality:
        Input modality such as 'thermal' or 'night_vision'.

    filename:
        Original uploaded filename.

    Returns
    -------
    InputImage
        Structured and validated image input.

    Raises
    ------
    ValueError
        If the image format or modality is unsupported.
    """

    if not is_supported_image(filename):
        raise ValueError(
            f"Unsupported image format: {filename}"
        )

    if not is_supported_modality(modality):
        raise ValueError(
            f"Unsupported input modality: {modality}"
        )

    return InputImage(
        image=image,
        modality=modality,
        filename=filename,
    )