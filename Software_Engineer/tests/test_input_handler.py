import pytest

from src.input_handler import create_input


def test_create_thermal_input():
    image = object()

    result = create_input(
        image=image,
        modality="thermal",
        filename="thermal.png",
    )

    assert result.image is image
    assert result.modality == "thermal"
    assert result.filename == "thermal.png"


def test_create_night_vision_input():
    image = object()

    result = create_input(
        image=image,
        modality="night_vision",
        filename="night.png",
    )

    assert result.image is image
    assert result.modality == "night_vision"
    assert result.filename == "night.png"


def test_invalid_modality_raises_error():
    image = object()

    with pytest.raises(ValueError):
        create_input(
            image=image,
            modality="rgb",
            filename="image.png",
        )


def test_invalid_file_raises_error():
    image = object()

    with pytest.raises(ValueError):
        create_input(
            image=image,
            modality="thermal",
            filename="image.pdf",
        )