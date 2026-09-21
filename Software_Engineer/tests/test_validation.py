from src.validation import (
    is_supported_image,
    is_supported_modality,
)


def test_png_is_supported():
    assert is_supported_image("thermal.png")


def test_jpg_is_supported():
    assert is_supported_image("thermal.jpg")


def test_jpeg_is_supported():
    assert is_supported_image("thermal.jpeg")


def test_pdf_is_not_supported():
    assert not is_supported_image("thermal.pdf")


def test_thermal_modality_is_supported():
    assert is_supported_modality("thermal")


def test_night_vision_modality_is_supported():
    assert is_supported_modality("night_vision")


def test_invalid_modality_is_not_supported():
    assert not is_supported_modality("rgb")