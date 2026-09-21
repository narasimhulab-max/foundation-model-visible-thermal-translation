import numpy as np
import pytest
import torch

from PIL import Image

from preprocessing.image_preprocessor import (
    ImagePreprocessor,
    PreprocessingConfig,
)


@pytest.fixture
def preprocessor():
    config = PreprocessingConfig(
        image_size=(256, 256)
    )

    return ImagePreprocessor(config)


def test_thermal_preprocessing(
    preprocessor,
):

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (480, 640),
            dtype=np.uint8,
        ),
        mode="L",
    )

    tensor = preprocessor.preprocess(
        image,
        modality="thermal",
    )

    assert isinstance(
        tensor,
        torch.Tensor,
    )

    assert tensor.shape == (
        1,
        1,
        256,
        256,
    )

    assert tensor.dtype == torch.float32

    assert torch.isfinite(tensor).all()

    assert tensor.min() >= -1.0

    assert tensor.max() <= 1.0


def test_night_vision_preprocessing(
    preprocessor,
):

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (720, 1280, 3),
            dtype=np.uint8,
        ),
        mode="RGB",
    )

    tensor = preprocessor.preprocess(
        image,
        modality="night_vision",
    )

    assert isinstance(
        tensor,
        torch.Tensor,
    )

    assert tensor.shape == (
        1,
        3,
        256,
        256,
    )

    assert tensor.dtype == torch.float32

    assert torch.isfinite(tensor).all()

    assert tensor.min() >= -1.0

    assert tensor.max() <= 1.0


def test_thermal_path_loading(
    preprocessor,
    tmp_path,
):

    image_path = tmp_path / "thermal.png"

    image = Image.fromarray(
        np.random.randint(
            0,
            256,
            (300, 400),
            dtype=np.uint8,
        ),
        mode="L",
    )

    image.save(image_path)

    tensor = preprocessor.preprocess(
        image_path,
        modality="thermal",
    )

    assert tensor.shape == (
        1,
        1,
        256,
        256,
    )


def test_invalid_modality(
    preprocessor,
):

    image = Image.fromarray(
        np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
        mode="L",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported modality",
    ):

        preprocessor.preprocess(
            image,
            modality="rgb_camera",
        )


def test_missing_file(
    preprocessor,
):

    with pytest.raises(
        FileNotFoundError,
    ):

        preprocessor.preprocess(
            "does_not_exist.png",
            modality="thermal",
        )


def test_modality_alias(
    preprocessor,
):

    image = Image.fromarray(
        np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
        mode="L",
    )

    tensor = preprocessor.preprocess(
        image,
        modality="infrared",
    )

    assert tensor.shape == (
        1,
        1,
        256,
        256,
    )


def test_output_range(
    preprocessor,
):

    image = Image.fromarray(
        np.full(
            (100, 100),
            255,
            dtype=np.uint8,
        ),
        mode="L",
    )

    tensor = preprocessor.preprocess(
        image,
        modality="thermal",
    )

    assert torch.allclose(
        tensor,
        torch.ones_like(tensor),
    )


def test_black_image_normalization(
    preprocessor,
):

    image = Image.fromarray(
        np.zeros(
            (100, 100),
            dtype=np.uint8,
        ),
        mode="L",
    )

    tensor = preprocessor.preprocess(
        image,
        modality="thermal",
    )

    assert torch.allclose(
        tensor,
        -torch.ones_like(tensor),
    )