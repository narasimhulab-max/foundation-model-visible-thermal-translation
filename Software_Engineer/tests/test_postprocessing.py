import pytest
import torch

from postprocessing.image_postprocessor import (
    tensor_to_rgb_image,
)


def test_tensor_to_rgb_image():
    tensor = torch.zeros(
        1,
        3,
        256,
        256,
    )

    image = tensor_to_rgb_image(
        tensor
    )

    assert image.mode == "RGB"
    assert image.size == (
        256,
        256,
    )


def test_wrong_dimensions_rejected():
    tensor = torch.zeros(
        3,
        256,
        256,
    )

    with pytest.raises(ValueError):
        tensor_to_rgb_image(
            tensor
        )


def test_wrong_channels_rejected():
    tensor = torch.zeros(
        1,
        1,
        256,
        256,
    )

    with pytest.raises(ValueError):
        tensor_to_rgb_image(
            tensor
        )


def test_batch_greater_than_one_rejected():
    tensor = torch.zeros(
        2,
        3,
        256,
        256,
    )

    with pytest.raises(ValueError):
        tensor_to_rgb_image(
            tensor
        )


def test_nan_rejected():
    tensor = torch.zeros(
        1,
        3,
        256,
        256,
    )

    tensor[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        tensor_to_rgb_image(
            tensor
        )