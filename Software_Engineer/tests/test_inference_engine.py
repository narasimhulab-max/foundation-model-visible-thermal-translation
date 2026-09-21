import pytest
import torch

from inference.inference_engine import InferenceEngine


class FakePix2Pix:
    """
    Lightweight fake model used to test the inference
    software layer without loading a real checkpoint.
    """

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        batch_size = x.shape[0]
        height = x.shape[2]
        width = x.shape[3]

        return torch.zeros(
            batch_size,
            3,
            height,
            width,
            dtype=torch.float32,
            device=x.device,
        )


def create_engine():
    return InferenceEngine(
        model=FakePix2Pix(),
        model_name="pix2pix_baseline",
        device="cpu",
    )


def test_valid_inference():
    engine = create_engine()

    input_tensor = torch.randn(
        1,
        1,
        256,
        256,
    )

    output = engine.predict(
        input_tensor,
        modality="night_vision",
    )

    assert output.shape == (
        1,
        3,
        256,
        256,
    )


def test_invalid_input_type():
    engine = create_engine()

    with pytest.raises(TypeError):
        engine.predict(
            "not a tensor",
            modality="night_vision",
        )


def test_invalid_input_dimensions():
    engine = create_engine()

    tensor = torch.randn(
        1,
        256,
        256,
    )

    with pytest.raises(ValueError):
        engine.predict(
            tensor,
            modality="night_vision",
        )


def test_nan_input_rejected():
    engine = create_engine()

    tensor = torch.randn(
        1,
        1,
        256,
        256,
    )

    tensor[0, 0, 0, 0] = float("nan")

    with pytest.raises(ValueError):
        engine.predict(
            tensor,
            modality="night_vision",
        )


def test_thermal_rejected_by_current_pix2pix():
    engine = create_engine()

    tensor = torch.randn(
        1,
        1,
        256,
        256,
    )

    with pytest.raises(ValueError):
        engine.predict(
            tensor,
            modality="thermal",
        )