import torch
from PIL import Image

from inference.inference_engine import InferenceEngine
from postprocessing.image_postprocessor import (
    tensor_to_rgb_image,
)


class FakePix2Pix:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, x):
        return torch.zeros(
            x.shape[0],
            3,
            x.shape[2],
            x.shape[3],
            device=x.device,
        )


def test_end_to_end_inference_pipeline():

    model = FakePix2Pix()

    engine = InferenceEngine(
        model=model,
        model_name="pix2pix_baseline",
        device="cpu",
    )

    input_tensor = torch.randn(
        1,
        1,
        256,
        256,
    )

    output_tensor = engine.predict(
        input_tensor,
        modality="night_vision",
    )

    output_image = tensor_to_rgb_image(
        output_tensor
    )

    assert isinstance(
        output_image,
        Image.Image,
    )

    assert output_image.mode == "RGB"

    assert output_image.size == (
        256,
        256,
    )