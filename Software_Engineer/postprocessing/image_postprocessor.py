from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def tensor_to_rgb_image(
    tensor: torch.Tensor,
) -> Image.Image:
    """
    Convert a model output tensor into a PIL RGB image.

    Expected input:
        [1, 3, H, W]

    Expected model output range:
        [-1, 1]
    """

    if not isinstance(
        tensor,
        torch.Tensor,
    ):
        raise TypeError(
            "Expected a torch.Tensor."
        )

    if tensor.ndim != 4:
        raise ValueError(
            "Expected tensor shape [B, C, H, W]."
        )

    if tensor.shape[0] != 1:
        raise ValueError(
            "Image conversion currently supports "
            "batch size 1 only."
        )

    if tensor.shape[1] != 3:
        raise ValueError(
            "Expected a 3-channel RGB tensor."
        )

    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            "Tensor contains NaN or infinite values."
        )

    tensor = tensor.detach().cpu()

    # [-1, 1] → [0, 1]
    tensor = (
        tensor.clamp(-1.0, 1.0)
        + 1.0
    ) / 2.0

    # BCHW → HWC
    array = tensor[0].permute(
        1,
        2,
        0,
    ).numpy()

    # [0, 1] → [0, 255]
    array = (
        array * 255.0
    ).round().astype(
        np.uint8
    )

    return Image.fromarray(
        array,
        mode="RGB",
    )