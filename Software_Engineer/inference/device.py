from __future__ import annotations

import torch


SUPPORTED_DEVICES = {
    "auto",
    "cpu",
    "cuda",
}


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve the device used for model inference.

    Parameters
    ----------
    requested:
        One of:
        - "auto"
        - "cpu"
        - "cuda"

    Returns
    -------
    torch.device
        Resolved computation device.

    Raises
    ------
    ValueError
        If an unsupported device is requested.

    RuntimeError
        If CUDA is explicitly requested but unavailable.
    """

    requested = requested.strip().lower()

    if requested not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Unsupported device '{requested}'. "
            f"Supported devices: {sorted(SUPPORTED_DEVICES)}"
        )

    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but CUDA is not available."
            )

        return torch.device("cuda")

    # auto mode
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")