from __future__ import annotations


MODEL_MODALITY_COMPATIBILITY = {
    "pix2pix_baseline": {
        "night_vision",
    },
}


def is_model_compatible(
    model_name: str,
    modality: str,
) -> bool:
    """
    Check whether a model supports a given input modality.
    """

    supported_modalities = MODEL_MODALITY_COMPATIBILITY.get(
        model_name,
        set(),
    )

    return modality in supported_modalities


def get_supported_modalities(
    model_name: str,
) -> set[str]:
    """
    Return modalities supported by a model.
    """

    return MODEL_MODALITY_COMPATIBILITY.get(
        model_name,
        set(),
    ).copy()