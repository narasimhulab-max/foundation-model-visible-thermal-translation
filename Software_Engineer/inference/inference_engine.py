from __future__ import annotations

import torch

from inference.device import resolve_device
from src.model_registry import is_model_compatible


class InferenceEngine:
    """
    Application-level inference interface.

    The UI should communicate with this class instead of
    directly calling an underlying ML model.
    """

    def __init__(
        self,
        model,
        model_name: str,
        device: str = "auto",
    ) -> None:

        self.model = model
        self.model_name = model_name
        self.device = resolve_device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(
        self,
        input_tensor: torch.Tensor,
        modality: str,
    ) -> torch.Tensor:
        """
        Run model inference.

        Parameters
        ----------
        input_tensor:
            Preprocessed BCHW tensor.

        modality:
            Input modality.

        Returns
        -------
        torch.Tensor
            Model output tensor.
        """

        if not is_model_compatible(
            self.model_name,
            modality,
        ):
            raise ValueError(
                f"Model '{self.model_name}' does not support "
                f"modality '{modality}'."
            )

        self._validate_input(input_tensor)

        input_tensor = input_tensor.to(
            self.device,
            non_blocking=True,
        )

        output = self.model(input_tensor)

        self._validate_output(output)

        return output

    @staticmethod
    def _validate_input(
        input_tensor: torch.Tensor,
    ) -> None:

        if not isinstance(
            input_tensor,
            torch.Tensor,
        ):
            raise TypeError(
                "Inference input must be a torch.Tensor."
            )

        if input_tensor.ndim != 4:
            raise ValueError(
                "Inference input must use BCHW format. "
                f"Received shape: {tuple(input_tensor.shape)}"
            )

        if input_tensor.shape[0] < 1:
            raise ValueError(
                "Inference input batch cannot be empty."
            )

        if not torch.is_floating_point(
            input_tensor
        ):
            raise TypeError(
                "Inference input must use a floating-point dtype."
            )

        if not torch.isfinite(
            input_tensor
        ).all():
            raise ValueError(
                "Inference input contains NaN or infinite values."
            )

    @staticmethod
    def _validate_output(
        output: torch.Tensor,
    ) -> None:

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise TypeError(
                "Model output must be a torch.Tensor."
            )

        if output.ndim != 4:
            raise ValueError(
                "Model output must use BCHW format. "
                f"Received shape: {tuple(output.shape)}"
            )

        if output.shape[1] != 3:
            raise ValueError(
                "Model output must contain 3 RGB channels. "
                f"Received {output.shape[1]} channels."
            )

        if not torch.isfinite(
            output
        ).all():
            raise ValueError(
                "Model output contains NaN or infinite values."
            )