import torch
from torch import nn


def load_model_torch(
    model_path: str = None,
    device: torch.device = None,
    model: nn.Module = None,
    strict: bool = True,
    map_location=None,
    *args,
    **kwargs,
) -> nn.Module:
    """
    Load a PyTorch model from a given path and move it to the specified device.

    Args:
        model_path (str): Path to the saved model file.
        device (torch.device): Device to move the model to.
        model (nn.Module): The model architecture, if the model file only contains the state dictionary.
        strict (bool): Whether to strictly enforce that the keys in the state dictionary match the keys returned by the model's
        `state_dict()` function.
        map_location (callable): A function to remap the storage locations of the loaded model.
        *args: Additional arguments to pass to `torch.load`.
        **kwargs: Additional keyword arguments to pass to `torch.load`.

    Returns:
        nn.Module: The loaded model.

    Raises:
        FileNotFoundError: If the model file is not found.
        RuntimeError: If there is an error while loading the model.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    try:
        if model is None:
            model = torch.load(
                model_path, map_location=map_location, *args, **kwargs
            )
        else:
            model.load_state_dict(
                torch.load(
                    model_path,
                    map_location=map_location,
                    *args,
                    **kwargs,
                ),
                strict=strict,
            )
        return model.to(device)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except RuntimeError as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
