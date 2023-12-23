import torch
from swarms.utils.load_model_torch import load_model_torch


def prep_torch_inference(
    model_path: str = None,
    device: torch.device = None,
    *args,
    **kwargs,
):
    """
    Prepare a Torch model for inference.

    Args:
        model_path (str): Path to the model file.
        device (torch.device): Device to run the model on.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        torch.nn.Module: The prepared model.
    """
    try:
        model = load_model_torch(model_path, device)
        model.eval()
        return model
    except Exception as e:
        # Add error handling code here
        print(f"Error occurred while preparing Torch model: {e}")
        return None
