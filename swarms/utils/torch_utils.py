import torch


def autodetect_device():
    """
    Autodetects the device to use for inference.

    Returns
    -------
        str
            The device to use for inference.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
