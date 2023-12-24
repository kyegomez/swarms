import torch
import logging
from typing import Union, List, Any
from torch.cuda import memory_allocated, memory_reserved


def check_device(
    log_level: Any = logging.INFO,
    memory_threshold: float = 0.8,
    capability_threshold: float = 3.5,
    return_type: str = "list",
) -> Union[torch.device, List[torch.device]]:
    """
    Checks for the availability of CUDA and returns the appropriate device(s).
    If CUDA is not available, returns a CPU device.
    If CUDA is available, returns a list of all available GPU devices.
    """
    logging.basicConfig(level=log_level)

    # Check for CUDA availability
    try:
        if not torch.cuda.is_available():
            logging.info("CUDA is not available. Using CPU...")
            return torch.device("cpu")
    except Exception as e:
        logging.error("Error checking for CUDA availability: ", e)
        return torch.device("cpu")

    logging.info("CUDA is available.")

    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    devices = []
    if num_gpus > 1:
        logging.info(f"Multiple GPUs available: {num_gpus}")
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    else:
        logging.info("Only one GPU is available.")
        devices = [torch.device("cuda")]

    # Check additional properties for each device
    for device in devices:
        try:
            torch.cuda.set_device(device)
            capability = torch.cuda.get_device_capability(device)
            total_memory = torch.cuda.get_device_properties(
                device
            ).total_memory
            allocated_memory = memory_allocated(device)
            reserved_memory = memory_reserved(device)
            device_name = torch.cuda.get_device_name(device)

            logging.info(
                f"Device: {device}, Name: {device_name}, Compute"
                f" Capability: {capability}, Total Memory:"
                f" {total_memory}, Allocated Memory:"
                f" {allocated_memory}, Reserved Memory:"
                f" {reserved_memory}"
            )
        except Exception as e:
            logging.error(
                f"Error retrieving properties for device {device}: ",
                e,
            )

    return devices


# devices = check_device()
# logging.info(f"Using device(s): {devices}")
