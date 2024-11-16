import os
from typing import Any


from clusterops import (
    execute_on_gpu,
    execute_on_multiple_gpus,
    execute_with_cpu_cores,
    list_available_gpus,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="clusterops_wrapper")


def exec_callable_with_clusterops(
    device: str = "cpu",
    device_id: int = 0,
    all_cores: bool = True,
    all_gpus: bool = False,
    func: callable = None,
    *args,
    **kwargs,
) -> Any:
    """
    Executes a given function on a specified device, either CPU or GPU.

    This method attempts to execute a given function on a specified device, either CPU or GPU. It logs the device selection and the number of cores or GPU ID used. If the device is set to CPU, it can use all available cores or a specific core specified by `device_id`. If the device is set to GPU, it uses the GPU specified by `device_id`.

    Args:
        device (str, optional): The device to use for execution. Defaults to "cpu".
        device_id (int, optional): The ID of the GPU to use if device is set to "gpu". Defaults to 0.
        all_cores (bool, optional): If True, uses all available CPU cores. Defaults to True.
        all_gpus (bool, optional): If True, uses all available GPUs. Defaults to False.
        func (callable): The function to execute.
        *args: Additional positional arguments to be passed to the execution method.
        **kwargs: Additional keyword arguments to be passed to the execution method.

    Returns:
        Any: The result of the execution.

    Raises:
        ValueError: If an invalid device is specified.
        Exception: If any other error occurs during execution.
    """
    try:
        logger.info(f"Attempting to run on device: {device}")
        if device == "cpu":
            logger.info("Device set to CPU")
            if all_cores is True:
                count = os.cpu_count()
                logger.info(f"Using all available CPU cores: {count}")
            else:
                count = device_id
                logger.info(f"Using specific CPU core: {count}")

            return execute_with_cpu_cores(
                count, func, *args, **kwargs
            )

        # If device gpu
        elif device == "gpu":
            logger.info("Device set to GPU")
            return execute_on_gpu(device_id, func, *args, **kwargs)
        elif device == "gpu" and all_gpus is True:
            logger.info("Device set to GPU and running all gpus")
            gpus = [int(gpu) for gpu in list_available_gpus()]
            return execute_on_multiple_gpus(
                gpus, func, *args, **kwargs
            )
        else:
            raise ValueError(
                f"Invalid device specified: {device}. Supported devices are 'cpu' and 'gpu'."
            )
    except ValueError as e:
        logger.error(f"Invalid device specified: {e}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
        raise e
