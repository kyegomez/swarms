import os
import psutil
from typing import Callable, Any
from loguru import logger


def run_on_cpu(func: Callable, *args, **kwargs) -> Any:
    """
    Executes a callable function on the CPU, ensuring it doesn't run on a GPU.
    This is achieved by setting the CPU affinity of the current process.

    Args:
        func (Callable): The function to be executed.
        *args: Variable length argument list to pass to the function.
        **kwargs: Arbitrary keyword arguments to pass to the function.

    Returns:
        Any: The result of the function execution.

    Raises:
        RuntimeError: If the CPU affinity cannot be set or restored.
    """
    try:
        # Get the current process
        process = psutil.Process(os.getpid())

        # Check if the platform supports cpu_affinity
        if not hasattr(process, "cpu_affinity"):
            logger.warning(
                "CPU affinity is not supported on this platform. Executing function without setting CPU affinity."
            )
            return func(*args, **kwargs)

        # Save the original CPU affinity
        original_affinity = process.cpu_affinity()
        logger.info(f"Original CPU affinity: {original_affinity}")

        try:
            # Set the CPU affinity to all available CPU cores (ensuring it's CPU-bound)
            all_cpus = list(range(os.cpu_count()))
            process.cpu_affinity(all_cpus)
            logger.info(f"Set CPU affinity to: {all_cpus}")

            # Run the function with provided arguments
            result = func(*args, **kwargs)

        except psutil.AccessDenied as e:
            logger.error(
                "Access denied while setting CPU affinity", exc_info=True
            )
            raise RuntimeError(
                "Access denied while setting CPU affinity"
            ) from e

        except psutil.NoSuchProcess as e:
            logger.error("Process does not exist", exc_info=True)
            raise RuntimeError("Process does not exist") from e

        except Exception as e:
            logger.error(
                "An error occurred during function execution",
                exc_info=True,
            )
            raise RuntimeError(
                "An error occurred during function execution"
            ) from e

        finally:
            # Restore the original CPU affinity
            try:
                process.cpu_affinity(original_affinity)
                logger.info(
                    f"Restored original CPU affinity: {original_affinity}"
                )
            except Exception as e:
                logger.error(
                    "Failed to restore CPU affinity", exc_info=True
                )
                raise RuntimeError("Failed to restore CPU affinity") from e

        return result

    except psutil.Error as e:
        logger.error("A psutil error occurred", exc_info=True)
        raise RuntimeError("A psutil error occurred") from e

    except Exception as e:
        logger.error("An unexpected error occurred", exc_info=True)
        raise RuntimeError("An unexpected error occurred") from e


# # Example usage
# def example_function(x: int, y: int) -> int:
#     return x + y

# if __name__ == "__main__":
#     try:
#         result = run_on_cpu(example_function, 3, 4)
#         print(f"Result: {result}")
#     except RuntimeError as e:
#         logger.error(f"Error occurred: {e}")
