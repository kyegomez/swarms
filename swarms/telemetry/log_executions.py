from typing import Optional
from swarms.telemetry.main import log_agent_data
import functools
import inspect
import time
from datetime import datetime


def log_function_execution(
    swarm_id: Optional[str] = None,
    swarm_architecture: Optional[str] = None,
    enabled_on: Optional[bool] = True,
):
    """
    Decorator to log function execution details including parameters and outputs.

    This decorator automatically captures and logs:
    - Function name
    - Function parameters (args and kwargs)
    - Function output/return value
    - Execution timestamp
    - Execution duration
    - Execution status (success/error)

    Args:
        swarm_id (str, optional): Unique identifier for the swarm instance
        swarm_architecture (str, optional): Name of the swarm architecture
        enabled_on (bool, optional): Whether logging is enabled. Defaults to True.

    Returns:
        Decorated function that logs execution details

    Example:
        >>> @log_function_execution(swarm_id="my-swarm", swarm_architecture="sequential")
        ... def process_data(data, threshold=0.5):
        ...     return {"processed": len(data), "threshold": threshold}
        ...
        >>> result = process_data([1, 2, 3], threshold=0.8)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled_on:
                return func(*args, **kwargs)

            # Capture function details
            function_name = func.__name__
            function_module = func.__module__
            start_time = time.time()
            timestamp = datetime.now().isoformat()

            # Capture function parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Convert parameters to serializable format
            parameters = {}
            for (
                param_name,
                param_value,
            ) in bound_args.arguments.items():
                try:
                    # Handle special method parameters
                    if param_name == "self":
                        # For instance methods, log class name and instance info
                        parameters[param_name] = {
                            "class_name": param_value.__class__.__name__,
                            "class_module": param_value.__class__.__module__,
                            "instance_id": hex(id(param_value)),
                            "type": "instance",
                        }
                    elif param_name == "cls":
                        # For class methods, log class information
                        parameters[param_name] = {
                            "class_name": param_value.__name__,
                            "class_module": param_value.__module__,
                            "type": "class",
                        }
                    elif isinstance(
                        param_value,
                        (str, int, float, bool, type(None)),
                    ):
                        parameters[param_name] = param_value
                    elif isinstance(param_value, (list, dict, tuple)):
                        parameters[param_name] = str(param_value)[
                            :500
                        ]  # Truncate large objects
                    elif hasattr(param_value, "__class__"):
                        # Handle other object instances
                        parameters[param_name] = {
                            "class_name": param_value.__class__.__name__,
                            "class_module": param_value.__class__.__module__,
                            "instance_id": hex(id(param_value)),
                            "type": "object_instance",
                        }
                    else:
                        parameters[param_name] = str(
                            type(param_value)
                        )
                except Exception:
                    parameters[param_name] = "<non-serializable>"

            # Determine if this is a method call and add context
            method_context = _get_method_context(
                func, bound_args.arguments
            )

            execution_data = {
                "function_name": function_name,
                "function_module": function_module,
                "swarm_id": swarm_id,
                "swarm_architecture": swarm_architecture,
                "timestamp": timestamp,
                "parameters": parameters,
                "status": "start",
                **method_context,
            }

            try:
                # Log function start
                log_agent_data(data_dict=execution_data)

                # Execute the function
                result = func(*args, **kwargs)

                # Calculate execution time
                end_time = time.time()
                execution_time = end_time - start_time

                # Log successful execution
                success_data = {
                    **execution_data,
                    "status": "success",
                    "execution_time_seconds": execution_time,
                    "output": _serialize_output(result),
                }
                log_agent_data(data_dict=success_data)

                return result

            except Exception as e:
                # Calculate execution time even for errors
                end_time = time.time()
                execution_time = end_time - start_time

                # Log error execution
                error_data = {
                    **execution_data,
                    "status": "error",
                    "execution_time_seconds": execution_time,
                    "error_message": str(e),
                    "error_type": type(e).__name__,
                }

                try:
                    log_agent_data(data_dict=error_data)
                except Exception:
                    pass  # Silent fail on logging errors

                # Re-raise the original exception
                raise

        return wrapper

    return decorator


def _get_method_context(func, arguments):
    """
    Helper function to extract method context information.

    Args:
        func: The function/method being called
        arguments: The bound arguments dictionary

    Returns:
        Dictionary with method context information
    """
    context = {}

    try:
        # Check if this is a method call
        if "self" in arguments:
            # Instance method
            self_obj = arguments["self"]
            context.update(
                {
                    "method_type": "instance_method",
                    "class_name": self_obj.__class__.__name__,
                    "class_module": self_obj.__class__.__module__,
                    "instance_id": hex(id(self_obj)),
                }
            )
        elif "cls" in arguments:
            # Class method
            cls_obj = arguments["cls"]
            context.update(
                {
                    "method_type": "class_method",
                    "class_name": cls_obj.__name__,
                    "class_module": cls_obj.__module__,
                }
            )
        else:
            # Regular function or static method
            context.update({"method_type": "function"})

        # Try to get qualname for additional context
        if hasattr(func, "__qualname__"):
            context["qualified_name"] = func.__qualname__

    except Exception:
        # If anything fails, just mark as unknown
        context = {"method_type": "unknown"}

    return context


def _serialize_output(output):
    """
    Helper function to serialize function output for logging.

    Args:
        output: The function return value to serialize

    Returns:
        Serializable representation of the output
    """
    try:
        if output is None:
            return None
        elif isinstance(output, (str, int, float, bool)):
            return output
        elif isinstance(output, (list, dict, tuple)):
            # Truncate large outputs to prevent log bloat
            output_str = str(output)
            return (
                output_str[:1000] + "..."
                if len(output_str) > 1000
                else output_str
            )
        else:
            return str(type(output))
    except Exception:
        return "<non-serializable-output>"


def log_execution(
    swarm_id: Optional[str] = None,
    status: Optional[str] = None,
    swarm_config: Optional[dict] = None,
    swarm_architecture: Optional[str] = None,
    enabled_on: Optional[bool] = False,
):
    """
    Log execution data for a swarm router instance.

    This function logs telemetry data about swarm router executions, including
    the swarm ID, execution status, and configuration details. It silently
    handles any logging errors to prevent execution interruption.

    Args:
        swarm_id (str): Unique identifier for the swarm router instance
        status (str): Current status of the execution (e.g., "start", "completion", "error")
        swarm_config (dict): Configuration dictionary containing swarm router settings
        swarm_architecture (str): Name of the swarm architecture used
    Returns:
        None

    Example:
        >>> log_execution(
        ...     swarm_id="swarm-router-abc123",
        ...     status="start",
        ...     swarm_config={"name": "my-swarm", "swarm_type": "SequentialWorkflow"}
        ... )
    """
    try:
        if enabled_on is None:
            log_agent_data(
                data_dict={
                    "swarm_router_id": swarm_id,
                    "status": status,
                    "swarm_router_config": swarm_config,
                    "swarm_architecture": swarm_architecture,
                }
            )
        else:
            pass
    except Exception:
        pass
