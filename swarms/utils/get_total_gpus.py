try:
    import ray
except ImportError:
    print(
        "Please install Ray. You can install it by running 'pip install ray'"
    )
    raise


def track_gpu_resources():
    """
    Retrieves and prints information about the total and available GPUs.

    Returns:
        tuple: A tuple containing the total number of GPUs and the number of available GPUs.
    """
    if not ray.is_initialized():
        ray.init()

    resources = ray.cluster_resources()
    available_resources = ray.available_resources()

    # Total GPUs
    total_gpus = resources.get("GPU", 0)
    print(f"Total GPUs: {total_gpus}")

    available_resources = available_resources.get("GPU", 0)

    print(f"Available GPUs: {available_resources}")
    print(f"Used GPUs: {total_gpus - available_resources}")

    return total_gpus, available_resources


def track_all_resources():
    """
    Prints detailed information about all resources in the Ray cluster.

    This function initializes Ray if it is not already initialized, and then retrieves
    information about the total resources and available resources in the cluster.
    It prints the resource name and quantity for both total resources and available resources.

    Note: This function requires the Ray library to be installed.

    Example usage:
        track_all_resources()
    """
    if not ray.is_initialized():
        ray.init()

    resources = ray.cluster_resources()
    available_resources = ray.available_resources()

    print("Total Resources:")
    for resource, quantity in resources.items():
        print(f"  {resource}: {quantity}")

    print("\nAvailable Resources:")
    for resource, quantity in available_resources.items():
        print(f"  {resource}: {quantity}")


def execute__callableon_gpu(
    num_cpus: int = None,
    num_gpus: int = None,
    pre_post_process=None,
    execute_before: bool = True,
    *args,
    **kwargs,
):
    """
    A decorator to execute functions with specified Ray resources, with optional pre/post processing.

    Args:
        num_cpus (int, optional): The number of CPUs to allocate for the function execution. Defaults to None.
        num_gpus (int, optional): The number of GPUs to allocate for the function execution. Defaults to None.
        pre_post_process (callable, optional): A callable function to be executed before or after the main function. Defaults to None.
        execute_before (bool, optional): Determines whether the pre_post_process should be executed before or after the main function. Defaults to True.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        The result of the main function execution.

    Example:
    >>> @execute__callableon_gpu(num_gpus=1)
    ... def add(a, b):
    ...     return a + b
    >>> add(1, 2)
    3

    """

    def decorator(func):
        # Initialize Ray, if not already done.
        if not ray.is_initialized():
            ray.init()

        # Register the function as a Ray remote function with specified resources.
        remote_func = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus)(
            func
        )

        # Optionally, register the callable if provided.
        if pre_post_process:
            remote_callable = ray.remote(
                num_cpus=num_cpus, num_gpus=num_gpus
            )(pre_post_process)

        def wrapper(*args, **kwargs):
            # Execute the callable before or after the main function, based on 'execute_before'
            if pre_post_process and execute_before:
                # Execute the callable and wait for its result before the main function
                callable_result = remote_callable.remote(*args, **kwargs)
                ray.get(callable_result)

            # Execute the main function
            result_ref = remote_func.remote(*args, **kwargs)
            result = ray.get(result_ref)

            if pre_post_process and not execute_before:
                # Execute the callable and wait for its result after the main function
                callable_result = remote_callable.remote(*args, **kwargs)
                ray.get(callable_result)

            return result

        return wrapper

    return decorator
