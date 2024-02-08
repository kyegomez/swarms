import inspect


def get_cls_init_params(cls) -> str:
    """
    Get the initialization parameters of a class.

    Args:
        cls: The class to retrieve the initialization parameters from.

    Returns:
        str: A string representation of the initialization parameters.

    """
    init_signature = inspect.signature(cls.__init__)
    params = init_signature.parameters
    params_str_list = []

    for name, param in params.items():
        if name == "self":
            continue
        if name == "kwargs":
            value = "Any keyword arguments"
        elif hasattr(cls, name):
            value = getattr(cls, name)
        else:
            value = cls.__dict__.get(name, "Unknown")

        params_str_list.append(
            f"    {name.capitalize().replace('_', ' ')}: {value}"
        )

    return "\n".join(params_str_list)
