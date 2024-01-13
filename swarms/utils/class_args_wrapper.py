import inspect


def print_class_parameters(cls, api_format: bool = False):
    """
    Print the parameters of a class constructor.

    Parameters:
    cls (type): The class to inspect.

    Example:
    >>> print_class_parameters(Agent)
    Parameter: x, Type: <class 'int'>
    Parameter: y, Type: <class 'int'>
    """
    try:
        # Get the parameters of the class constructor
        sig = inspect.signature(cls.__init__)
        params = sig.parameters

        if api_format:
            param_dict = {}
            for name, param in params.items():
                if name == "self":
                    continue
                param_dict[name] = str(param.annotation)
            return param_dict

        # Print the parameters
        for name, param in params.items():
            if name == "self":
                continue
            print(f"Parameter: {name}, Type: {param.annotation}")

    except Exception as e:
        print(f"An error occurred while inspecting the class: {e}")
