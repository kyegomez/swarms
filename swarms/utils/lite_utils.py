def litellm_check_for_tools(model_name: str):
    """Check if the model supports tools."""
    from litellm.utils import supports_function_calling

    return supports_function_calling(model_name)
