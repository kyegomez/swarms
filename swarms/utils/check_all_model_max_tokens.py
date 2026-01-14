from litellm import model_list, get_max_tokens

# Add model overrides here
MODEL_MAX_TOKEN_OVERRIDES = {
    "llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf": 4096,  # Example override
}


def check_all_model_max_tokens(
    as_list: bool = False, print_on: bool = True
):
    """
    Check and display the maximum token limits for all available models.

    Args:
        as_list (bool): If True, returns a list of model/token dicts.
                        If False, returns a formatted string.

    Returns:
        list or str: List of results or formatted string, depending on as_list.

    Note:
        Models not mapped in litellm and with no override set will be
        marked with a [WARNING] in the output.
    """
    results = []
    for model in model_list:
        model_info = {}
        try:
            max_tokens = get_max_tokens(model)
        except Exception:
            max_tokens = MODEL_MAX_TOKEN_OVERRIDES.get(
                model, "[NOT MAPPED]"
            )
            if max_tokens == "[NOT MAPPED]":
                model_info["warning"] = (
                    f"[WARNING] {model}: not mapped in litellm and no override set."
                )
        model_info["model"] = model
        model_info["max_tokens"] = max_tokens
        results.append(model_info)

    if as_list:
        return results
    else:
        text = ""
        for model_info in results:
            if "warning" in model_info:
                text += f"{model_info['warning']}\n"
            text += (
                f"{model_info['model']}: {model_info['max_tokens']}\n"
            )
            text += "─" * 80 + "\n"
        if print_on:
            print(text)
        return text


def get_single_model_max_tokens(model_name: str) -> int:
    """
    Get the maximum token limit for a single model.
    """
    try:
        return get_max_tokens(model_name)
    except Exception:
        raise ValueError(f"Model {model_name} not found in litellm")


if __name__ == "__main__":
    print(check_all_model_max_tokens(as_list=True))
