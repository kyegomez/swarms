from litellm import model_list, get_max_tokens
from swarms.utils.formatter import formatter

# Add model overrides here
MODEL_MAX_TOKEN_OVERRIDES = {
    "llama-2-70b-chat:2796ee9483c3fd7aa2e171d38f4ca12251a30609463dcfd4cd76703f22e96cdf": 4096,  # Example override
}


def check_all_model_max_tokens():
    """
    Check and display the maximum token limits for all available models.

    This function iterates through all models in the litellm model list and attempts
    to retrieve their maximum token limits. For models that are not properly mapped
    in litellm, it checks for custom overrides in MODEL_MAX_TOKEN_OVERRIDES.

    Returns:
        None: Prints the results to console using formatter.print_panel()

    Note:
        Models that are not mapped in litellm and have no override set will be
        marked with a [WARNING] in the output.
    """
    text = ""
    for model in model_list:
        # skip model names
        try:
            max_tokens = get_max_tokens(model)
        except Exception:
            max_tokens = MODEL_MAX_TOKEN_OVERRIDES.get(
                model, "[NOT MAPPED]"
            )
            if max_tokens == "[NOT MAPPED]":
                text += f"[WARNING] {model}: not mapped in litellm and no override set.\n"
        text += f"{model}: {max_tokens}\n"
        text += "─" * 80 + "\n"  # Add borderline for each model
    formatter.print_panel(text, "All Model Max Tokens")
    return text


# if __name__ == "__main__":
#     print(check_all_model_max_tokens())
