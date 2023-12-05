import tiktoken


def limit_tokens_from_string(
    string: str, model: str = "gpt-4", limit: int = 500
) -> str:
    """Limits the number of tokens in a string

    Args:
        string (str): _description_
        model (str): _description_
        limit (int): _description_

    Returns:
        str: _description_
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        encoding = tiktoken.encoding_for_model(
            "gpt2"
        )  # Fallback for others.

    encoded = encoding.encode(string)

    out = encoding.decode(encoded[:limit])
    return out
