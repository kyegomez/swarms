from litellm import encode


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens in the given text."""
    return len(encode(model=model, text=text))
