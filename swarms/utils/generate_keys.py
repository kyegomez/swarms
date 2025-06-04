import secrets
import string


def generate_api_key(prefix: str = "sk-", length: int = 32) -> str:
    """
    Generate a secure API key with a custom prefix.

    Args:
        prefix (str): The prefix to add to the API key (default: "sk-")
        length (int): The length of the random part of the key (default: 32)

    Returns:
        str: A secure API key in the format: prefix + random_string

    Example:
        >>> generate_api_key("sk-")
        'sk-abc123...'
    """
    # Validate prefix
    if not isinstance(prefix, str):
        raise ValueError("Prefix must be a string")

    # Validate length
    if length < 8:
        raise ValueError("Length must be at least 8 characters")

    # Generate random string using alphanumeric characters
    alphabet = string.ascii_letters + string.digits
    random_part = "".join(
        secrets.choice(alphabet) for _ in range(length)
    )

    # Combine prefix and random part
    api_key = f"{prefix}{random_part}"

    return api_key
