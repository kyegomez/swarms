import secrets
import string
import re


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


def validate_api_key(api_key: str, prefix: str = "sk-") -> bool:
    """
    Validate if an API key matches the expected format.

    Args:
        api_key (str): The API key to validate
        prefix (str): The expected prefix (default: "sk-")

    Returns:
        bool: True if the API key is valid, False otherwise
    """
    if not isinstance(api_key, str):
        return False

    # Check if key starts with prefix
    if not api_key.startswith(prefix):
        return False

    # Check if the rest of the key contains only alphanumeric characters
    random_part = api_key[len(prefix) :]
    if not re.match(r"^[a-zA-Z0-9]+$", random_part):
        return False

    return True
