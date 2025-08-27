import os


def check_swarms_api_key():
    """
    Check if the Swarms API key is set.

    Returns:
        str: The value of the SWARMS_API_KEY environment variable.

    Raises:
        ValueError: If the SWARMS_API_KEY environment variable is not set.
    """
    if os.getenv("SWARMS_API_KEY") is None:
        raise ValueError(
            "Swarms API key is not set. Please set the SWARMS_API_KEY environment variable. "
            "You can get your key here: https://swarms.world/platform/api-keys"
        )
    return os.getenv("SWARMS_API_KEY")
