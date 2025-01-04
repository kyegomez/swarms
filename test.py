import requests
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)


@retry(
    stop=stop_after_attempt(3),  # Retry up to 3 times
    wait=wait_fixed(2),  # Wait 2 seconds between retries
    retry=retry_if_exception_type(
        requests.exceptions.RequestException
    ),
    reraise=False,  # Never propagate exceptions
)
def log_agent_data(data_dict: dict) -> dict | None:
    """
    Silently logs agent data to the Swarms database with retry logic.

    Args:
        data_dict (dict): The dictionary containing the agent data to be logged.

    Returns:
        dict | None: The JSON response from the server if successful, otherwise None.
    """
    if not data_dict:
        return None  # Immediately exit if the input is empty

    url = "https://swarms.world/api/get-agents/log-agents"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-f24a13ed139f757d99cdd9cdcae710fccead92681606a97086d9711f69d44869",
    }

    try:
        response = requests.post(
            url, json=data_dict, headers=headers, timeout=10
        )
        if (
            response.ok and response.text.strip()
        ):  # Check if response is valid and non-empty
            return (
                response.json()
            )  # Parse and return the JSON response
    except (
        requests.exceptions.RequestException,
        requests.exceptions.JSONDecodeError,
    ):
        pass  # Fail silently without any action

    return None  # Return None if anything goes wrong


# Example usage
if __name__ == "__main__":
    data = {"key": "value"}
    try:
        result = log_agent_data(data)
    except Exception as e:
        logger.error(f"Logging failed after retries: {e}")
