import json
import os
from loguru import logger
import aiohttp


async def send_telemetry(
    data: dict,
    swarms_api_key: str = None,
):
    """
    send_telemetry sends the data to the SWARMS API for logging.

    Args:
        data (dict): The data to be logged.
        swarms_api_key (str, optional): The SWARMS API key. Defaults to None.

    Returns:
        tuple: The response status and data from the API.


    Example:
        data = {
            "user_id": "123",
            "action": "login",
            "timestamp": "2022-01-01T00:00:00Z",
        }
        response_status, response_data = await send_telemetry(data)


    """
    url = "https://swarms.world/api/add-telemetry"

    if not swarms_api_key:
        swarms_api_key = get_swarms_api_key()

    session = aiohttp.ClientSession()

    headers = {"Content-Type": "application/json"}

    payload = {"data": data, "swarms_api_key": swarms_api_key}
    payload = json.dumps(payload)

    try:
        logger.debug(f"Sending data to {url} with payload: {payload}")
        async with session.post(
            url, json=payload, headers=headers
        ) as response:
            response_status = response.status
            response_data = await response.json()

            logger.info(
                f"Received response: {response_status} - {response_data}"
            )
            return response_status, response_data
    except Exception as e:
        logger.error(f"Error during request: {str(e)}")
        raise


def get_swarms_api_key():
    """Fetch the SWARMS_API_KEY environment variable or prompt the user for it."""
    swarms_api_key = os.getenv("SWARMS_API_KEY")
    return swarms_api_key
