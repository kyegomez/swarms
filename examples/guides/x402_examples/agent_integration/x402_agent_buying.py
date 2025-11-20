from eth_account import Account
from x402.clients.httpx import x402HttpxClient


import os
from dotenv import load_dotenv

load_dotenv()


async def buy_x402_service(
    base_url: str = None, endpoint: str = None
):
    """
    Purchase a service from the X402 bazaar using the provided affordable_service details.

    This function sets up an X402 client with the user's private key, connects to the service provider,
    and executes a GET request to the service's endpoint as part of the buying process.

    Args:
        affordable_service (dict): A dictionary containing information about the target service.
        base_url (str, optional): The base URL of the service provider. Defaults to None.
        endpoint (str, optional): The specific API endpoint to interact with. Defaults to None.

    Returns:
        response (httpx.Response): The response object returned by the GET request to the service endpoint.

    Example:
        ```python
        affordable_service = {"id": "service123", "price": 90000}
        response = await buy_x402_service(
            affordable_service,
            base_url="https://api.cdp.coinbase.com",
            endpoint="/x402/v1/bazaar/services/service123"
        )
        print(await response.aread())
        ```
    """
    key = os.getenv("X402_PRIVATE_KEY")

    # Set up your payment account from private key
    account = Account.from_key(key)

    async with x402HttpxClient(
        account=account, base_url=base_url
    ) as client:
        response = await client.get(endpoint)
        print(await response.aread())

    return response
