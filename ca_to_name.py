import base64
import requests
from struct import unpack


class SolanaTokenInfoFetcher:
    def __init__(self, rpc_url="https://api.mainnet-beta.solana.com"):
        """
        Initializes the token info fetcher.

        Args:
            rpc_url (str): The Solana RPC URL.
        """
        self.rpc_url = rpc_url

    def make_rpc_request(self, method, params):
        """
        Makes an RPC request to the Solana JSON RPC API.

        Args:
            method (str): The RPC method to call.
            params (list): The parameters for the method.

        Returns:
            dict: The JSON response from the RPC call.
        """
        headers = {"Content-Type": "application/json"}
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        try:
            response = requests.post(self.rpc_url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making RPC request: {e}")
            return None

    def get_token_info(self, mint_address):
        """
        Fetches token metadata (name, symbol) from the token mint account.

        Args:
            mint_address (str): The token mint address.

        Returns:
            dict: A dictionary containing the token name and symbol.
        """
        params = [
            mint_address,
            {"encoding": "base64"}
        ]
        response = self.make_rpc_request("getAccountInfo", params)

        if not response or "result" not in response or not response["result"]["value"]:
            print(f"Failed to fetch data for mint address: {mint_address}")
            return None

        # Decode the account data
        account_data = response["result"]["value"]["data"][0]
        decoded_data = base64.b64decode(account_data)

        # Extract token name and symbol
        try:
            token_name = decoded_data[0:32].decode("utf-8").rstrip("\x00")
            token_symbol = decoded_data[32:44].decode("utf-8").rstrip("\x00")
            decimals = unpack("<B", decoded_data[44:45])[0]
            return {
                "name": token_name,
                "symbol": token_symbol,
                "decimals": decimals
            }
        except Exception as e:
            print(f"Error decoding token metadata: {e}")
            return None


if __name__ == "__main__":
    # Replace with the token mint address you want to query
    mint_address = "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU"  # Example: USDC -> 4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU

    # Initialize the token info fetcher
    fetcher = SolanaTokenInfoFetcher()

    # Fetch and display token information
    token_info = fetcher.get_token_info(mint_address)

    if token_info:
        print(f"Token Name: {token_info['name']}")
        print(f"Token Symbol: {token_info['symbol']}")
        print(f"Decimals: {token_info['decimals']}")
    else:
        print("Failed to fetch token information.")
