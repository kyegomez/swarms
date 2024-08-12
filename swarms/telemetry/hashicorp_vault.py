import requests
from loguru import logger
import os


def fetch_secrets_from_vault(
    client_id: str = os.getenv("HCP_CLIENT_ID"),
    client_secret: str = os.getenv("HCP_CLIENT_SECRET"),
    organization_id: str = os.getenv("HCP_ORGANIZATION_ID"),
    project_id: str = os.getenv("HCP_PROJECT_ID"),
    app_id: str = os.getenv("HCP_APP_ID"),
) -> str:
    """
    Fetch secrets from HashiCorp Vault using service principal authentication.

    Args:
        client_id (str): The client ID for the service principal.
        client_secret (str): The client secret for the service principal.
        organization_id (str): The ID of the organization in HCP.
        project_id (str): The ID of the project in HCP.
        app_id (str): The ID of the app in HCP.

    Returns:
        str: A dictionary containing the fetched secrets.

    Raises:
        Exception: If there is an error retrieving the API token or secrets.
    """
    # Step 1: Generate the API Token
    token_url = "https://auth.idp.hashicorp.com/oauth2/token"
    token_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
        "audience": "https://api.hashicorp.cloud",
    }
    token_headers = {"Content-Type": "application/x-www-form-urlencoded"}

    logger.info("Requesting API token from HashiCorp Vault")
    response = requests.post(
        token_url, data=token_data, headers=token_headers
    )

    if response.status_code != 200:
        logger.error(
            f"Failed to retrieve API token. Status Code: {response.status_code}, Response: {response.text}"
        )
        response.raise_for_status()

    api_token = response.json().get("access_token")

    if not api_token:
        raise Exception("Failed to retrieve API token")

    # Step 2: Fetch Secrets
    secrets_url = f"https://api.cloud.hashicorp.com/secrets/2023-06-13/organizations/{organization_id}/projects/{project_id}/apps/{app_id}/open"
    secrets_headers = {"Authorization": f"Bearer {api_token}"}

    logger.info("Fetching secrets from HashiCorp Vault")
    response = requests.get(secrets_url, headers=secrets_headers)

    if response.status_code != 200:
        logger.error(
            f"Failed to fetch secrets. Status Code: {response.status_code}, Response: {response.text}"
        )
        response.raise_for_status()

    secrets = response.json()

    for secret in secrets["secrets"]:
        name = secret.get("name")
        value = secret.get("version", {}).get("value")
        print(f"Name: {name}, Value: {value}")

    return name, value


# def main() -> None:
#     """
#     Main function to fetch secrets from HashiCorp Vault and print them.

#     Raises:
#         EnvironmentError: If required environment variables are not set.
#     """
#     HCP_CLIENT_ID = os.getenv("HCP_CLIENT_ID")
#     HCP_CLIENT_SECRET = os.getenv("HCP_CLIENT_SECRET")
#     ORGANIZATION_ID = os.getenv("HCP_ORGANIZATION_ID")
#     PROJECT_ID = os.getenv("HCP_PROJECT_ID")
#     APP_ID = os.getenv("HCP_APP_ID")

#     # if not all([HCP_CLIENT_ID, HCP_CLIENT_SECRET, ORGANIZATION_ID, PROJECT_ID, APP_ID]):
#     #     raise EnvironmentError("One or more environment variables are missing: HCP_CLIENT_ID, HCP_CLIENT_SECRET, ORGANIZATION_ID, PROJECT_ID, APP_ID")

#     secrets = fetch_secrets_from_vault(
#         HCP_CLIENT_ID,
#         HCP_CLIENT_SECRET,
#         ORGANIZATION_ID,
#         PROJECT_ID,
#         APP_ID,
#     )
#     print(secrets)

# for secret in secrets["secrets"]:
#     name = secret.get("name")
#     value = secret.get("version", {}).get("value")
#     print(f"Name: {name}, Value: {value}")


# if __name__ == "__main__":
#     main()
