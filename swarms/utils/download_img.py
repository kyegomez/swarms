from io import BytesIO

import requests
from PIL import Image


def download_img_from_url(url: str):
    """
    Downloads an image from the given URL and saves it locally.

    Args:
        url (str): The URL of the image to download.

    Raises:
        ValueError: If the URL is empty or invalid.
        IOError: If there is an error while downloading or saving the image.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    try:
        response = requests.get(url)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content))
        image.save("downloaded_image.jpg")

        print("Image downloaded successfully.")
    except requests.exceptions.RequestException as e:
        raise OSError("Error while downloading the image.") from e
    except OSError as e:
        raise OSError("Error while saving the image.") from e
