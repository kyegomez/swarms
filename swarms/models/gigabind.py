import requests
from tenacity import retry, stop_after_attempt, wait_fixed



class Gigabind:
    """Gigabind API.

    Args:
        host (str, optional): host. Defaults to None.
        proxy_url (str, optional): proxy_url. Defaults to None.
        port (int, optional): port. Defaults to 8000.
        endpoint (str, optional): endpoint. Defaults to "embeddings".

    Examples:
        >>> from swarms.models.gigabind import Gigabind
        >>> api = Gigabind(host="localhost", port=8000, endpoint="embeddings")
        >>> response = api.run(text="Hello, world!", vision="image.jpg")
        >>> print(response)
    """

    def __init__(
        self,
        host: str = None,
        proxy_url: str = None,
        port: int = 8000,
        endpoint: str = "embeddings",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.host = host
        self.proxy_url = proxy_url
        self.port = port
        self.endpoint = endpoint

        # Set the URL to the API
        if self.proxy_url is not None:
            self.url = f"{self.proxy_url}"
        else:
            self.url = f"http://{host}:{port}/{endpoint}"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def run(
        self,
        text: str = None,
        vision: str = None,
        audio: str = None,
        *args,
        **kwargs,
    ):
        """Run the Gigabind API.

        Args:
            text (str, optional): text. Defaults to None.
            vision (str, optional): images. Defaults to None.
            audio (str, optional): audio file paths. Defaults to None.

        Raises:
            ValueError: At least one of text, vision or audio must be provided

        Returns:
            embeddings: embeddings
        """
        try:
            # Prepare the data to send to the API
            data = {}
            if text is not None:
                data["text"] = text
            if vision is not None:
                data["vision"] = vision
            if audio is not None:
                data["audio"] = audio
            else:
                raise ValueError(
                    "At least one of text, vision or audio must be"
                    " provided"
                )

            # Send a POST request to the API and return the response
            response = requests.post(
                self.url, json=data, *args, **kwargs
            )
            return response.json()
        except Exception as error:
            print(f"Gigabind API error: {error}")
            return None

    def generate_summary(self, text: str = None, *args, **kwargs):
        # Prepare the data to send to the API
        data = {}
        if text is not None:
            data["text"] = text
        else:
            raise ValueError(
                "At least one of text, vision or audio must be"
                " provided"
            )

        # Send a POST request to the API and return the response
        response = requests.post(self.url, json=data, *args, **kwargs)
        return response.json()


# api = Gigabind(host="localhost", port=8000, endpoint="embeddings")
# response = api.run(text="Hello, world!", vision="image.jpg")
# print(response)
