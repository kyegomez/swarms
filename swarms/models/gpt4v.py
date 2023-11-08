import base64
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Union

import requests
from dotenv import load_dotenv
from openai import OpenAI
from termcolor import colored

# ENV
load_dotenv()


def logging_config():
    """Configures logging"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    return logger


@dataclass
class GPT4VisionResponse:
    """A response structure for GPT-4"""

    answer: str


@dataclass
class GPT4Vision:
    """
    GPT4Vision model class

    Attributes:
    -----------
    max_retries: int
        The maximum number of retries to make to the API
    backoff_factor: float
        The backoff factor to use for exponential backoff
    timeout_seconds: int
        The timeout in seconds for the API request
    api_key: str
        The API key to use for the API request
    quality: str
        The quality of the image to generate
    max_tokens: int
        The maximum number of tokens to use for the API request

    Methods:
    --------
    process_img(self, img_path: str) -> str:
        Processes the image to be used for the API request
    __call__(self, img: Union[str, List[str]], tasks: List[str]) -> GPT4VisionResponse:
        Makes a call to the GPT-4 Vision API and returns the image url

    Example:
    >>> gpt4vision = GPT4Vision()
    >>> img = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
    >>> tasks = ["A painting of a dog"]
    >>> answer = gpt4vision(img, tasks)
    >>> print(answer)


    """

    max_retries: int = 3
    model: str = "gpt-4-vision-preview"
    backoff_factor: float = 2.0
    timeout_seconds: int = 10
    api_key: Optional[str] = None
    # 'Low' or 'High' for respesctively fast or high quality, but high more token usage
    quality: str = "low"
    # Max tokens to use for the API request, the maximum might be 3,000 but we don't know
    max_tokens: int = 200
    client = OpenAI(
        api_key=api_key,
        max_retries=max_retries,
    )
    logger = logging_config()

    class Config:
        """Config class for the GPT4Vision model"""

        arbitary_types_allowed = True

    def process_img(self, img: str) -> str:
        """Processes the image to be used for the API request"""
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def __call__(
        self,
        img: Union[str, List[str]],
        tasks: List[str],
    ) -> GPT4VisionResponse:
        """
        Calls the GPT-4 Vision API and returns the image url

        Parameters:
        -----------
        img: Union[str, List[str]]
            The image to be used for the API request
        tasks: List[str]
            The tasks to be used for the API request

        Returns:
        --------
        answer: GPT4VisionResponse
            The response from the API request

        Example:
        --------
        >>> gpt4vision = GPT4Vision()
        >>> img = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
        >>> tasks = ["A painting of a dog"]
        >>> answer = gpt4vision(img, tasks)
        >>> print(answer)


        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Image content
        image_content = [{
            "type": "imavge_url",
            "image_url": img
        } if img.startswith("http") else {
            "type": "image",
            "data": img
        } for img in img]

        messages = [{
            "role":
                "user",
            "content":
                image_content + [{
                    "type": "text",
                    "text": q
                } for q in tasks],
        }]

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "detail": self.quality,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                answer = response.json(
                )["choices"][0]["message"]["content"]["text"]
                return GPT4VisionResponse(answer=answer)
            except requests.exceptions.HTTPError as error:
                self.logger.error(
                    f"HTTP error: {error.response.status_code}, {error.response.text}"
                )
                if error.response.status_code in [429, 500, 503]:
                    # Exponential backoff = 429(too many requesys)
                    # And 503 = (Service unavailable) errors
                    time.sleep(self.backoff_factor**attempt)
                else:
                    break

            except requests.exceptions.RequestException as error:
                self.logger.error(f"Request error: {error}")
                time.sleep(self.backoff_factor**attempt)
            except Exception as error:
                self.logger.error(
                    f"Unexpected Error: {error} try optimizing your api key and try"
                    " again")
                raise error from None

        raise TimeoutError("API Request timed out after multiple retries")

    def run(self, task: str, img: str) -> str:
        """
        Runs the GPT-4 Vision API

        Parameters:
        -----------
        task: str
            The task to be used for the API request
        img: str
            The image to be used for the API request

        Returns:
        --------
        out: str
            The response from the API request

        Example:
        --------
        >>> gpt4vision = GPT4Vision()
        >>> task = "A painting of a dog"
        >>> img = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
        >>> answer = gpt4vision.run(task, img)
        >>> print(answer)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role":
                        "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task}"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"{img}",
                        },
                    ],
                }],
                max_tokens=self.max_tokens,
            )

            out = response.choices[0].text
            return out
        except Exception as error:
            print(
                colored(
                    (f"Error when calling GPT4Vision, Error: {error} Try optimizing"
                     " your key, and try again"),
                    "red",
                ))

    async def arun(self, task: str, img: str) -> str:
        """
        Asynchronous run method for GPT-4 Vision

        Parameters:
        -----------
        task: str
            The task to be used for the API request
        img: str
            The image to be used for the API request

        Returns:
        --------
        out: str
            The response from the API request

        Example:
        --------
        >>> gpt4vision = GPT4Vision()
        >>> task = "A painting of a dog"
        >>> img = "https://cdn.openai.com/dall-e/encoded/feats/feats_01J9J5ZKJZJY9.png"
        >>> answer = await gpt4vision.arun(task, img)
        >>> print(answer)
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role":
                        "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{task}"
                        },
                        {
                            "type": "image_url",
                            "image_url": f"{img}",
                        },
                    ],
                }],
                max_tokens=self.max_tokens,
            )
            out = response.choices[0].text
            return out
        except Exception as error:
            print(
                colored(
                    (f"Error when calling GPT4Vision, Error: {error} Try optimizing"
                     " your key, and try again"),
                    "red",
                ))
