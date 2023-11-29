import os
import asyncio
import base64
import concurrent.futures
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import openai
import requests
from cachetools import TTLCache
from dotenv import load_dotenv
from openai import OpenAI
from ratelimit import limits, sleep_and_retry
from termcolor import colored

# ENV
load_dotenv()


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
    run(self, img: Union[str, List[str]], tasks: List[str]) -> GPT4VisionResponse:
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
    openai_api_key: Optional[str] = None or os.getenv(
        "OPENAI_API_KEY"
    )
    # 'Low' or 'High' for respesctively fast or high quality, but high more token usage
    quality: str = "low"
    # Max tokens to use for the API request, the maximum might be 3,000 but we don't know
    max_tokens: int = 200
    client = OpenAI(
        api_key=openai_api_key,
    )
    dashboard: bool = True
    call_limit: int = 1
    period_seconds: int = 60

    # Cache for storing API Responses
    cache = TTLCache(maxsize=100, ttl=600)  # Cache for 10 minutes

    class Config:
        """Config class for the GPT4Vision model"""

        arbitary_types_allowed = True

    def process_img(self, img: str) -> str:
        """Processes the image to be used for the API request"""
        with open(img, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @sleep_and_retry
    @limits(
        calls=call_limit, period=period_seconds
    )  # Rate limit of 10 calls per minute
    def run(self, task: str, img: str):
        """
        Run the GPT-4 Vision model

        Task: str
            The task to run
        Img: str
            The image to run the task on

        """
        if self.dashboard:
            self.print_dashboard()
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": str(img),
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            out = print(response.choices[0])
            # out = self.clean_output(out)
            return out
        except openai.OpenAIError as e:
            # logger.error(f"OpenAI API error: {e}")
            return (
                f"OpenAI API error: Could not process the image. {e}"
            )
        except Exception as e:
            return (
                "Unexpected error occurred while processing the"
                f" image. {e}"
            )

    def clean_output(self, output: str):
        # Regex pattern to find the Choice object representation in the output
        pattern = r"Choice\(.*?\(content=\"(.*?)\".*?\)\)"
        match = re.search(pattern, output, re.DOTALL)

        if match:
            # Extract the content from the matched pattern
            content = match.group(1)
            # Replace escaped quotes to get the clean content
            content = content.replace(r"\"", '"')
            print(content)
        else:
            print("No content found in the output.")

    async def arun(self, task: str, img: str):
        """
        Arun is an async version of run

        Task: str
            The task to run
        Img: str
            The image to run the task on

        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img,
                                },
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
            )

            return print(response.choices[0])
        except openai.OpenAIError as e:
            # logger.error(f"OpenAI API error: {e}")
            return (
                f"OpenAI API error: Could not process the image. {e}"
            )
        except Exception as e:
            return (
                "Unexpected error occurred while processing the"
                f" image. {e}"
            )

    def run_batch(
        self, tasks_images: List[Tuple[str, str]]
    ) -> List[str]:
        """Process a batch of tasks and images"""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.run, task, img)
                for task, img in tasks_images
            ]
            results = [future.result() for future in futures]
        return results

    async def run_batch_async(
        self, tasks_images: List[Tuple[str, str]]
    ) -> List[str]:
        """Process a batch of tasks and images asynchronously"""
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(None, self.run, task, img)
            for task, img in tasks_images
        ]
        return await asyncio.gather(*futures)

    async def run_batch_async_with_retries(
        self, tasks_images: List[Tuple[str, str]]
    ) -> List[str]:
        """Process a batch of tasks and images asynchronously with retries"""
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(
                None, self.run_with_retries, task, img
            )
            for task, img in tasks_images
        ]
        return await asyncio.gather(*futures)

    def print_dashboard(self):
        dashboard = print(
            colored(
                f"""
            GPT4Vision Dashboard
            -------------------
            Max Retries: {self.max_retries}
            Model: {self.model}
            Backoff Factor: {self.backoff_factor}
            Timeout Seconds: {self.timeout_seconds}
            Image Quality: {self.quality}
            Max Tokens: {self.max_tokens}

            """,
                "green",
            )
        )
        return dashboard

    def health_check(self):
        """Health check for the GPT4Vision model"""
        try:
            response = requests.get(
                "https://api.openai.com/v1/engines"
            )
            return response.status_code == 200
        except requests.RequestException as error:
            print(f"Health check failed: {error}")
            return False

    def sanitize_input(self, text: str) -> str:
        """
        Sanitize input to prevent injection attacks.

        Parameters:
        text: str - The input text to be sanitized.

        Returns:
        The sanitized text.
        """
        # Example of simple sanitization, this should be expanded based on the context and usage
        sanitized_text = re.sub(r"[^\w\s]", "", text)
        return sanitized_text
