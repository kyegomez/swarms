import os

import openai
import requests
from dotenv import load_dotenv

from swarms.models.base_llm import AbstractLLM

# Load .env file
load_dotenv()

# OpenAI API Key env
def openai_api_key_env():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    return openai_api_key
    

class OpenAITTS(AbstractLLM):
    """OpenAI TTS model

    Attributes:
        model_name (str): _description_
        proxy_url (str): _description_
        openai_api_key (str): _description_
        voice (str): _description_
        chunk_size (_type_): _description_
        
    Methods:
        run: _description_
        
        
    Examples:
    >>> from swarms.models.openai_tts import OpenAITTS
    >>> tts = OpenAITTS(
    ...     model_name = "tts-1-1106",
    ...     proxy_url = "https://api.openai.com/v1/audio/speech",
    ...     openai_api_key = openai_api_key_env,
    ...     voice = "onyx",
    ... )
    >>> tts.run("Hello world")
    
    """
    def __init__(
        self,
        model_name: str = "tts-1-1106",
        proxy_url: str = "https://api.openai.com/v1/audio/speech",
        openai_api_key: str = openai_api_key_env,
        voice: str = "onyx",
        chunk_size = 1024 * 1024,
    ):
        super().__init__()
        self.model_name = model_name
        self.proxy_url = proxy_url
        self.openai_api_key = openai_api_key
        self.voice = voice
        self.chunk_size = chunk_size
        
    def run(
        self,
        task: str,
        *args,
        **kwargs
    ):
        """Run the tts model

        Args:
            task (str): _description_

        Returns:
            _type_: _description_
        """
        response = requests.post(
            self.proxy_url,
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
            },
            json={
                "model": self.model_name,
                "input": task,
                "voice": self.voice,
            },
        )
        
        audio = b""
        for chunk in response.iter_content(chunk_size = 1024 * 1024):
            audio += chunk
        return audio