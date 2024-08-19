import hashlib
import json
import os
from typing import Any, Dict, Optional, List


class PromptCache:
    """
    A framework to handle prompt caching for any LLM API. This reduces costs, latency,
    and allows reuse of long-form context across multiple API requests.
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        llm_api_function: Optional[Any] = None,
        text: Optional[List[str]] = None,
    ):
        """
        Initializes the PromptCache instance.

        Args:
            cache_dir (str): Directory where cached responses are stored.
            llm_api_function (Optional[Any]): The function that interacts with the LLM API.
                                              It should accept a prompt and return the response.
        """
        self.cache_dir = cache_dir
        self.llm_api_function = llm_api_function
        self.text = text

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _generate_cache_key(self, prompt: str) -> str:
        """
        Generates a unique cache key for a given prompt.

        Args:
            prompt (str): The prompt to generate a cache key for.

        Returns:
            str: A unique cache key.
        """
        return hashlib.md5(prompt.encode("utf-8")).hexdigest()

    def _cache_file_path(self, cache_key: str) -> str:
        """
        Constructs the file path for the cache file.

        Args:
            cache_key (str): The cache key for the prompt.

        Returns:
            str: The path to the cache file.
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Loads a cached response if available.

        Args:
            cache_key (str): The cache key for the prompt.

        Returns:
            Optional[Dict[str, Any]]: The cached response, or None if not found.
        """
        cache_file = self._cache_file_path(cache_key)
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)
        return None

    def _save_to_cache(
        self, cache_key: str, response: Dict[str, Any]
    ) -> None:
        """
        Saves the API response to the cache.

        Args:
            cache_key (str): The cache key for the prompt.
            response (Dict[str, Any]): The API response to be cached.
        """
        cache_file = self._cache_file_path(cache_key)
        with open(cache_file, "w") as f:
            json.dump(response, f)

    def get_response(self, prompt: str) -> Dict[str, Any]:
        """
        Retrieves the response for a prompt, using cache if available.

        Args:
            prompt (str): The prompt to retrieve the response for.

        Returns:
            Dict[str, Any]: The API response, either from cache or freshly fetched.
        """
        cache_key = self._generate_cache_key(prompt)
        cached_response = self._load_from_cache(cache_key)

        if cached_response is not None:
            return cached_response

        # If the response is not cached, use the LLM API to get the response
        if self.llm_api_function is None:
            raise ValueError("LLM API function is not defined.")

        response = self.llm_api_function(prompt)
        self._save_to_cache(cache_key, response)

        return response

    def clear_cache(self) -> None:
        """
        Clears the entire cache directory.
        """
        for cache_file in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, cache_file))


# Example usage
if __name__ == "__main__":
    # Dummy LLM API function
    def mock_llm_api(prompt: str) -> Dict[str, Any]:
        return {"response": f"Mock response to '{prompt}'"}

    # Initialize the cache
    cache = PromptCache(llm_api_function=mock_llm_api)

    # Example prompts
    prompt1 = "What is the capital of France?"
    prompt2 = "Explain the theory of relativity."

    # Get responses
    print(cache.get_response(prompt1))
    print(cache.get_response(prompt2))
