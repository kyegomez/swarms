import requests
import os

class Anthropic:
    """Anthropic large language models."""

    def __init__(
            self, 
            model="claude-2", 
            max_tokens_to_sample=256, 
            temperature=None, 
            top_k=None, 
            top_p=None, 
            streaming=False, 
            default_request_timeout=None
        ):
        self.model = model
        self.max_tokens_to_sample = max_tokens_to_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.streaming = streaming
        self.default_request_timeout = default_request_timeout or 600
        self.anthropic_api_url = os.getenv("ANTHROPIC_API_URL", "https://api.anthropic.com")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    def _default_params(self):
        """Get the default parameters for calling Anthropic API."""
        d = {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return d

    def generate(self, prompt, stop=None):
        """Call out to Anthropic's completion endpoint."""
        stop = stop or []
        params = self._default_params()
        headers = {"Authorization": f"Bearer {self.anthropic_api_key}"}
        data = {
            "prompt": prompt,
            "stop_sequences": stop,
            **params
        }
        response = requests.post(f"{self.anthropic_api_url}/completions", headers=headers, json=data, timeout=self.default_request_timeout)
        return response.json().get("completion")
    
    def __call__(self, prompt, stop=None):
        """Call out to Anthropic's completion endpoint."""
        stop = stop or []
        params = self._default_params()
        headers = {"Authorization": f"Bearer {self.anthropic_api_key}"}
        data = {
            "prompt": prompt,
            "stop_sequences": stop,
            **params
        }
        response = requests.post(f"{self.anthropic_api_url}/completions", headers=headers, json=data, timeout=self.default_request_timeout)
        return response.json().get("completion")