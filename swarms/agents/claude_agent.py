import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from swarms.structs.agent import Agent


def check_anthropic_package():
    """Check if the Anthropic/Claude package is installed, and install it if not."""
    try:
        import anthropic  # type: ignore

        return anthropic
    except Exception:
        logger.info("Anthropic package not found. Attempting to install...")

        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "anthropic"])
            import anthropic  # type: ignore

            return anthropic
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install Anthropic package: {e}")
            raise RuntimeError("Anthropic package installation failed.") from e


class ClaudeCodeAssistant(Agent):
    """
    Claude Code (Anthropic) assistant wrapper for the swarms framework.

    This class provides a minimal, robust integration point for Claude Code /
    Anthropic so that it can be used as an external agent inside Swarms.

    Notes:
    - It lazily installs/imports the `anthropic` package when first used.
    - The concrete SDK methods vary over time; this wrapper attempts to call
      commonly-used client methods (`responses.create`, `completions.create`,
      or `complete`) when available.
    """

    def __init__(
        self,
        name: str,
        description: str = "Claude Code assistant wrapper",
        instructions: Optional[str] = None,
        model: str = "claude-2",
        tools: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.metadata = metadata or {}

        super().__init__(*args, **kwargs)

        # Attempt to import the Anthropic package and initialize a client
        anthropic = check_anthropic_package()

        # Support a few common client entrypoints (`Anthropic`, `Client`)
        client_cls = getattr(anthropic, "Anthropic", None) or getattr(anthropic, "Client", None)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set; ClaudeCodeAssistant will not be able to make API calls.")
            self.client = None
        else:
            try:
                if client_cls:
                    # Some SDKs accept api_key in constructor, others via env var
                    try:
                        self.client = client_cls(api_key=api_key)
                    except TypeError:
                        self.client = client_cls()
                else:
                    # Fallback: expose the imported module as client and hope for the best
                    self.client = anthropic
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {e}")
                self.client = None

    def _build_prompt(self, task: str) -> str:
        if self.instructions:
            return f"{self.instructions}\n\nUser: {task}\nAssistant:"
        return task

    def run(self, task: str, *args, **kwargs) -> str:
        """Run a task using the Claude Code / Anthropic client.

        This method tries several common SDK call patterns. If none are
        available it raises a helpful error explaining how to configure the
        environment.
        """
        if not self.client:
            raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY or install the anthropic package.")

        prompt = self._build_prompt(task)

        # Try `responses.create` (newer SDKs)
        try:
            if hasattr(self.client, "responses"):
                # Example: client.responses.create(model=..., input=...)
                resp = self.client.responses.create(model=self.model, input=prompt)
                text = getattr(resp, "output", None) or resp
                # Best effort to extract text
                if hasattr(text, "text"):
                    return text.text
                if isinstance(text, dict):
                    # Try common dict shapes
                    return text.get("text") or text.get("output") or str(text)
                return str(text)

            # Try `completions.create`
            if hasattr(self.client, "completions"):
                resp = self.client.completions.create(model=self.model, prompt=prompt)
                # Extract text in common shapes
                if hasattr(resp, "completion"):
                    return resp.completion
                if isinstance(resp, dict):
                    return resp.get("completion") or resp.get("text") or str(resp)

            # Try top-level `complete` function
            if hasattr(self.client, "complete"):
                resp = self.client.complete(model=self.model, prompt=prompt)
                if isinstance(resp, dict):
                    return resp.get("completion") or resp.get("text") or str(resp)
                return str(resp)

        except Exception as e:
            logger.error(f"Error calling Anthropic client: {e}")
            raise

        raise NotImplementedError(
            "Unable to call Anthropic/Claude client with detected SDK. Ensure the `anthropic` package is installed and ANTHROPIC_API_KEY is set."
        )

    def call(self, task: str, *args, **kwargs) -> str:
        return self.run(task, *args, **kwargs)

    def batch_run(self, tasks: List[str], *args, **kwargs) -> List[Any]:
        return [self.run(t, *args, **kwargs) for t in tasks]

    def run_concurrently(self, tasks: List[str], *args, **kwargs) -> List[Any]:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            return list(executor.map(self.run, tasks))
