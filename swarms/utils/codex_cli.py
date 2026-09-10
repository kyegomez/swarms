"""Optional, text-only backend using an authenticated Codex CLI."""

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil


def validate_codex_agent(agent: Any) -> None:
    """Reject agent options that require an unsupported Codex capability.

    Args:
        agent: Agent configuration to validate before setup or invocation.

    Raises:
        ValueError: An option requires tools, streaming, or API configuration.
    """
    unsupported = [
        name
        for name in (
            "tools",
            "tools_list_dictionary",
            "tool_schema",
            "list_base_models",
            "mcp_url",
            "mcp_urls",
            "mcp_config",
            "mcp_configs",
            "handoffs",
            "think_tool",
            "stream",
            "streaming_on",
            "streaming_callback",
            "multi_modal",
            "llm_args",
            "llm_base_url",
            "llm_api_key",
            "prompt_caching",
            "random_models_on",
            "react_on",
        )
        if getattr(agent, name, None)
    ]
    if agent.max_loops == "auto":
        unsupported.append('max_loops="auto"')
    if unsupported:
        raise ValueError(
            "The Codex backend supports bounded text-only runs; "
            "unsupported options: " + ", ".join(unsupported)
        )


class CodexCLI:
    """Generate text through the official CLI without handling login tokens.

    Args:
        model_name: Codex model identifier, or None for the CLI default.
        system_prompt: Instructions accompanying each conversation.
        timeout: Positive, finite maximum subprocess duration in seconds.

    Raises:
        ValueError: The model or timeout is invalid.
        FileNotFoundError: Codex is not installed on PATH.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        system_prompt: str = "",
        timeout: float = 180,
    ) -> None:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError(
                "codex_config.timeout must be positive and finite"
            )
        if model_name is not None and (
            not isinstance(model_name, str) or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string or None"
            )
        self.executable = shutil.which("codex")
        if self.executable is None:
            raise FileNotFoundError(
                "Codex CLI was not found on PATH. Install @openai/codex "
                "and run `codex login` before using llm_backend='codex'."
            )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.timeout = timeout

    def run(
        self,
        task: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Return the final response to text or a role-preserving transcript.

        Args:
            task: Text appended as a user turn when provided.
            messages: System, developer, user, or assistant text messages.
            system_prompt: Per-call override, used for memory compression.
            **kwargs: Non-empty unsupported options are rejected.

        Returns:
            The final non-empty Codex response.

        Raises:
            ValueError: Input is empty, malformed, or requires unsupported tools.
            TimeoutError: The CLI exceeds the configured deadline.
            RuntimeError: The CLI fails or produces no final message.
        """
        if any(kwargs.values()):
            raise ValueError(
                "Codex supports text only; unsupported run options"
            )
        transcript = list(messages or [])
        if task is not None:
            transcript.append({"role": "user", "content": task})
        if not transcript:
            raise ValueError(
                "Codex requires a non-empty task or messages"
            )
        for message in transcript:
            if (
                not isinstance(message, dict)
                or message.get("role")
                not in {"system", "developer", "user", "assistant"}
                or not isinstance(message.get("content"), str)
                or message.get("tool_calls")
                or message.get("function_call")
            ):
                raise ValueError(
                    "Codex accepts text messages without tool calls"
                )
        if not any(
            message["content"].strip() for message in transcript
        ):
            raise ValueError(
                "Codex requires non-empty message content"
            )
        prompt = json.dumps(
            {
                "system_prompt": (
                    self.system_prompt
                    if system_prompt is None
                    else system_prompt
                ),
                "messages": transcript,
            },
            ensure_ascii=False,
        )
        with tempfile.TemporaryDirectory(
            prefix="swarms-codex-"
        ) as directory:
            output = Path(directory) / "response.txt"
            command = [
                self.executable,
                "exec",
                "--ignore-user-config",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--color",
                "never",
                "--cd",
                directory,
                "--output-last-message",
                str(output),
                "-c",
                'web_search="disabled"',
                "-c",
                "features.shell_tool=false",
                "-c",
                "features.multi_agent=false",
                "-c",
                "features.apps=false",
                "-c",
                "features.plugins=false",
                "-c",
                "project_doc_max_bytes=0",
            ]
            if self.model_name is not None:
                command.extend(["--model", self.model_name])
            command.append("-")
            with subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                cwd=directory,
            ) as process:
                try:
                    _, stderr = process.communicate(
                        "Continue this conversation, following system_prompt "
                        "and the message roles. Return only the final answer.\n"
                        + prompt,
                        timeout=self.timeout,
                    )
                except (
                    subprocess.TimeoutExpired,
                    KeyboardInterrupt,
                ) as error:
                    # The npm launcher may have a native Codex child process.
                    try:
                        children = psutil.Process(
                            process.pid
                        ).children(recursive=True)
                        for child in children:
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    process.kill()
                    process.communicate()
                    if isinstance(error, KeyboardInterrupt):
                        raise
                    raise TimeoutError(
                        f"Codex exceeded codex_config.timeout={self.timeout}s"
                    ) from error
                if process.returncode:
                    raise RuntimeError(
                        f"Codex exited with status {process.returncode}. "
                        "Check `codex login status`, model access, and CLI version.\n"
                        + stderr.strip()[-2000:]
                    )
            response = (
                output.read_text(encoding="utf-8").strip()
                if output.is_file()
                else ""
            )
            if not response:
                raise RuntimeError(
                    "Codex completed without a final text response"
                )
            return response
