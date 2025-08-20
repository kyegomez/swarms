"""
Claude Code Agent Tool - Setup Guide

This tool provides a Claude Code Agent that can:
- Generate code and applications from natural language descriptions
- Write files, execute shell commands, and manage Git repositories
- Perform web searches and file operations
- Handle complex development tasks with retry logic

SETUP GUIDE:
1. Install dependencies:
   pip install claude-code-sdk
   npm install -g @anthropic-ai/claude-code

2. Set environment variable:
   export ANTHROPIC_API_KEY="your-api-key-here"

3. Use the tool:
   from claude_as_a_tool import developer_worker_agent

   result = developer_worker_agent(
       task="Create a Python web scraper",
       system_prompt="You are a helpful coding assistant"
   )

REQUIRED: ANTHROPIC_API_KEY environment variable must be set
"""

import asyncio
from typing import Any, Dict

from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

load_dotenv()


class ClaudeAppGenerator:
    """
    Generates applications using Claude Code SDK based on specifications.
    """

    def __init__(
        self,
        name: str = "Developer Worker Agent",
        description: str = "A developer worker agent that can generate code and write it to a file.",
        retries: int = 3,
        retry_delay: float = 2.0,
        system_prompt: str = None,
        debug_mode: bool = False,
        max_steps: int = 40,
        model: str = "claude-sonnet-4-20250514",
        max_thinking_tokens: int = 1000,
    ):
        """
        Initialize the app generator.

        Args:
            name: Name of the app
            description: Description of the app
            retries: Number of retries
            retry_delay: Delay between retries
            system_prompt: System prompt
            debug_mode: Enable extra verbose logging for Claude outputs
            max_steps: Maximum number of steps
            model: Model to use
        """
        self.name = name
        self.description = description
        self.retries = retries
        self.retry_delay = retry_delay
        self.system_prompt = system_prompt
        self.model = model
        self.debug_mode = debug_mode
        self.max_steps = max_steps
        self.max_thinking_tokens = max_thinking_tokens

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
    )
    async def generate_app_with_claude(
        self, task: str
    ) -> Dict[str, Any]:
        """
        Generate app using Claude Code SDK with robust error handling and retry logic.

        Args:
            task: Task to be completed

        Returns:
            Dict containing generation results
        """
        # Log the Claude SDK configuration
        claude_options = ClaudeCodeOptions(
            system_prompt=self.system_prompt,
            max_turns=self.max_steps,  # Sufficient for local app development and GitHub setup
            allowed_tools=[
                "Read",
                "Write",
                "Bash",
                "GitHub",
                "Git",
                "Grep",
                "WebSearch",
            ],
            continue_conversation=True,  # Start fresh each time
            model=self.model,
            max_thinking_tokens=self.max_thinking_tokens,
        )

        async with ClaudeSDKClient(options=claude_options) as client:
            # Generate the application
            await client.query(task)

            response_text = []
            message_count = 0

            async for message in client.receive_response():
                message_count += 1

                if hasattr(message, "content"):
                    for block in message.content:
                        if hasattr(block, "text"):
                            text_content = block.text
                            response_text.append(text_content)
                            logger.info(text_content)

                        elif hasattr(block, "type"):
                            if self.debug_mode and hasattr(
                                block, "input"
                            ):
                                input_str = str(block.input)
                                if len(input_str) > 200:
                                    input_str = (
                                        input_str[:200]
                                        + "... (truncated)"
                                    )
                                print(f"Tool Input: {input_str}")

                elif type(message).__name__ == "ResultMessage":
                    result_text = str(message.result)
                    response_text.append(result_text)

        return response_text

    def run(self, task: str) -> Dict[str, Any]:
        """
        Synchronous wrapper for app generation to work with ThreadPoolExecutor.

        Args:
            spec: App specification

        Returns:
            Dict containing generation results
        """
        return asyncio.run(self.generate_app_with_claude(task))


def developer_worker_agent(task: str, system_prompt: str) -> str:
    """
    Developer Worker Agent

    This function instantiates a ClaudeAppGenerator agent, which is a highly capable developer assistant designed to automate software development tasks.
    The agent leverages the Claude Code SDK to interpret natural language instructions and generate code, scripts, or even entire applications.
    It can interact with files, execute shell commands, perform web searches, and utilize version control systems such as Git and GitHub.
    The agent is robust, featuring retry logic, customizable system prompts, and debug modes for verbose output.
    It is ideal for automating repetitive coding tasks, prototyping, and integrating with developer workflows.

    Capabilities:
    - Generate code based on detailed task descriptions.
    - Write generated code to files.
    - Execute shell commands and scripts.
    - Interact with Git and GitHub for version control operations.
    - Perform web searches to gather information or code snippets.
    - Provide detailed logs and debugging information if enabled.
    - Handle errors gracefully with configurable retry logic.

    Args:
        task (str): The development task or instruction for the agent to complete.
        system_prompt (str): The system prompt to guide the agent's behavior and context.

    Returns:
        str: The result of the agent's execution for the given task.
    """
    claude_code_sdk = ClaudeAppGenerator(system_prompt=system_prompt)
    return claude_code_sdk.run(task)


# agent = Agent(
#     agent_name="Developer Worker Agent",
#     agent_description="A developer worker agent that can generate code and write it to a file.",
#     tools=[developer_worker_agent],
#     system_prompt="You are a developer worker agent. You are given a task and you need to complete it.",
# )

# agent.run(
#     task="Write a simple python script that prints 'Hello, World!'"
# )

# if __name__ == "__main__":
#     task = "Write a simple python script that prints 'Hello, World!'"
#     system_prompt = "You are a developer worker agent. You are given a task and you need to complete it."
#     print(developer_worker_agent(task, system_prompt))
