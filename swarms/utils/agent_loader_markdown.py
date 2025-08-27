import os
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator

# Type checking imports to avoid circular dependency
if TYPE_CHECKING:
    from swarms.structs.agent import Agent

# Lazy import to avoid circular dependency

# Default model configuration
DEFAULT_MODEL = "gpt-4.1"


class MarkdownAgentConfig(BaseModel):
    """Configuration model for agents loaded from Claude Code markdown files."""

    name: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    mcp_url: Optional[int] = None
    system_prompt: Optional[str] = None
    max_loops: Optional[int] = Field(default=1, ge=1)
    autosave: Optional[bool] = False
    dashboard: Optional[bool] = False
    verbose: Optional[bool] = False
    dynamic_temperature_enabled: Optional[bool] = False
    saved_state_path: Optional[str] = None
    user_name: Optional[str] = "default_user"
    retry_attempts: Optional[int] = Field(default=3, ge=1)
    context_length: Optional[int] = Field(default=100000, ge=1000)
    return_step_meta: Optional[bool] = False
    output_type: Optional[str] = "str"
    auto_generate_prompt: Optional[bool] = False
    streaming_on: Optional[bool] = False

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError(
                "System prompt must be a non-empty string"
            )
        return v


class MarkdownAgentLoader:
    """
    Loader for creating agents from markdown files using Claude Code sub-agent format.

    Supports both single markdown file and multiple markdown files.
    Uses YAML frontmatter format for agent configuration.

    Features:
    - Single markdown file loading
    - Multiple markdown files loading (batch processing)
    - YAML frontmatter parsing
    - Agent configuration extraction from YAML metadata
    - Error handling and validation
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the AgentLoader.
        """
        self.max_workers = max_workers
        self.max_workers = os.cpu_count() * 2

    def parse_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            content: Markdown content with potential YAML frontmatter

        Returns:
            Dictionary with parsed YAML data and remaining content
        """
        lines = content.split("\n")

        # Check if content starts with YAML frontmatter
        if not lines[0].strip() == "---":
            return {"frontmatter": {}, "content": content}

        # Find end of frontmatter
        end_marker = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end_marker = i
                break

        if end_marker == -1:
            return {"frontmatter": {}, "content": content}

        # Extract frontmatter and content
        frontmatter_text = "\n".join(lines[1:end_marker])
        remaining_content = "\n".join(lines[end_marker + 1 :]).strip()

        try:
            frontmatter_data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            return {"frontmatter": {}, "content": content}

        return {
            "frontmatter": frontmatter_data,
            "content": remaining_content,
        }

    def parse_markdown_file(
        self, file_path: str
    ) -> MarkdownAgentConfig:
        """
        Parse a single markdown file to extract agent configuration.
        Uses Claude Code sub-agent YAML frontmatter format.

        Args:
            file_path: Path to markdown file

        Returns:
            MarkdownAgentConfig object with parsed configuration

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parsing fails or no YAML frontmatter found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Markdown file {file_path} not found."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Parse YAML frontmatter (Claude Code sub-agent format)
            yaml_result = self.parse_yaml_frontmatter(content)
            frontmatter = yaml_result["frontmatter"]
            remaining_content = yaml_result["content"]

            if not frontmatter:
                raise ValueError(
                    f"No YAML frontmatter found in {file_path}. File must use Claude Code sub-agent format with YAML frontmatter."
                )

            # Use YAML frontmatter data
            config_data = {
                "name": frontmatter.get("name", Path(file_path).stem),
                "description": frontmatter.get(
                    "description", "Agent loaded from markdown"
                ),
                "model_name": frontmatter.get("model_name")
                or frontmatter.get("model", DEFAULT_MODEL),
                "temperature": frontmatter.get("temperature", 0.1),
                "max_loops": frontmatter.get("max_loops", 1),
                "mcp_url": frontmatter.get("mcp_url"),
                "system_prompt": remaining_content.strip(),
                "streaming_on": frontmatter.get(
                    "streaming_on", False
                ),
            }

            # Use default model if not specified
            if not config_data["model_name"]:
                config_data["model_name"] = DEFAULT_MODEL

            logger.info(
                f"Successfully parsed markdown file: {file_path}"
            )
            return MarkdownAgentConfig(**config_data)

        except Exception as e:
            logger.error(
                f"Error parsing markdown file {file_path}: {str(e)}"
            )
            raise ValueError(
                f"Error parsing markdown file {file_path}: {str(e)}"
            )

    def load_agent_from_markdown(
        self, file_path: str, **kwargs
    ) -> "Agent":
        """
        Load a single agent from a markdown file.

        Args:
            file_path: Path to markdown file
            **kwargs: Additional arguments to override default configuration

        Returns:
            Configured Agent instance
        """
        # Lazy import to avoid circular dependency
        from swarms.structs.agent import Agent

        config = self.parse_markdown_file(file_path)

        # Override with any provided kwargs
        config_dict = config.model_dump()
        config_dict.update(kwargs)

        # Map config fields to Agent parameters, handling special cases
        field_mapping = {
            "name": "agent_name",  # name -> agent_name
            "description": "agent_description",  # not used by Agent
            "mcp_url": "mcp_url",  # not used by Agent
        }

        agent_fields = {}
        for config_key, config_value in config_dict.items():
            # Handle special field mappings
            if config_key in field_mapping:
                agent_key = field_mapping[config_key]
                if agent_key:  # Only include if mapped to something
                    agent_fields[agent_key] = config_value
            else:
                # Direct mapping for most fields
                agent_fields[config_key] = config_value

        try:

            logger.info(
                f"Creating agent '{config.name}' from {file_path}"
            )
            agent = Agent(**agent_fields)
            logger.info(
                f"Successfully created agent '{config.name}' from {file_path}"
            )
            return agent
        except Exception as e:
            import traceback

            logger.error(
                f"Error creating agent from {file_path}: {str(e)}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(
                f"Error creating agent from {file_path}: {str(e)}"
            )

    def load_agents_from_markdown(
        self,
        file_paths: Union[str, List[str]],
        concurrent: bool = True,
        max_file_size_mb: float = 10.0,
        **kwargs,
    ) -> List["Agent"]:
        """
        Load multiple agents from markdown files with optional concurrent processing.

        Args:
            file_paths: Single file path, directory path, or list of file paths
            concurrent: Whether to use concurrent processing for multiple files
            max_file_size_mb: Maximum file size in MB to prevent memory issues
            **kwargs: Additional arguments to override default configuration

        Returns:
            List of configured Agent instances
        """
        # Lazy import to avoid circular dependency

        agents = []
        paths_to_process = []

        # Handle different input types
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                # Directory - find all .md files
                md_files = list(Path(file_paths).glob("*.md"))
                paths_to_process = [str(f) for f in md_files]
            elif os.path.isfile(file_paths):
                # Single file
                paths_to_process = [file_paths]
            else:
                raise FileNotFoundError(
                    f"Path {file_paths} not found."
                )
        elif isinstance(file_paths, list):
            paths_to_process = file_paths
        else:
            raise ValueError(
                "file_paths must be a string or list of strings"
            )

        # Validate file sizes to prevent memory issues
        for file_path in paths_to_process:
            try:
                file_size_mb = os.path.getsize(file_path) / (
                    1024 * 1024
                )
                if file_size_mb > max_file_size_mb:
                    logger.warning(
                        f"Skipping {file_path}: size {file_size_mb:.2f}MB exceeds limit {max_file_size_mb}MB"
                    )
                    paths_to_process.remove(file_path)
            except OSError:
                logger.warning(
                    f"Could not check size of {file_path}, skipping validation"
                )

        # Use concurrent processing for multiple files if enabled
        if concurrent and len(paths_to_process) > 1:
            logger.info(
                f"Loading {len(paths_to_process)} agents concurrently with {self.max_workers} workers..."
            )

            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Submit all tasks
                future_to_path = {
                    executor.submit(
                        self.load_agent_from_markdown,
                        file_path,
                        **kwargs,
                    ): file_path
                    for file_path in paths_to_process
                }

                # Collect results as they complete with timeout
                for future in as_completed(
                    future_to_path, timeout=300
                ):  # 5 minute timeout
                    file_path = future_to_path[future]
                    try:
                        agent = future.result(
                            timeout=60
                        )  # 1 minute per agent
                        agents.append(agent)
                        logger.info(
                            f"Successfully loaded agent from {file_path}"
                        )
                    except TimeoutError:
                        logger.error(f"Timeout loading {file_path}")
                        continue
                    except Exception as e:
                        logger.error(
                            f"Failed to load {file_path}: {str(e)}"
                        )
                        continue
        else:
            # Sequential processing for single file or when concurrent is disabled
            logger.info(
                f"Loading {len(paths_to_process)} agents sequentially..."
            )
            for file_path in paths_to_process:
                try:
                    agent = self.load_agent_from_markdown(
                        file_path, **kwargs
                    )
                    agents.append(agent)
                except Exception as e:
                    logger.warning(
                        f"Skipping {file_path} due to error: {str(e)}"
                    )
                    continue

        logger.info(
            f"Successfully loaded {len(agents)} agents from markdown files"
        )
        return agents

    def load_single_agent(self, file_path: str, **kwargs) -> "Agent":
        """
        Convenience method for loading a single agent.
        Uses Claude Code sub-agent YAML frontmatter format.

        Args:
            file_path: Path to markdown file with YAML frontmatter
            **kwargs: Additional configuration overrides

        Returns:
            Configured Agent instance
        """
        # Lazy import to avoid circular dependency

        return self.load_agent_from_markdown(file_path, **kwargs)

    def load_multiple_agents(
        self, file_paths: Union[str, List[str]], **kwargs
    ) -> List["Agent"]:
        """
        Convenience method for loading multiple agents.
        Uses Claude Code sub-agent YAML frontmatter format.

        Args:
            file_paths: Directory path or list of file paths with YAML frontmatter
            **kwargs: Additional configuration overrides

        Returns:
            List of configured Agent instances
        """
        # Lazy import to avoid circular dependency

        return self.load_agents_from_markdown(file_paths, **kwargs)


# Convenience functions
def load_agent_from_markdown(file_path: str, **kwargs) -> "Agent":
    """
    Load a single agent from a markdown file using the Claude Code YAML frontmatter format.

    This function provides a simple interface for loading an agent configuration
    from a markdown file. It supports all configuration overrides accepted by the
    underlying `AgentLoader` and agent class.

    Args:
        file_path (str): Path to the markdown file containing YAML frontmatter
            with agent configuration.
        **kwargs: Optional keyword arguments to override agent configuration
            parameters. Common options include:
                - max_loops (int): Maximum number of reasoning loops.
                - autosave (bool): Enable automatic state saving.
                - dashboard (bool): Enable dashboard monitoring.
                - verbose (bool): Enable verbose logging.
                - dynamic_temperature_enabled (bool): Enable dynamic temperature.
                - saved_state_path (str): Path for saving agent state.
                - user_name (str): User identifier.
                - retry_attempts (int): Number of retry attempts.
                - context_length (int): Maximum context length.
                - return_step_meta (bool): Return step metadata.
                - output_type (str): Output format type.
                - auto_generate_prompt (bool): Auto-generate prompts.
                - artifacts_on (bool): Enable artifacts.
                - streaming_on (bool): Enable streaming output.
                - mcp_url (str): MCP server URL if needed.

    Returns:
        Agent: Configured Agent instance loaded from the markdown file.

    Example:
        >>> agent = load_agent_from_markdown("finance_advisor.md", max_loops=3, verbose=True)
        >>> response = agent.run("What is the best investment strategy for 2024?")
    """
    # Lazy import to avoid circular dependency

    loader = MarkdownAgentLoader()
    return loader.load_single_agent(file_path, **kwargs)


def load_agents_from_markdown(
    file_paths: Union[str, List[str]],
    concurrent: bool = True,
    max_file_size_mb: float = 10.0,
    **kwargs,
) -> List["Agent"]:
    """
    Load multiple agents from markdown files using the Claude Code YAML frontmatter format.

    This function supports loading agents from a list of markdown files or from all
    markdown files in a directory. It can process files concurrently for faster loading,
    and allows configuration overrides for all loaded agents.

    Args:
        file_paths (Union[str, List[str]]): Either a directory path containing markdown
            files or a list of markdown file paths to load.
        concurrent (bool, optional): If True, enables concurrent processing for faster
            loading of multiple files. Defaults to True.
        max_file_size_mb (float, optional): Maximum file size (in MB) for each markdown
            file to prevent memory issues. Files exceeding this size will be skipped.
            Defaults to 10.0.
        **kwargs: Optional keyword arguments to override agent configuration
            parameters for all loaded agents. See `load_agent_from_markdown` for
            available options.

    Returns:
        List[Agent]: List of configured Agent instances loaded from the markdown files.

    Example:
        >>> agents = load_agents_from_markdown(
        ...     ["agent1.md", "agent2.md"],
        ...     concurrent=True,
        ...     max_loops=2,
        ...     verbose=True
        ... )
        >>> for agent in agents:
        ...     print(agent.name)
    """
    # Lazy import to avoid circular dependency

    loader = MarkdownAgentLoader()
    return loader.load_agents_from_markdown(
        file_paths,
        concurrent=concurrent,
        max_file_size_mb=max_file_size_mb,
        **kwargs,
    )
