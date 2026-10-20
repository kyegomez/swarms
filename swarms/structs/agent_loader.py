import csv
import json
import os
import traceback
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    ValidationError,
)
from tqdm import tqdm

from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.structs.agent import Agent
from swarms.utils.types import ReturnTypes


class AgentConfig(BaseModel):
    """
    Agent configuration read from a Markdown, CSV, JSON or YAML file.

    Field names are Agent parameters; fields left unset fall back to the Agent defaults.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    agent_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("agent_name", "name"),
    )
    agent_description: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices(
            "agent_description", "description"
        ),
    )
    system_prompt: Optional[str] = Field(default=None, min_length=1)
    model_name: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("model_name", "model"),
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    max_loops: Optional[Union[PositiveInt, Literal["auto"]]] = None
    mcp_url: Optional[str] = None
    autosave: Optional[bool] = None
    dashboard: Optional[bool] = None
    verbose: Optional[bool] = None
    dynamic_temperature_enabled: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices(
            "dynamic_temperature_enabled", "dynamic_temperature"
        ),
    )
    saved_state_path: Optional[str] = None
    user_name: Optional[str] = None
    retry_attempts: Optional[int] = Field(default=None, ge=1)
    context_length: Optional[int] = Field(default=None, ge=1)
    return_step_meta: Optional[bool] = None
    output_type: Optional[str] = None
    auto_generate_prompt: Optional[bool] = None
    streaming_on: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("streaming_on", "streaming"),
    )

    def agent_kwargs(self) -> Dict[str, Any]:
        """
        Return the configured fields as Agent keyword arguments.

        Returns:
            Dict[str, Any]: The fields that were set and are not None.
        """
        return self.model_dump(exclude_none=True)


class AgentLoader:
    """
    Loader class for creating Agent objects from various file formats.

    This class provides methods to load agents from Markdown, YAML, CSV and JSON files.
    """

    def __init__(
        self,
        concurrent: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize the AgentLoader instance.

        Args:
            concurrent (bool): Whether to load files concurrently.
            max_workers (Optional[int]): Worker threads for concurrent loading. Defaults to twice the CPU count.
        """
        self.concurrent = concurrent
        self.max_workers = max_workers or (os.cpu_count() or 1) * 2

    def parse_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML frontmatter from markdown content.

        Args:
            content: Markdown content with potential YAML frontmatter

        Returns:
            Dictionary with parsed YAML data and remaining content
        """
        lines = content.split("\n")

        if not lines[0].strip() == "---":
            return {"frontmatter": {}, "content": content}

        end_marker = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == "---":
                end_marker = i
                break

        if end_marker == -1:
            return {"frontmatter": {}, "content": content}

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

    def parse_markdown_file(self, file_path: str) -> AgentConfig:
        """
        Parse a Claude Code sub-agent markdown file into an agent configuration.

        Args:
            file_path: Path to a markdown file with YAML frontmatter; the body is the system prompt.

        Returns:
            AgentConfig: The parsed configuration.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file has no frontmatter or fails validation.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Markdown file {file_path} not found."
            )

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            yaml_result = self.parse_yaml_frontmatter(content)
            frontmatter = yaml_result["frontmatter"]

            if not frontmatter:
                raise ValueError(
                    f"No YAML frontmatter found in {file_path}. File must use Claude Code sub-agent format with YAML frontmatter."
                )

            config = AgentConfig.model_validate(
                {
                    "name": Path(file_path).stem,
                    "description": "Agent loaded from markdown",
                    **frontmatter,
                    "system_prompt": yaml_result["content"].strip(),
                }
            )

            logger.info(
                f"Successfully parsed markdown file: {file_path}"
            )
            return config

        except Exception as e:
            logger.error(
                f"Error parsing markdown file {file_path}: {str(e)}"
            )
            raise ValueError(
                f"Error parsing markdown file {file_path}: {str(e)}"
            )

    def load_agent_from_markdown(
        self, file_path: str, **kwargs
    ) -> Agent:
        """
        Load a single agent from a Markdown file.

        Args:
            file_path (str): Path to the Markdown file containing the agent definition.
            **kwargs: Agent arguments that override the parsed configuration.

        Returns:
            Agent: The loaded Agent object.
        """
        config = self.parse_markdown_file(file_path)

        try:
            logger.info(
                f"Creating agent '{config.agent_name}' from {file_path}"
            )
            agent = Agent(**{**config.agent_kwargs(), **kwargs})
            logger.info(
                f"Successfully created agent '{config.agent_name}' from {file_path}"
            )
            return agent
        except Exception as e:
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
    ) -> List[Agent]:
        """
        Load multiple agents from one or more Markdown files.

        Args:
            file_paths (Union[str, List[str]]): A Markdown file, a directory of them, or a list of paths.
            concurrent (bool, optional): Whether to load files concurrently. Defaults to True.
            max_file_size_mb (float, optional): Maximum file size in MB to process. Defaults to 10.0.
            **kwargs: Agent arguments applied to every loaded agent.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        agents = []
        paths_to_process = []

        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                md_files = list(Path(file_paths).glob("*.md"))
                paths_to_process = [str(f) for f in md_files]
            elif os.path.isfile(file_paths):
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

        if concurrent and len(paths_to_process) > 1:
            logger.info(
                f"Loading {len(paths_to_process)} agents concurrently with {self.max_workers} workers..."
            )

            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                future_to_path = {
                    executor.submit(
                        self.load_agent_from_markdown,
                        file_path,
                        **kwargs,
                    ): file_path
                    for file_path in paths_to_process
                }

                for future in as_completed(
                    future_to_path, timeout=300
                ):
                    file_path = future_to_path[future]
                    try:
                        agent = future.result(timeout=60)
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

    def load_agents_from_yaml(
        self,
        yaml_file: str,
        return_type: ReturnTypes = "auto",
        **kwargs,
    ) -> List[Agent]:
        """
        Load agents from a YAML file.

        Args:
            yaml_file (str): Path to the YAML file containing agent definitions.
            return_type (ReturnTypes, optional): The return type for the loader. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        return create_agents_from_yaml(
            yaml_file=yaml_file, return_type=return_type, **kwargs
        )

    def load_many_agents_from_yaml(
        self,
        yaml_files: List[str],
        return_types: List[ReturnTypes] = ["auto"],
        **kwargs,
    ) -> List[Agent]:
        """
        Load agents from multiple YAML files.

        Args:
            yaml_files (List[str]): List of YAML file paths containing agent definitions.
            return_types (List[ReturnTypes], optional): List of return types for each YAML file. Defaults to ["auto"].
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects from all files.
        """
        return [
            self.load_agents_from_yaml(
                yaml_file=yaml_file,
                return_type=return_types[i],
                **kwargs,
            )
            for i, yaml_file in enumerate(yaml_files)
        ]

    def load_agents_from_csv(
        self, csv_file: Union[str, Path], **kwargs
    ) -> List[Agent]:
        """
        Load agents from a CSV, JSON or YAML file of agent rows, in file order.

        Args:
            csv_file (Union[str, Path]): Path to the file containing agent definitions.
            **kwargs: Agent arguments applied to every loaded agent.

        Returns:
            List[Agent]: A list of loaded Agent objects. Invalid rows are skipped.
        """
        file_path = Path(csv_file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")

        suffix = self._suffix(file_path)
        with open(file_path, "r", newline="") as f:
            if suffix == ".csv":
                # A blank cell means unset, not an empty value.
                rows = [
                    {
                        k: v
                        for k, v in row.items()
                        if v not in ("", None)
                    }
                    for row in csv.DictReader(f)
                ]
            elif suffix == ".json":
                rows = json.load(f)
            else:
                rows = yaml.safe_load(f)

        agents: List[Agent] = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(self._process_agent, row, kwargs)
                for row in rows
            ]

            for future in tqdm(
                futures,
                total=len(futures),
                desc="Loading agents",
            ):
                try:
                    agent = future.result()
                    if agent:
                        agents.append(agent)
                except Exception as e:
                    print(f"Error processing agent: {e}")

        print(f"Loaded {len(agents)} agents from {file_path}")
        return agents

    def create_agent_file(
        self,
        file_path: Union[str, Path],
        agents: List[Union[BaseModel, Dict[str, Any]]],
    ) -> None:
        """
        Validate agent configurations and write them to a CSV, JSON or YAML file.

        Args:
            file_path (Union[str, Path]): Destination file; its extension picks the format.
            agents (List[Union[BaseModel, Dict[str, Any]]]): Agent configurations to write.

        Raises:
            ValidationError: If a configuration is invalid.
        """
        file_path = Path(file_path)
        suffix = self._suffix(file_path)

        rows = []
        for agent in agents:
            data = (
                agent.model_dump()
                if isinstance(agent, BaseModel)
                else agent
            )
            rows.append(
                AgentConfig.model_validate(data).agent_kwargs()
            )

        with open(file_path, "w", newline="") as f:
            if suffix == ".csv":
                fieldnames = list(
                    dict.fromkeys(k for r in rows for k in r)
                )
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            elif suffix == ".json":
                json.dump(rows, f, indent=2)
            else:
                yaml.dump(rows, f, default_flow_style=False)

        print(
            f"Created {suffix[1:]} file with {len(rows)} agents at {file_path}"
        )

    def auto(self, file_path: str, *args, **kwargs):
        """
        Automatically load agents from a file based on its extension.

        Args:
            file_path (str): Path to the agent file (Markdown, YAML, or CSV).
            *args: Additional positional arguments passed to the underlying loader.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.

        Raises:
            ValueError: If the file type is not supported.
        """
        if file_path.endswith(".md"):
            return self.load_agents_from_markdown(
                file_path, *args, **kwargs
            )
        elif file_path.endswith(".yaml"):
            return self.load_agents_from_yaml(
                file_path, *args, **kwargs
            )
        elif file_path.endswith(".csv"):
            return self.load_agents_from_csv(
                file_path, *args, **kwargs
            )
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def load_single_agent(self, *args, **kwargs):
        """
        Load a single agent from a file of a supported type.

        Args:
            *args: Positional arguments passed to the underlying loader.
            **kwargs: Keyword arguments passed to the underlying loader.

        Returns:
            Agent: The loaded Agent object.
        """
        return self.auto(*args, **kwargs)

    def load_multiple_agents(
        self, file_paths: List[str], *args, **kwargs
    ):
        """
        Load multiple agents from a list of files of various supported types.

        Args:
            file_paths (List[str]): List of file paths to agent files (Markdown, YAML, or CSV).
            *args: Additional positional arguments passed to the underlying loader.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects from all files.
        """
        return [
            self.auto(file_path, *args, **kwargs)
            for file_path in file_paths
        ]

    @staticmethod
    def _suffix(file_path: Path) -> str:
        """
        Return the file's format as a normalised extension.

        Args:
            file_path: The agent file.

        Returns:
            str: ".csv", ".json" or ".yaml".

        Raises:
            ValueError: If the extension is not supported.
        """
        ext = file_path.suffix.lower()
        if ext == ".yml":
            return ".yaml"
        if ext not in (".csv", ".json", ".yaml"):
            raise ValueError(f"Unsupported file type: {ext}")
        return ext

    def _process_agent(
        self,
        agent_data: Union[BaseModel, Dict[str, Any]],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """
        Validate one agent row and build its Agent.

        Args:
            agent_data: The row, as a dict or pydantic model.
            overrides: Agent arguments that take precedence over the row.

        Returns:
            Optional[Agent]: The agent, or None when the row is invalid.
        """
        if isinstance(agent_data, BaseModel):
            agent_data = agent_data.model_dump()
        try:
            config = AgentConfig.model_validate(agent_data)
        except ValidationError as e:
            print(f"Skipping invalid agent configuration: {e}")
            return None
        return self._create_agent(
            {**config.agent_kwargs(), **(overrides or {})}
        )

    def _create_agent(self, agent_kwargs: Dict[str, Any]) -> Agent:
        """
        Create an Agent from validated keyword arguments.

        Args:
            agent_kwargs: Agent constructor arguments.

        Returns:
            Agent: The new agent.
        """
        return Agent(**agent_kwargs)


def load_agent_from_markdown(file_path: str, **kwargs) -> Agent:
    """
    Load a single agent from a markdown file using the Claude Code YAML frontmatter format.

    Args:
        file_path (str): Path to the markdown file containing YAML frontmatter.
        **kwargs: Agent arguments that override the parsed configuration.

    Returns:
        Agent: Configured Agent instance loaded from the markdown file.

    Example:
        >>> agent = load_agent_from_markdown("finance_advisor.md", max_loops=3)
    """
    return AgentLoader().load_agent_from_markdown(file_path, **kwargs)


def load_agents_from_markdown(
    file_paths: Union[str, List[str]],
    concurrent: bool = True,
    max_file_size_mb: float = 10.0,
    **kwargs,
) -> List[Agent]:
    """
    Load agents from markdown files using the Claude Code YAML frontmatter format.

    Args:
        file_paths (Union[str, List[str]]): A directory of markdown files or a list of paths.
        concurrent (bool, optional): Load files concurrently. Defaults to True.
        max_file_size_mb (float, optional): Files larger than this are skipped. Defaults to 10.0.
        **kwargs: Agent arguments applied to every loaded agent.

    Returns:
        List[Agent]: Configured Agent instances loaded from the markdown files.

    Example:
        >>> agents = load_agents_from_markdown(["agent1.md", "agent2.md"], max_loops=2)
    """
    return AgentLoader().load_agents_from_markdown(
        file_paths,
        concurrent=concurrent,
        max_file_size_mb=max_file_size_mb,
        **kwargs,
    )
