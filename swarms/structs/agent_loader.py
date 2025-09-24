import os
from typing import List, Union

from swarms.agents.create_agents_from_yaml import (
    create_agents_from_yaml,
)
from swarms.utils.types import ReturnTypes
from swarms.structs.agent import Agent
from swarms.structs.csv_to_agent import CSVAgentLoader
from swarms.utils.agent_loader_markdown import (
    load_agent_from_markdown,
    load_agents_from_markdown,
    MarkdownAgentLoader,
)


class AgentLoader:
    """
    Loader class for creating Agent objects from various file formats.

    This class provides methods to load agents from Markdown, YAML, and CSV files.
    """

    def __init__(self, concurrent: bool = True):
        """
        Initialize the AgentLoader instance.
        """
        self.concurrent = concurrent
        pass

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
            file_paths (Union[str, List[str]]): Path or list of paths to Markdown file(s) containing agent definitions.
            concurrent (bool, optional): Whether to load files concurrently. Defaults to True.
            max_file_size_mb (float, optional): Maximum file size in MB to process. Defaults to 10.0.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        return load_agents_from_markdown(
            file_paths=file_paths,
            concurrent=concurrent,
            max_file_size_mb=max_file_size_mb,
            **kwargs,
        )

    def load_agent_from_markdown(
        self, file_path: str, **kwargs
    ) -> Agent:
        """
        Load a single agent from a Markdown file.

        Args:
            file_path (str): Path to the Markdown file containing the agent definition.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            Agent: The loaded Agent object.
        """
        return load_agent_from_markdown(file_path=file_path, **kwargs)

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
        self, csv_file: str, **kwargs
    ) -> List[Agent]:
        """
        Load agents from a CSV file.

        Args:
            csv_file (str): Path to the CSV file containing agent definitions.
            **kwargs: Additional keyword arguments passed to the underlying loader.

        Returns:
            List[Agent]: A list of loaded Agent objects.
        """
        loader = CSVAgentLoader(file_path=csv_file)
        return loader.load_agents()

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

    def parse_markdown_file(self, file_path: str):
        """
        Parse a Markdown file and return the agents defined within.

        Args:
            file_path (str): Path to the Markdown file.

        Returns:
            List[Agent]: A list of Agent objects parsed from the file.
        """
        return MarkdownAgentLoader(
            max_workers=os.cpu_count()
        ).parse_markdown_file(file_path=file_path)
