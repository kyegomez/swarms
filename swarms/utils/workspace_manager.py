"""
Simple workspace management functions for creating files and folders.

Raw utility functions for easy file and folder creation operations.
"""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any


def create_folder(
    folder_name: str, parent_path: Optional[str] = None
) -> Path:
    """
    Create a new folder.

    Args:
        folder_name: Name of the folder to create
        parent_path: Parent directory path. If None, creates in current directory.

    Returns:
        Path object of the created folder
    """
    if parent_path:
        folder_path = Path(parent_path) / folder_name
    else:
        folder_path = Path(folder_name)

    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def file_exists(
    file_name: str, parent_path: Optional[str] = None
) -> bool:
    """
    Check if a file exists.

    Args:
        file_name: Name of the file to check
        parent_path: Parent directory path. If None, checks in current directory.

    Returns:
        True if file exists, False otherwise
    """
    if parent_path:
        file_path = Path(parent_path) / file_name
    else:
        file_path = Path(file_name)

    return file_path.exists() and file_path.is_file()


def update_file(
    file_name: str, content: str, parent_path: Optional[str] = None
) -> Path:
    """
    Update an existing file with new content.

    Args:
        file_name: Name of the file to update
        content: New content to write to the file
        parent_path: Parent directory path. If None, updates in current directory.

    Returns:
        Path object of the updated file

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if parent_path:
        file_path = Path(parent_path) / file_name
    else:
        file_path = Path(file_name)

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")

    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_or_update_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a new file or update existing file with content.

    Args:
        file_name: Name of the file to create or update
        content: Content to write to the file
        parent_path: Parent directory path. If None, creates/updates in current directory.

    Returns:
        Path object of the created or updated file
    """
    if parent_path:
        file_path = Path(parent_path) / file_name
    else:
        file_path = Path(file_name)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a new file with content.

    Args:
        file_name: Name of the file to create
        content: Content to write to the file
        parent_path: Parent directory path. If None, creates in current directory.

    Returns:
        Path object of the created file
    """
    if parent_path:
        file_path = Path(parent_path) / file_name
    else:
        file_path = Path(file_name)

    if file_path.exists():
        raise FileExistsError(
            f"File {file_path} already exists. Use create_or_update_file() to update existing files."
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def create_python_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a Python file with content.

    Args:
        file_name: Name of the file (with or without .py extension)
        content: Python code content
        parent_path: Parent directory path

    Returns:
        Path object of the created Python file
    """
    if not file_name.endswith(".py"):
        file_name += ".py"
    return create_file(file_name, content, parent_path)


def create_or_update_python_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a Python file or update existing Python file with content.

    Args:
        file_name: Name of the file (with or without .py extension)
        content: Python code content
        parent_path: Parent directory path

    Returns:
        Path object of the created or updated Python file
    """
    if not file_name.endswith(".py"):
        file_name += ".py"
    return create_or_update_file(file_name, content, parent_path)


def create_json_file(
    file_name: str,
    data: Dict[str, Any],
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a JSON file with data.

    Args:
        file_name: Name of the file (with or without .json extension)
        data: Dictionary data to serialize to JSON
        parent_path: Parent directory path

    Returns:
        Path object of the created JSON file
    """
    if not file_name.endswith(".json"):
        file_name += ".json"
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return create_file(file_name, content, parent_path)


def create_or_update_json_file(
    file_name: str,
    data: Dict[str, Any],
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a JSON file or update existing JSON file with data.

    Args:
        file_name: Name of the file (with or without .json extension)
        data: Dictionary data to serialize to JSON
        parent_path: Parent directory path

    Returns:
        Path object of the created or updated JSON file
    """
    if not file_name.endswith(".json"):
        file_name += ".json"
    content = json.dumps(data, indent=2, ensure_ascii=False)
    return create_or_update_file(file_name, content, parent_path)


def create_yaml_file(
    file_name: str,
    data: Dict[str, Any],
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a YAML file with data.

    Args:
        file_name: Name of the file (with or without .yaml/.yml extension)
        data: Dictionary data to serialize to YAML
        parent_path: Parent directory path

    Returns:
        Path object of the created YAML file
    """
    if not (
        file_name.endswith(".yaml") or file_name.endswith(".yml")
    ):
        file_name += ".yaml"
    content = yaml.dump(
        data, default_flow_style=False, allow_unicode=True
    )
    return create_file(file_name, content, parent_path)


def create_or_update_yaml_file(
    file_name: str,
    data: Dict[str, Any],
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a YAML file or update existing YAML file with data.

    Args:
        file_name: Name of the file (with or without .yaml/.yml extension)
        data: Dictionary data to serialize to YAML
        parent_path: Parent directory path

    Returns:
        Path object of the created or updated YAML file
    """
    if not (
        file_name.endswith(".yaml") or file_name.endswith(".yml")
    ):
        file_name += ".yaml"
    content = yaml.dump(
        data, default_flow_style=False, allow_unicode=True
    )
    return create_or_update_file(file_name, content, parent_path)


def create_markdown_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a Markdown file with content.

    Args:
        file_name: Name of the file (with or without .md extension)
        content: Markdown content
        parent_path: Parent directory path

    Returns:
        Path object of the created Markdown file
    """
    if not file_name.endswith(".md"):
        file_name += ".md"
    return create_file(file_name, content, parent_path)


def create_or_update_markdown_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a Markdown file or update existing Markdown file with content.

    Args:
        file_name: Name of the file (with or without .md extension)
        content: Markdown content
        parent_path: Parent directory path

    Returns:
        Path object of the created or updated Markdown file
    """
    if not file_name.endswith(".md"):
        file_name += ".md"
    return create_or_update_file(file_name, content, parent_path)


def create_text_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a text file with content.

    Args:
        file_name: Name of the file (with or without .txt extension)
        content: Text content
        parent_path: Parent directory path

    Returns:
        Path object of the created text file
    """
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    return create_file(file_name, content, parent_path)


def create_or_update_text_file(
    file_name: str,
    content: str = "",
    parent_path: Optional[str] = None,
) -> Path:
    """
    Create a text file or update existing text file with content.

    Args:
        file_name: Name of the file (with or without .txt extension)
        content: Text content
        parent_path: Parent directory path

    Returns:
        Path object of the created or updated text file
    """
    if not file_name.endswith(".txt"):
        file_name += ".txt"
    return create_or_update_file(file_name, content, parent_path)


def create_empty_file(
    file_name: str, parent_path: Optional[str] = None
) -> Path:
    """
    Create an empty file.

    Args:
        file_name: Name of the file
        parent_path: Parent directory path

    Returns:
        Path object of the created empty file
    """
    return create_file(file_name, "", parent_path)


def create_project_structure(
    structure: Dict[str, Any], parent_path: Optional[str] = None
) -> Dict[str, Path]:
    """
    Create a nested project structure from a dictionary.

    Args:
        structure: Dictionary defining the project structure
        parent_path: Parent directory path

    Returns:
        Dictionary mapping structure keys to created Path objects

    Example:
        structure = {
            "src": {
                "main.py": "print('Hello World')",
                "utils": {
                    "__init__.py": "",
                    "helper.py": "def helper(): pass"
                }
            },
            "tests": {
                "test_main.py": "import unittest"
            },
            "README.md": "# My Project"
        }
    """
    created_paths = {}
    base_path = Path(parent_path) if parent_path else Path.cwd()

    def _create_structure(structure_dict, current_path):
        for key, value in structure_dict.items():
            item_path = current_path / key

            if isinstance(value, dict):
                # It's a folder
                item_path.mkdir(parents=True, exist_ok=True)
                created_paths[key] = item_path
                _create_structure(value, item_path)
            else:
                # It's a file
                content = str(value) if value is not None else ""
                item_path.parent.mkdir(parents=True, exist_ok=True)
                item_path.write_text(content, encoding="utf-8")
                created_paths[key] = item_path

    _create_structure(structure, base_path)
    return created_paths
