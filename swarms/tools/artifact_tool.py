from typing import List

from swarms.artifacts.main_artifact import Artifact, FileVersion


def artifact_save(
    file_path: str,
    file_type: str,
    contents: str = "",
    versions: List[FileVersion] = [],
    edit_count: int = 0,
):
    """
    Saves an artifact with the given file path, file type, contents, versions, and edit count.

    Args:
        file_path (str): The path of the file.
        file_type (str): The type of the file.
        contents (str, optional): The contents of the file. Defaults to an empty string.
        versions (List[FileVersion], optional): The list of file versions. Defaults to an empty list.
        edit_count (int, optional): The number of times the file has been edited. Defaults to 0.

    Returns:
        Artifact: The saved artifact.
    """
    out = Artifact(
        file_path=file_path,
        file_type=file_type,
        contents=contents,
        versions=versions,
        edit_count=edit_count,
    )

    out.save()

    return out


def edit_artifact(
    file_path: str,
    file_type: str,
    contents: str = "",
    versions: List[FileVersion] = [],
    edit_count: int = 0,
):
    """
    Edits an artifact with the given file path, file type, contents, versions, and edit count.

    Args:
        file_path (str): The path of the file.
        file_type (str): The type of the file.
        contents (str, optional): The contents of the file. Defaults to an empty string.
        versions (List[FileVersion], optional): The list of file versions. Defaults to an empty list.
        edit_count (int, optional): The number of times the file has been edited. Defaults to 0.

    Returns:
        Artifact: The edited artifact.
    """
    out = Artifact(
        file_path=file_path,
        file_type=file_type,
        contents=contents,
        versions=versions,
        edit_count=edit_count,
    )

    out.edit(contents)

    return out
