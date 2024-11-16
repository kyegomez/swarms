from typing import Union, Dict, List
from swarms.artifacts.main_artifact import Artifact


def handle_artifact_outputs(
    file_path: str,
    data: Union[str, Dict, List],
    output_type: str = "txt",
    folder_path: str = "./artifacts",
) -> str:
    """
    Handle different types of data and create files in various formats.

    Args:
        file_path: Path where the file should be saved
        data: Input data that can be string, dict or list
        output_type: Type of output file (txt, md, pdf, csv, json)
        folder_path: Folder to save artifacts

    Returns:
        str: Path to the created file
    """
    # Create artifact with appropriate file type
    artifact = Artifact(
        folder_path=folder_path,
        file_path=file_path,
        file_type=output_type,
        contents=data,
        edit_count=0,
    )

    # Save the file
    # artifact.save()
    artifact.save_as(output_format=output_type)
