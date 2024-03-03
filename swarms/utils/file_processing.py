import logging
import tempfile
import shutil
import os
import re
from swarms.utils.parse_code import extract_code_from_markdown
import json


def zip_workspace(workspace_path: str, output_filename: str):
    """
    Zips the specified workspace directory and returns the path to the zipped file.
    Ensure the output_filename does not have .zip extension as it's added by make_archive.
    """
    temp_dir = tempfile.mkdtemp()
    # Remove .zip if present in output_filename to avoid duplication
    base_output_path = os.path.join(
        temp_dir, output_filename.replace(".zip", "")
    )
    zip_path = shutil.make_archive(
        base_output_path, "zip", workspace_path
    )
    return zip_path  # make_archive already appends .zip


def sanitize_file_path(file_path: str):
    """
    Cleans and sanitizes the file path to be valid for Windows.
    """
    sanitized_path = file_path.replace("`", "").strip()
    # Replace any invalid characters here with an underscore or remove them
    sanitized_path = re.sub(r'[<>:"/\\|?*]', "_", sanitized_path)
    return sanitized_path


def parse_tagged_output(output, workspace_path: str):
    """
    Parses tagged output and saves files to the workspace directory.
    Adds logging for each step of the process.
    """
    pattern = r"<!--START_FILE_PATH-->(.*?)<!--END_FILE_PATH-->(.*?)<!--START_CONTENT-->(.*?)<!--END_CONTENT-->"
    files = re.findall(pattern, output, re.DOTALL)
    if not files:
        logging.error("No files found in the output to parse.")
        return

    for file_path, _, content in files:
        sanitized_path = sanitize_file_path(file_path)
        content = extract_code_from_markdown(
            content
        )  # Remove code block markers
        full_path = os.path.join(workspace_path, sanitized_path)
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as file:
                file.write(content.strip())
            logging.info(f"File saved: {full_path}")
        except Exception as e:
            logging.error(
                f"Failed to save file {sanitized_path}: {e}"
            )


def load_json(json_string: str):
    """
    Loads a JSON string and returns the corresponding Python object.

    Args:
        json_string (str): The JSON string to be loaded.

    Returns:
        object: The Python object representing the JSON data.
    """
    json_data = json.loads(json_string)
    return json_data


# Create file that
def create_file(
    content: str,
    file_path: str,
):
    """
    Creates a file with the specified content at the specified file path.

    Args:
        content (str): The content to be written to the file.
        file_path (str): The path to the file to be created.
    """
    with open(file_path, "w") as file:
        file.write(content)
    return file_path
