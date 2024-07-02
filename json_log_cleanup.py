import os
import shutil
from loguru import logger


def cleanup_json_logs(name: str = None):
    # Define the root directory and the target directory
    root_dir = os.getcwd()
    target_dir = os.path.join(root_dir, name)

    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # If the file is a JSON log file, .log file or .txt file
            if (
                filename.endswith(".json")
                or filename.endswith(".log")
                # or filename.endswith(".txt")
            ):
                # Construct the full file paths
                file_path = os.path.join(dirpath, filename)
                target_path = os.path.join(target_dir, filename)

                # Move the file to the target directory
                shutil.move(file_path, target_path)
                logger.info(f"Moved file {file_path} to {target_path}")

    # Delete Chroma and Ruff cache
    chroma_cache = os.path.join(root_dir, ".chroma_cache")
    ruff_cache = os.path.join(root_dir, ".ruff_cache")
    dist_cache = os.path.join(root_dir, "dist")

    if os.path.exists(chroma_cache):
        shutil.rmtree(chroma_cache)
        logger.info(f"Deleted Chroma cache at {chroma_cache}")

    if os.path.exists(ruff_cache):
        shutil.rmtree(ruff_cache)
        logger.info(f"Deleted Ruff cache at {ruff_cache}")

    if os.path.exists(dist_cache):
        shutil.rmtree(dist_cache)
        logger.info(f"Deleted the dist cache at {dist_cache}")

    # Delete the "chroma" folder
    chroma_folder = os.path.join(root_dir, "chroma")
    if os.path.exists(chroma_folder):
        shutil.rmtree(chroma_folder)
        logger.info(f"Deleted Chroma folder at {chroma_folder}")

    logger.info(
        f"All JSON, LOG and TXT files have been moved to {target_dir}"
    )


# Call the function
cleanup_json_logs("sequential_workflow_agents")
