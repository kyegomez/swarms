import os
import shutil


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
                or filename.endswith(".txt")
            ):
                # Construct the full file paths
                file_path = os.path.join(dirpath, filename)
                target_path = os.path.join(target_dir, filename)

                # Move the file to the target directory
                shutil.move(file_path, target_path)

    print(f"All JSON, LOG and TXT files have been moved to {target_dir}")


# Call the function
cleanup_json_logs("artifacts_logs")
