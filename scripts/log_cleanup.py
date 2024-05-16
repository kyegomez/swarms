import os
import shutil

# Create a new directory for the log files if it doesn't exist
if not os.path.exists("artifacts_five"):
    os.makedirs("artifacts_five")

# Walk through the current directory
for dirpath, dirnames, filenames in os.walk("."):
    for filename in filenames:
        # If the file is a log file
        if filename.endswith(".log"):
            # Construct the full file path
            file_path = os.path.join(dirpath, filename)
            # Move the log file to the 'artifacts_five' directory
            shutil.move(file_path, "artifacts_five")

print(
    "Moved all log files into the 'artifacts_five' directory and"
    " deleted their original location."
)
