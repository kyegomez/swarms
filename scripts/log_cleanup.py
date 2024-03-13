import os
import shutil

# Create a new directory for the log files if it doesn't exist
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")

# Walk through the current directory
for dirpath, dirnames, filenames in os.walk("."):
    for filename in filenames:
        # If the file is a log file
        if filename.endswith(".log"):
            # Construct the full file path
            file_path = os.path.join(dirpath, filename)
            # Move the log file to the 'artifacts' directory
            shutil.move(file_path, "artifacts")

print(
    "Moved all log files into the 'artifacts' directory and deleted"
    " their original location."
)
