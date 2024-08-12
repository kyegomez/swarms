#!/bin/bash

# Create the new directory if it doesn't exist
sudo mkdir -p /artifacts_logs

# Find all .log files in the root directory and its subdirectories
find / -name "*.log" -print0 | while IFS= read -r -d '' file; do
    # Use sudo to move the file to the new directory
    sudo mv "$file" /artifacts_logs/
done