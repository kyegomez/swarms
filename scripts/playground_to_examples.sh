#!/bin/bash

# Change to the root directory
cd /

# Iterate over all the .py files in the directory
for file in *.py; do
    # Get the base name of the file without the .py
    base_name=$(basename "$file" .py)

    # Rename the file to remove .py from the end
    mv "$file" "${base_name}"
done