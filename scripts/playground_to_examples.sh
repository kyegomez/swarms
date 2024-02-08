#!/bin/bash

# Define the directory to search
dir="playground"

# Check if the directory exists
if [ -d "$dir" ]
then
    # Use find to locate all .py files in the directory and its subdirectories
    for file in $(find $dir -name "*.py")
    do
        # Extract the file name and directory
        base=$(basename $file .py)
        dir=$(dirname $file)

        # Check if the file name already contains _example
        if [[ $base == *_example ]]
        then
            echo "Skipping $file as it already contains _example"
            continue
        fi

        # Append _example to the file name
        newname="${base}_example.py"

        # Rename the file
        mv $file $dir/$newname

        echo "Renamed $file to $dir/$newname"
    done
else
    echo "Directory $dir does not exist."
fi