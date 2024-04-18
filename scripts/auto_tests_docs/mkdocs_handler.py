import os


def generate_file_list(directory, output_file):
    """
    Generate a list of files in a directory in the specified format and write it to a file.

    Args:
    directory (str): The directory to list the files from.
    output_file (str): The file to write the output to.
    """
    with open(output_file, "w") as f:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".md"):
                    # Remove the directory from the file path and replace slashes with dots
                    file_path = (
                        os.path.join(root, file)
                        .replace(directory + "/", "")
                        .replace("/", ".")
                    )
                    # Remove the file extension
                    file_name, _ = os.path.splitext(file)
                    # Write the file name and path to the output file
                    f.write(f'- {file_name}: "swarms/utils/{file_path}"\n')


# Use the function to generate the file list
generate_file_list("docs/swarms/structs", "file_list.txt")
