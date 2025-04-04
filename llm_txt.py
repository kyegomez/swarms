import os
from pathlib import Path


def concat_all_md_files(root_dir, output_filename="llm.txt"):
    """
    Recursively searches for all .md files in directory and subdirectories,
    then concatenates them into a single output file.

    Args:
        root_dir (str): Root directory to search for .md files
        output_filename (str): Name of output file (default: llm.txt)

    Returns:
        str: Path to the created output file
    """
    try:
        root_dir = Path(root_dir).resolve()
        if not root_dir.is_dir():
            raise ValueError(f"Directory not found: {root_dir}")

        # Collect all .md files recursively
        md_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(".md"):
                    full_path = Path(root) / file
                    md_files.append(full_path)

        if not md_files:
            print(
                f"No .md files found in {root_dir} or its subdirectories"
            )
            return None

        # Create output file in root directory
        output_path = root_dir / output_filename

        with open(output_path, "w", encoding="utf-8") as outfile:
            for md_file in sorted(md_files):
                try:
                    # Get relative path for header
                    rel_path = md_file.relative_to(root_dir)
                    with open(
                        md_file, "r", encoding="utf-8"
                    ) as infile:
                        content = infile.read()
                        outfile.write(f"# File: {rel_path}\n\n")
                        outfile.write(content)
                        outfile.write(
                            "\n\n" + "-" * 50 + "\n\n"
                        )  # Separator
                except Exception as e:
                    print(f"Error processing {rel_path}: {str(e)}")
                    continue

        print(
            f"Created {output_path} with {len(md_files)} files merged"
        )
        return str(output_path)

    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return None


if __name__ == "__main__":
    concat_all_md_files("docs")
