import pathlib
from typing import List, Tuple


def rename_test_files(
    tests_dir: str = "tests", dry_run: bool = True
) -> List[Tuple[str, str]]:
    """
    Recursively finds all Python files in the tests directory and adds 'test_'
    prefix to their names if they don't already contain 'test_' in the filename.

    Args:
        tests_dir (str): Path to the tests directory. Defaults to "tests".
        dry_run (bool): If True, only shows what would be renamed without actually renaming.
                       If False, performs the actual renaming. Defaults to True.

    Returns:
        List[Tuple[str, str]]: List of tuples containing (old_path, new_path) for each renamed file.
    """
    renamed_files = []
    tests_path = pathlib.Path(tests_dir)

    if not tests_path.exists():
        print(f"Error: Tests directory '{tests_dir}' does not exist.")
        return renamed_files

    # Find all Python files recursively
    python_files = list(tests_path.rglob("*.py"))

    print(
        f"Found {len(python_files)} Python files in '{tests_dir}' directory"
    )
    print("=" * 60)

    for file_path in python_files:
        filename = file_path.name

        # Skip files that already have 'test_' in their name
        if "test_" in filename.lower():
            print(f"✓ SKIP: {file_path} (already contains 'test_')")
            continue

        # Skip __init__.py files
        if filename == "__init__.py":
            print(f"✓ SKIP: {file_path} (__init__.py file)")
            continue

        # Create new filename with 'test_' prefix
        new_filename = f"test_{filename}"
        new_path = file_path.parent / new_filename

        # Check if target filename already exists
        if new_path.exists():
            print(
                f"⚠️  WARNING: {new_path} already exists, skipping {file_path}"
            )
            continue

        if dry_run:
            print(
                f"🔍 DRY RUN: Would rename {file_path} -> {new_path}"
            )
        else:
            try:
                file_path.rename(new_path)
                print(f"✅ RENAMED: {file_path} -> {new_path}")
            except OSError as e:
                print(f"❌ ERROR: Failed to rename {file_path}: {e}")
                continue

        renamed_files.append((str(file_path), str(new_path)))

    print("=" * 60)
    print(
        f"Summary: {len(renamed_files)} files {'would be' if dry_run else 'were'} renamed"
    )

    return renamed_files


def main():
    """
    Main function to demonstrate the rename functionality.
    """
    print("Python Test File Renamer")
    print("=" * 60)

    # First run in dry-run mode to see what would be changed
    print("Running in DRY RUN mode first...")
    print()

    renamed_files = rename_test_files(tests_dir="tests", dry_run=True)

    if not renamed_files:
        print("No files need to be renamed.")
        return

    print("\nFiles that would be renamed:")
    for old_path, new_path in renamed_files:
        print(f"  {old_path} -> {new_path}")

    # Ask user if they want to proceed with actual renaming
    print(f"\nFound {len(renamed_files)} files that need renaming.")
    user_input = (
        input(
            "Do you want to proceed with the actual renaming? (y/N): "
        )
        .strip()
        .lower()
    )

    if user_input in ["y", "yes"]:
        print("\nProceeding with actual renaming...")
        rename_test_files(tests_dir="tests", dry_run=False)
    else:
        print("Renaming cancelled.")


if __name__ == "__main__":
    main()
