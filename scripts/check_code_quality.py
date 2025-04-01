#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path

def run_command(command: list[str], cwd: Path) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {' '.join(command)}:")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return False

def main():
    """Run all code quality checks."""
    root_dir = Path(__file__).parent.parent
    success = True

    # Run flake8
    print("\nRunning flake8...")
    if not run_command(["flake8", "swarms", "tests"], root_dir):
        success = False

    # Run pyupgrade
    print("\nRunning pyupgrade...")
    if not run_command(["pyupgrade", "--py39-plus", "swarms", "tests"], root_dir):
        success = False

    # Run black
    print("\nRunning black...")
    if not run_command(["black", "--check", "swarms", "tests"], root_dir):
        success = False

    # Run ruff
    print("\nRunning ruff...")
    if not run_command(["ruff", "check", "swarms", "tests"], root_dir):
        success = False

    if not success:
        print("\nCode quality checks failed. Please fix the issues and try again.")
        sys.exit(1)
    else:
        print("\nAll code quality checks passed!")

if __name__ == "__main__":
    main() 