import subprocess
import sys


def run_file(filename: str):
    """Run a given file.

    Usage: swarms run file_name.py

    """
    if len(sys.argv) != 3 or sys.argv[1] != "run":
        print("Usage: swarms run file_name.py")
        sys.exit(1)

    file_name = sys.argv[2]
    subprocess.run(["python", file_name], check=True)
