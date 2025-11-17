import os
import subprocess
import logging
import time
import psutil

# Configure logging
logging.basicConfig(
    filename="test_runner.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def run_tests_in_subfolders(
    base_folders: list,
    file_extension=".py",
    python_interpreter="python",
):
    report_file = "test_report.txt"

    with open(report_file, "w") as report:
        for base_folder in base_folders:
            if not os.path.exists(base_folder):
                logging.warning(
                    f"Base folder does not exist: {base_folder}"
                )
                continue

            for root, dirs, files in os.walk(base_folder):
                for file in files:
                    if file.endswith(file_extension):
                        file_path = os.path.join(root, file)
                        try:
                            logging.info(f"Running {file_path}...")

                            # Start time measurement
                            start_time = time.time()

                            # Get initial memory usage
                            process = psutil.Process(os.getpid())
                            initial_memory = (
                                process.memory_info().rss
                            )  # Resident Set Size

                            result = subprocess.run(
                                [python_interpreter, file_path],
                                capture_output=True,
                                text=True,
                            )

                            # End time measurement
                            end_time = time.time()

                            # Get final memory usage
                            final_memory = process.memory_info().rss

                            # Calculate metrics
                            execution_time = end_time - start_time
                            memory_used = (
                                final_memory - initial_memory
                            )

                            report.write(f"Running {file_path}:\n")
                            report.write(result.stdout)
                            report.write(result.stderr)
                            report.write(
                                f"\nExecution Time: {execution_time:.2f} seconds\n"
                            )
                            report.write(
                                f"Memory Used: {memory_used / (1024 ** 2):.2f} MB\n"
                            )  # Convert to MB
                            report.write("\n" + "-" * 40 + "\n")

                            logging.info(
                                f"Completed {file_path} with return code {result.returncode}"
                            )
                            logging.info(
                                f"Execution Time: {execution_time:.2f} seconds, Memory Used: {memory_used / (1024 ** 2):.2f} MB"
                            )

                        except FileNotFoundError:
                            logging.error(
                                f"File not found: {file_path}"
                            )
                            report.write(
                                f"File not found: {file_path}\n"
                            )
                        except Exception as e:
                            logging.error(
                                f"Error running {file_path}: {e}"
                            )
                            report.write(
                                f"Error running {file_path}: {e}\n"
                            )


# Example usage
base_folders = [
    "folder1",
    "folder2",
]  # Replace with your actual folder names
file_extension = ".py"  # Specify the file extension to run
python_interpreter = "python"  # Specify the Python interpreter to use

run_tests_in_subfolders(
    base_folders, file_extension, python_interpreter
)
