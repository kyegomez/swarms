import os


def parse_log_file(filename: str):
    """
    Parse a log file and return a list of log entries.

    Each log entry is a dictionary with keys for the timestamp, name, level, and message.

    Args:
        filename (str): The name of the log file.

    Returns:
        list: A list of log entries.

    Raises:
        FileNotFoundError: If the log file does not exist.
        ValueError: If a log entry does not have the correct format.
    """
    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file {filename} does not exist.")

    log_entries = []

    with open(filename) as file:
        for line in file:
            parts = line.split(" - ")
            # Check if the log entry has the correct format
            if len(parts) != 4:
                raise ValueError(
                    f"The log entry '{line}' does not have the"
                    " correct format."
                )
            timestamp, name, level, message = parts
            log_entry = {
                "timestamp": timestamp,
                "name": name,
                "level": level,
                "message": message.rstrip("\n"),
            }
            log_entries.append(log_entry)

    return log_entries
