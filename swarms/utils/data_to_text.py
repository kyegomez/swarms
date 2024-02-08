import os
import csv
import json
from swarms.utils.pdf_to_text import pdf_to_text


def csv_to_text(file):
    with open(file, "r") as file:
        reader = csv.reader(file)
        data = list(reader)
    return str(data)


def json_to_text(file):
    with open(file, "r") as file:
        data = json.load(file)
    return json.dumps(data)


def txt_to_text(file):
    with open(file, "r") as file:
        data = file.read()
    return data


def data_to_text(file):
    """
    Converts the given data file to text format.

    Args:
        file (str): The path to the data file.

    Returns:
        str: The text representation of the data file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, ext = os.path.splitext(file)
    if ext == ".csv":
        return csv_to_text(file)
    elif ext == ".json":
        return json_to_text(file)
    elif ext == ".txt":
        return txt_to_text(file)
    elif ext == ".pdf":
        return pdf_to_text(file)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
