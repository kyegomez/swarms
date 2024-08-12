import gzip
import json
import os
from typing import Dict, Iterable

ROOT = os.path.dirname(os.path.abspath(__file__))


def read_problems_from_jsonl(filename: str) -> Iterable[Dict]:
    return {task["task_id"]: task for task in stream_jsonl(filename)}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Stream JSONL data from a file.

    Args:
        filename (str): The path to the JSONL file.

    Yields:
        Dict: A dictionary representing each JSON object in the file.
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)

    else:
        with open(filename) as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        filename (str): The path to the output file.
        data (Iterable[Dict]): The data to be written to the file.
        append (bool, optional): If True, append to an existing file.
            If False, overwrite the file. Defaults to False.
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write(json.dumps(x) + "\n").encode("utf-8")
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))
