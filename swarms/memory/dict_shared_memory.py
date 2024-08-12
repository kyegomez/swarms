import datetime
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict


class DictSharedMemory:
    """A class representing a shared memory that stores entries as a dictionary.

    Attributes:
        file_loc (Path): The file location where the memory is stored.
        lock (threading.Lock): A lock used for thread synchronization.

    Methods:
        __init__(self, file_loc: str = None) -> None: Initializes the shared memory.
        add_entry(self, score: float, agent_id: str, agent_cycle: int, entry: Any) -> bool: Adds an entry to the internal memory.
        get_top_n(self, n: int) -> None: Gets the top n entries from the internal memory.
        write_to_file(self, data: Dict[str, Dict[str, Any]]) -> bool: Writes the internal memory to a file.
    """

    def __init__(self, file_loc: str = None) -> None:
        """Initialize the shared memory. In the current architecture the memory always consists of a set of soltuions or evaluations.
        Moreover, the project is designed around LLMs for the proof of concepts, so we treat all entry content as a string.
        """
        if file_loc is not None:
            self.file_loc = Path(file_loc)
            if not self.file_loc.exists():
                self.file_loc.touch()

        self.lock = threading.Lock()

    def add(
        self,
        score: float,
        agent_id: str,
        agent_cycle: int,
        entry: Any,
    ) -> bool:
        """Add an entry to the internal memory."""
        with self.lock:
            entry_id = str(uuid.uuid4())
            data = {}
            epoch = datetime.datetime.utcfromtimestamp(0)
            epoch = (datetime.datetime.utcnow() - epoch).total_seconds()
            data[entry_id] = {
                "agent": agent_id,
                "epoch": epoch,
                "score": score,
                "cycle": agent_cycle,
                "content": entry,
            }
            status = self.write_to_file(data)
            self.plot_performance()
            return status

    def get_top_n(self, n: int) -> None:
        """Get the top n entries from the internal memory."""
        with self.lock:
            with open(self.file_loc) as f:
                try:
                    file_data = json.load(f)
                except Exception as e:
                    file_data = {}
                    raise e

            sorted_data = dict(
                sorted(
                    file_data.items(),
                    key=lambda item: item[1]["score"],
                    reverse=True,
                )
            )
            top_n = dict(list(sorted_data.items())[:n])
            return top_n

    def write_to_file(self, data: Dict[str, Dict[str, Any]]) -> bool:
        """Write the internal memory to a file."""
        if self.file_loc is not None:
            with open(self.file_loc) as f:
                try:
                    file_data = json.load(f)
                except Exception as e:
                    file_data = {}
                    raise e

            file_data = file_data | data
            with open(self.file_loc, "w") as f:
                json.dump(file_data, f, indent=4)

                f.flush()
                os.fsync(f.fileno())

            return True
