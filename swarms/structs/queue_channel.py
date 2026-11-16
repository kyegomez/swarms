"""A CommsChannel whose consumers block on new messages instead of polling."""

import queue
import threading
from typing import Any, Dict, Iterator, List, Optional

from swarms.structs.comms_channel import CommsChannel


class QueueChannel(CommsChannel):
    """Every post goes on a queue, so a consumer can block for the next message.

    Agents read messages() as usual. A consumer thread calls wait() or
    iterates the channel, and close() ends the iteration once the
    producer is done.

    Args:
        q: A queue.Queue, a multiprocessing.Queue, or anything with put,
            get and get_nowait. Defaults to a new queue.Queue. Across
            processes only the queue is shared: messages() holds what this
            process posted.

    Example:
        channel = QueueChannel()
        threading.Thread(target=lambda: (workflow.run(task), channel.close())).start()
        for message in channel:
            print(message["agent_name"], message["content"])
    """

    def __init__(self, q: Optional[Any] = None):
        self.queue = q if q is not None else queue.Queue()
        self._history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def post(self, agent_name: str, content: str) -> None:
        message = {"agent_name": agent_name, "content": str(content)}
        with self._lock:
            self._history.append(message)
        self.queue.put(message)

    def messages(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    def wait(
        self, timeout: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Block until the next message is posted.

        Args:
            timeout: Seconds to wait. None waits forever.

        Returns:
            Optional[Dict[str, Any]]: The message, or None when the channel
                was closed or the timeout passed.
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        while (message := self.wait()) is not None:
            yield message

    def close(self) -> None:
        """Wake the waiting consumer and end its iteration."""
        self.queue.put(None)

    def clear(self) -> None:
        with self._lock:
            self._history.clear()
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return
