import threading
from typing import Callable, Tuple


class AgentJob(threading.Thread):
    """A class that handles multithreading logic.

    Args:
        function (Callable): The function to be executed in a separate thread.
        args (Tuple): The arguments to be passed to the function.
    """

    def __init__(self, function: Callable, args: Tuple):
        threading.Thread.__init__(self)
        self.function = function
        self.args = args

    def run(self) -> None:
        """Runs the function in a separate thread."""
        self.function(*self.args)
