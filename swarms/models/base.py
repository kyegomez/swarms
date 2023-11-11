import time
from abc import ABC, abstractmethod


def count_tokens(text: str) -> int:
    return len(text.split())


class AbstractModel(ABC):
    """
    AbstractModel

    """

    # abstract base class for language models
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.temperature = 1.0
        self.max_tokens = None
        self.history = ""

    @abstractmethod
    def run(self, task: str) -> str:
        """generate text using language model"""
        pass

    def chat(self, task: str, history: str = "") -> str:
        """Chat with the model"""
        complete_task = task + " | " + history  # Delimiter for clarity
        return self.run(complete_task)

    def __call__(self, task: str) -> str:
        """Call the model"""
        return self.run(task)

    def _sec_to_first_token(self) -> float:
        # Assuming the first token appears instantly after the model starts
        return 0.001

    def _tokens_per_second(self) -> float:
        """Tokens per second"""
        elapsed_time = self.end_time - self.start_time
        if elapsed_time == 0:
            return float("inf")
        return self._num_tokens() / elapsed_time

    def _num_tokens(self, text: str) -> int:
        """Number of tokens"""
        return count_tokens(text)

    def _time_for_generation(self, task: str) -> float:
        """Time for Generation"""
        self.start_time = time.time()
        self.run(task)
        self.end_time = time.time()
        return self.end_time - self.start_time

    @abstractmethod
    def generate_summary(self, text: str) -> str:
        """Generate Summary"""
        pass

    def set_temperature(self, value: float):
        """Set Temperature"""
        self.temperature = value

    def set_max_tokens(self, value: int):
        """Set new max tokens"""
        self.max_tokens = value

    def clear_history(self):
        """Clear history"""
        self.history = ""

    def get_generation_time(self) -> float:
        """Get generation time"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0

    def metrics(self) -> str:
        _sec_to_first_token = self._sec_to_first_token()
        _tokens_per_second = self._tokens_per_second()
        _num_tokens = self._num_tokens(self.history)
        _time_for_generation = self._time_for_generation(self.history)

        return f"""
        SEC TO FIRST TOKEN: {_sec_to_first_token}
        TOKENS/SEC: {_tokens_per_second}
        TOKENS: {_num_tokens}
        Tokens/SEC: {_time_for_generation}
        """
