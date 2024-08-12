import logging
from abc import ABC
from dataclasses import dataclass

from tenacity import Retrying, stop_after_attempt, wait_exponential


@dataclass
class ExponentialBackoffMixin(ABC):
    """
    A mixin class that provides exponential backoff functionality.
    """

    min_retry_delay: float = 2
    """
    The minimum delay between retries in seconds.
    """

    max_retry_delay: float = 10
    """
    The maximum delay between retries in seconds.
    """

    max_attempts: int = 10
    """
    The maximum number of retry attempts.
    """

    def after_hook(s: str) -> None:
        return logging.warning(s)

    """
    A callable that is executed after each retry attempt.
    """

    def retrying(self) -> Retrying:
        """
        Returns a Retrying object configured with the exponential backoff settings.
        """
        return Retrying(
            wait=wait_exponential(
                min=self.min_retry_delay, max=self.max_retry_delay
            ),
            stop=stop_after_attempt(self.max_attempts),
            reraise=True,
            after=self.after_hook,
        )
