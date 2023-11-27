from typing import Dict, List, Optional, Union


class AbstractWorker:
    """(In preview) An abstract class for AI worker.

    An worker can communicate with other workers and perform actions.
    Different workers can differ in what actions they perform in the `receive` method.
    """

    def __init__(
        self,
        name: str,
    ):
        """
        Args:
            name (str): name of the worker.
        """
        # a dictionary of conversations, default value is list
        self._name = name

    @property
    def name(self):
        """Get the name of the worker."""
        return self._name

    def run(self, task: str):
        """Run the worker agent once"""

    def send(
        self,
        message: Union[Dict, str],
        recipient,  # add AbstractWorker
        request_reply: Optional[bool] = None,
    ):
        """(Abstract method) Send a message to another worker."""

    async def a_send(
        self,
        message: Union[Dict, str],
        recipient,  # add AbstractWorker
        request_reply: Optional[bool] = None,
    ):
        """(Aabstract async method) Send a message to another worker."""

    def receive(
        self,
        message: Union[Dict, str],
        sender,  # add AbstractWorker
        request_reply: Optional[bool] = None,
    ):
        """(Abstract method) Receive a message from another worker."""

    async def a_receive(
        self,
        message: Union[Dict, str],
        sender,  # add AbstractWorker
        request_reply: Optional[bool] = None,
    ):
        """(Abstract async method) Receive a message from another worker."""

    def reset(self):
        """(Abstract method) Reset the worker."""

    def generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender=None,  # Optional["AbstractWorker"] = None,
        **kwargs,
    ) -> Union[str, Dict, None]:
        """(Abstract method) Generate a reply based on the received messages.

        Args:
            messages (list[dict]): a list of messages received.
            sender: sender of an Agent instance.
        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """

    async def a_generate_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender=None,  # Optional["AbstractWorker"] = None,
        **kwargs,
    ) -> Union[str, Dict, None]:
        """(Abstract async method) Generate a reply based on the received messages.

        Args:
            messages (list[dict]): a list of messages received.
            sender: sender of an Agent instance.
        Returns:
            str or dict or None: the generated reply. If None, no reply is generated.
        """
