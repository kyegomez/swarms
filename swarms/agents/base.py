from abc import abstractmethod
from typing import Dict, List, Union, Optional


class AbstractAgent:
    """(In preview) An abstract class for AI agent.

    An agent can communicate with other agents and perform actions.
    Different agents can differ in what actions they perform in the `receive` method.

    Agents are full and completed:

    Agents = llm + tools + memory


    """

    def __init__(self, name: str, *args, **kwargs):
        """
        Args:
            name (str): name of the agent.
        """
        # a dictionary of conversations, default value is list
        self._name = name

    @property
    def name(self):
        """Get the name of the agent."""
        return self._name

    def tools(self, tools):
        """init tools"""

    def memory(self, memory_store):
        """init memory"""

    def reset(self):
        """(Abstract method) Reset the agent."""

    @abstractmethod
    def run(self, task: str, *args, **kwargs):
        """Run the agent once"""

    def _arun(self, taks: str):
        """Run Async run"""

    def chat(self, messages: List[Dict]):
        """Chat with the agent"""

    def _achat(self, messages: List[Dict]):
        """Asynchronous Chat"""

    def step(self, message: str):
        """Step through the agent"""

    def _astep(self, message: str):
        """Asynchronous step"""

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
