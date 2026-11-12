"""Channels that agents read before a run and post their answer to after it."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger


class CommsChannel(ABC):
    """A place agents read what others said and post what they say.

    Subclass it to exchange messages through any medium. An agent given a
    channel reads it as prior turns before a run, posts the task once, and
    posts its final answer after a successful run.
    """

    @abstractmethod
    def post(self, agent_name: str, content: str) -> None:
        """Append one message.

        Args:
            agent_name: Who is speaking, such as an agent name or the user name.
            content: The message text.
        """

    @abstractmethod
    def messages(self) -> List[Dict[str, Any]]:
        """Every message so far, oldest first.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """

    def clear(self) -> None:
        """Remove every message. Optional, for channels that can."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot be cleared"
        )


def claim_agents(
    agents: Optional[List[Any]],
    channel: Optional[CommsChannel],
    owner: str,
) -> None:
    """Detach member agents' own channels so only the owner posts.

    Args:
        agents: The structure's members. Anything without a channel is skipped.
        channel: The owner's channel. Nothing happens when it is None.
        owner: The structure's name, for the warning.
    """
    if channel is None:
        return
    for agent in agents or []:
        if getattr(agent, "channel", None) is not None:
            logger.warning(
                f"{owner} owns the comms channel; ignoring the one on "
                f"agent '{getattr(agent, 'agent_name', agent)}'"
            )
            agent.channel = None
