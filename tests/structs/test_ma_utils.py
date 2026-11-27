from typing import Any, Callable, List
from unittest.mock import MagicMock

import pytest

from swarms import (
    Agent,
    GroupChat,
    HierarchicalSwarm,
    LLMCouncil,
    MixtureOfAgents,
    MultiAgentRouter,
)


@pytest.mark.parametrize(
    "build, repeat_ok",
    [
        (lambda a: MultiAgentRouter(agents=a), True),
        (lambda a: HierarchicalSwarm(agents=a), True),
        (lambda a: LLMCouncil(council_members=a), False),
        (lambda a: GroupChat(agents=a, channel=MagicMock()), False),
        (lambda a: MixtureOfAgents(agents=a), False),
    ],
)
def test_agents_that_share_a_name_are_rejected(
    build: Callable[[List[Agent]], Any], repeat_ok: bool
) -> None:
    """Two agents with one name raise; one listed twice may be allowed."""
    agent = Agent(agent_name="Twin", print_on=False)
    agent.channel = own = MagicMock()
    with pytest.raises(ValueError, match="'Twin'"):
        build([agent, Agent(agent_name="Twin", print_on=False)])
    assert agent.channel is own
    if repeat_ok:
        build([agent, agent])
    else:
        with pytest.raises(ValueError, match="'Twin'"):
            build([agent, agent])
