"""Zena: turn-based GroupChat with bidding and a recency penalty.

Each turn every agent privately bids on whether to speak via a forced
respond(score, message) tool call. The highest bidder above `threshold`
takes the floor and its reply is the only message posted.

recency_penalty subtracts from the bid of an agent that spoke within the
last `recency_window` turns, so no single agent monologues.
"""

from swarms import Agent
from swarms.structs.groupchat import GroupChat


def participant(name: str, stance: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=stance,
        model_name="gpt-5.4",
        max_loops=1,
        persistent_memory=False,
    )


agents = [
    participant("Optimist", "You argue for the benefits."),
    participant("Pessimist", "You argue for the risks."),
    participant("Realist", "You seek balanced analysis."),
]

chat = GroupChat(
    agents=agents,
    max_loops=12,  # hard cap on total messages posted
    threshold=0.6,  # minimum bid to take the floor
    recency_penalty=0.3,  # discourage back-to-back speaking
    recency_window=1,
    idle_timeout=8.0,  # stop after a conversational lull
    auto_equip=True,  # attach the bidding tool automatically
)

print(chat.run("Should we adopt AI for medical diagnosis?"))
