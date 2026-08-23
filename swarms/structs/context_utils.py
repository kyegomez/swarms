"""
Shared context handling for multi-agent structures.

Every structure that runs several agents against one shared conversation hits
the same two problems, and each used to solve them separately - or not at all.

**Re-sending what an agent already has.** Handing an agent the whole shared
conversation on each invocation puts that history into the agent's own memory,
so the next invocation sends it again on top of what the agent already holds.
Context then grows exponentially across loops, and the agent sees its own
output twice - the second time mislabelled as something the user said.
:func:`new_context_for` sends only what is new.

**Recording a transcript instead of an answer.** ``Agent.run`` honours the
agent's ``output_type``, which defaults to ``"str-all-except-first"`` - the
agent's *entire* conversation, not its answer. Writing that into the shared
conversation re-injects everything the agent was given, which every later agent
then reads. :func:`agent_answer` takes the final message instead.

Both functions are deliberately stateless: the caller owns the cursor dict, so
a structure can reset it per task without this module tracking anything.
"""

from typing import Any, Dict, List, Optional

NO_NEW_MESSAGES = (
    "No new messages since your last turn. Continue from your own "
    "previous response."
)


def new_context_for(
    agent_name: str,
    conversation: Any,
    delivered: Dict[str, int],
    empty_message: str = NO_NEW_MESSAGES,
) -> str:
    """
    The part of a shared conversation an agent has not been given yet.

    Two things are excluded:

    * anything delivered to this agent before - it is already in the agent's
      own memory, and re-sending it is what makes context grow exponentially;
    * the agent's own messages - it holds those as its own assistant turns, so
      echoing them back arrives mislabelled as the user's words.

    Args:
        agent_name: The agent about to run. Also the role its own messages
            carry in the shared conversation.
        conversation: A :class:`~swarms.structs.conversation.Conversation`.
        delivered: Maps agent name to how many messages it has received.
            Mutated in place. Reset it per task, or a reused structure will
            tell the next task's agents there is nothing new.
        empty_message: Returned when everything new came from this agent
            itself. An empty string would read as "no instruction".

    Returns:
        The new messages, rendered as ``"role: content"`` lines.

    Example:
        >>> delivered = {}
        >>> new_context_for("Writer", conversation, delivered)
        'User: draft it\\n\\nEditor: needs a stronger opening'
    """
    history = (
        getattr(conversation, "conversation_history", None) or []
    )
    start = delivered.get(agent_name, 0)
    fresh = history[start:]
    delivered[agent_name] = len(history)

    lines: List[str] = []
    for message in fresh:
        if not isinstance(message, dict):
            continue
        if message.get("role") == agent_name:
            continue
        timestamp = message.get("timestamp")
        prefix = f"[{timestamp}] " if timestamp else ""
        lines.append(
            f"{prefix}{message.get('role')}: {message.get('content')}"
        )

    return "\n\n".join(lines) if lines else empty_message


def agent_answer(agent: Any, fallback: Any = None) -> Optional[str]:
    """
    An agent's final answer, rather than whatever ``run`` chose to return.

    ``Agent.run`` returns according to ``output_type``, which by default is the
    agent's whole conversation. That is the right thing for a caller reading
    the result and the wrong thing to record as the agent's contribution to a
    shared conversation.

    Args:
        agent: Anything agent-like. Structures also nest other structures and
            test doubles here, so a missing or non-string result falls back
            rather than raising.
        fallback: Returned when no final message is available.

    Returns:
        The agent's last message, or ``fallback``.
    """
    try:
        answer = agent.short_memory.get_final_message_content()
    except Exception:
        return fallback

    if isinstance(answer, str) and answer.strip():
        return answer
    return fallback


def get_final_agent_answer(
    agents: List[Any], agent_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Replace each agent's raw ``run`` output with its final answer.

    The batch form of :func:`agent_answer`, for structures that collect a
    whole layer of workers at once. Anything without a usable final message
    keeps whatever ``run`` returned.

    Args:
        agents: The agents that produced ``agent_outputs``.
        agent_outputs: Maps agent name to the value ``run`` returned.
            Not mutated; a new mapping is returned.

    Returns:
        The same mapping with transcripts replaced by answers.
    """
    answers = dict(agent_outputs)
    for agent in agents:
        name = getattr(agent, "agent_name", None)
        if name in answers:
            answers[name] = agent_answer(
                agent, fallback=answers[name]
            )
    return answers
