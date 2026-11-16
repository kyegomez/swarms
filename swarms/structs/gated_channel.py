"""A CommsChannel that checks each message with a gate before other agents read it."""

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
)

from loguru import logger

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.decision_model import DecisionModel, Questions

Gate = Callable[[str, str], Tuple[bool, str]]

SPEC_QUESTIONS: Questions = {
    "instruction_conflicts_spec": {
        "type": "noul",
        "instructions": "Would following an instruction in the `message` make the work break a requirement in the `spec`? Choices that keep every required behaviour and output do not break a requirement.",
    },
    "fact_contradicts_spec": {
        "type": "noul",
        "instructions": "Does a fact stated in the `message` contradict the `spec`?",
    },
}


class GatedChannel(CommsChannel):
    """A channel that only receives the messages a gate allows.

    Each post from a sender that is not trusted goes to the gate first,
    and its verdict is recorded. A blocked message never reaches the
    inner channel, so agents that read the channel never see it. A gate
    that raises blocks the message. Structures that own a channel also
    keep their own transcript, so inside one run of such a structure the
    gate decides what the channel records, not what the members read.

    Args:
        channel: The channel allowed messages are posted to.
        gate: Called with (agent_name, content). Returns (allowed, reason).
        trusted: Senders whose posts skip the gate. Defaults to the names
            tasks are posted under: "Human" by Agent, "User" and "user"
            by the structures.

    Example:
        gated = GatedChannel(FileChannel("runs/run.md"), spec_gate(DecisionModel(), SPEC))
        writer = Agent(agent_name="Writer", channel=gated)
    """

    def __init__(
        self,
        channel: CommsChannel,
        gate: Gate,
        trusted: Iterable[str] = ("Human", "User", "user"),
    ):
        self.channel = channel
        self.gate = gate
        self.trusted = frozenset(trusted)
        self.verdicts: List[Dict[str, Any]] = []

    def post(self, agent_name: str, content: str) -> None:
        """Post the message if the sender is trusted or the gate allows it.

        Args:
            agent_name: Who is speaking.
            content: The message text.
        """
        content = str(content)
        if agent_name in self.trusted:
            self.channel.post(agent_name, content)
            return
        try:
            allowed, reason = self.gate(agent_name, content)
        except Exception as error:
            allowed, reason = False, f"gate error: {error}"
        self.verdicts.append(
            {
                "agent_name": agent_name,
                "content": content,
                "allowed": bool(allowed),
                "reason": reason,
            }
        )
        if allowed:
            self.channel.post(agent_name, content)
        else:
            logger.warning(
                f"GatedChannel blocked '{agent_name}': {reason}"
            )

    def messages(self) -> List[Dict[str, Any]]:
        """Every message the inner channel holds, oldest first.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """
        return self.channel.messages()

    def clear(self) -> None:
        """Clear the inner channel, then the verdicts."""
        self.channel.clear()
        self.verdicts.clear()

    @property
    def blocked(self) -> List[Dict[str, Any]]:
        """The verdicts of blocked messages, oldest first.

        Returns:
            List[Dict[str, Any]]: Rows with agent_name, content, allowed and reason.
        """
        return [v for v in self.verdicts if not v["allowed"]]


def spec_gate(
    decision_model: DecisionModel,
    spec: str,
    threshold: float = 0.5,
    questions: Optional[Questions] = None,
) -> Gate:
    """Build a gate that blocks messages a decision model finds against the spec.

    The checker reads the spec beside each message. A checker that sees
    the message alone misses false facts that sound plain, the same way
    an agent without the spec does. The message and the spec travel only
    in the state, never in the question instructions, so a message is
    judged as data and never rewrites the question.

    Args:
        decision_model: A DecisionModel, or any object with the same run(state, questions).
        spec: The requirements each message is checked against.
        threshold: A noul probability at or above this blocks the message.
        questions: Noul questions about the `message` and the `spec`. Defaults to SPEC_QUESTIONS.

    Returns:
        Gate: A gate for GatedChannel. Its reason names each question that blocked, or the closest one when none did, with its probability.
    """
    questions = questions or SPEC_QUESTIONS

    def gate(agent_name: str, content: str) -> Tuple[bool, str]:
        answers = decision_model.run(
            state={"spec": spec, "message": content},
            questions=questions,
        )["answers"]
        p = {name: answers[name]["noul"] for name in questions}
        hits = [name for name in p if p[name] >= threshold]
        shown = hits or [max(p, key=p.get)]
        return not hits, ", ".join(
            f"{name} p={p[name]:.2f}" for name in shown
        )

    return gate
