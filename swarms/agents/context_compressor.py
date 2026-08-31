"""
Context window compression for autonomous (``max_loops="auto"``) agent runs.

When an agent's conversation history approaches its context window limit,
this module summarizes the accumulated history and replaces it with a dense
summary, preserving the system prompt and keeping the agent within its token
budget for an unbounded run.
"""

from typing import Any, Optional

from litellm import completion
from loguru import logger

from swarms.utils.litellm_tokenizer import count_tokens


COMPRESSION_SYSTEM_PROMPT = """
You are a conversation compression expert. Your job is to produce a faithful, dense summary of an ongoing agent conversation so that the agent can continue its work without losing critical context.

Requirements:
- Preserve every concrete fact, decision, constraint, tool result, and open question.
- Preserve the original task and any user-provided instructions.
- Preserve the latest intermediate conclusions the agent has reached.
- Preserve any error messages, failed attempts, and retry context.
- Drop conversational filler, redundant restatements, and verbose chain-of-thought.
- Output ONLY the summary body. No preamble. No meta commentary.
"""


COMPRESSION_USER_TEMPLATE = """Compress the following agent conversation into a dense summary that the agent can use as its new working memory:

<conversation>
{history}
</conversation>

Produce the compressed memory now."""


class ContextCompressor:
    """
    Monitors an agent's short-term memory token usage during
    ``max_loops="auto"`` runs and compresses the history into a summary when
    usage crosses a configurable fraction of the context window.

    The agent's ``system_prompt`` is preserved because it is stored on the
    ``Conversation`` object outside of ``conversation_history``; only the
    turn-by-turn transcript is collapsed into a summary message.
    """

    def __init__(
        self,
        threshold: float = 0.9,
        summarizer_model: Optional[str] = None,
        summarizer_temperature: float = 0.2,
        summarizer_max_tokens: int = 4000,
    ):
        """
        Args:
            threshold (float): Fraction of ``context_length`` at which to
                trigger compression (e.g. ``0.9`` fires at 90% full).
            summarizer_model (Optional[str]): Model used to produce the
                summary. If ``None``, the agent's own ``model_name`` is used.
            summarizer_temperature (float): Temperature for the summary call.
            summarizer_max_tokens (int): Max tokens for the summary output.
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError(
                f"threshold must be in (0, 1], got {threshold}"
            )
        self.threshold = threshold
        self.summarizer_model = summarizer_model
        self.summarizer_temperature = summarizer_temperature
        self.summarizer_max_tokens = summarizer_max_tokens

    def usage_ratio(self, agent: Any) -> float:
        """Return current token usage as a fraction of the context window."""
        context_length = getattr(agent, "context_length", None)
        if not context_length:
            return 0.0
        history = agent.short_memory.return_history_as_string()
        return count_tokens(history) / float(context_length)

    def should_compress(self, agent: Any) -> bool:
        """True once usage has crossed the threshold.

        Loop-mode gating (auto vs integer) is now handled by the agent's
        ``context_compression`` flag at construction; this method just
        measures the token budget.
        """
        return self.usage_ratio(agent) >= self.threshold

    def _summarize(self, agent: Any, history: str) -> str:
        model = self.summarizer_model or getattr(
            agent, "model_name", None
        )
        if not model:
            raise ValueError(
                "No summarizer_model configured and agent has no model_name"
            )
        response = completion(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": COMPRESSION_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": COMPRESSION_USER_TEMPLATE.format(
                        history=history
                    ),
                },
            ],
            temperature=self.summarizer_temperature,
            max_tokens=self.summarizer_max_tokens,
        )
        return response.choices[0].message.content

    def compress(self, agent: Any) -> Optional[str]:
        """
        Summarize the full conversation history and replace it in the
        agent's ``short_memory``. Returns the summary string, or ``None`` if
        there is nothing to compress.
        """
        history = agent.short_memory.return_history_as_string()
        if not history.strip():
            return None

        prior_tokens = count_tokens(history)
        agent_name = getattr(agent, "agent_name", "agent")
        logger.info(
            f"[ContextCompressor] Triggering compression for "
            f"'{agent_name}' at {prior_tokens} / "
            f"{agent.context_length} tokens"
        )

        summary = self._summarize(agent, history)

        # Conversation.compact() also wipes and re-seeds MEMORY.md, which would otherwise grow unbounded.
        summary_content = (
            "[Compressed Memory Summary]\n"
            "The following is a compressed summary of earlier "
            "conversation turns, replacing the raw transcript to "
            "stay within the context window.\n\n"
            f"{summary}"
        )
        agent.short_memory.compact(summary=summary_content)

        new_tokens = count_tokens(
            agent.short_memory.return_history_as_string()
        )
        logger.info(
            f"[ContextCompressor] Compressed {prior_tokens} -> "
            f"{new_tokens} tokens for '{agent_name}'"
        )
        return summary

    def maybe_compress(self, agent: Any) -> Optional[str]:
        """Compress only if conditions are met. Returns the summary or None."""
        if not self.should_compress(agent):
            return None
        try:
            return self.compress(agent)
        except Exception as e:
            logger.error(
                f"[ContextCompressor] Compression failed: {e}. "
                f"Continuing without compression."
            )
            return None
