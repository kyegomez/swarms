from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from swarms.structs.conversation import Conversation
from swarms.utils.litellm_tokenizer import count_tokens


@dataclass
class TransformConfig:
    """Configuration for message transforms."""

    enabled: bool = False
    method: str = "middle-out"
    max_tokens: Optional[int] = None
    max_messages: Optional[int] = None
    model_name: str = "gpt-4"
    preserve_system_messages: bool = True
    preserve_recent_messages: int = 2


@dataclass
class TransformResult:
    """Result of message transformation."""

    messages: List[Dict[str, Any]]
    original_token_count: int
    compressed_token_count: int
    original_message_count: int
    compressed_message_count: int
    compression_ratio: float
    was_compressed: bool


class MessageTransforms:
    """
    Handles message transformations for context size management.

    Supports middle-out compression which removes or truncates messages
    from the middle of the conversation while preserving the beginning
    and end, which are typically more important for context.
    """

    def __init__(self, config: TransformConfig):
        """
        Initialize the MessageTransforms with configuration.

        Args:
            config: TransformConfig object with transformation settings
        """
        self.config = config

    def transform_messages(
        self,
        messages: List[Dict[str, Any]],
        target_model: Optional[str] = None,
    ) -> TransformResult:
        """
        Transform messages according to the configured strategy.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            target_model: Optional target model name to determine context limits

        Returns:
            TransformResult containing transformed messages and metadata
        """
        if not self.config.enabled or not messages:
            return TransformResult(
                messages=messages,
                original_token_count=self._count_total_tokens(
                    messages
                ),
                compressed_token_count=self._count_total_tokens(
                    messages
                ),
                original_message_count=len(messages),
                compressed_message_count=len(messages),
                compression_ratio=1.0,
                was_compressed=False,
            )

        # Use target model if provided, otherwise use config model
        model_name = target_model or self.config.model_name

        # Get model context limits
        max_tokens = self._get_model_context_limit(model_name)
        max_messages = self._get_model_message_limit(model_name)

        # Override with config values if specified
        if self.config.max_tokens is not None:
            max_tokens = self.config.max_tokens
        if self.config.max_messages is not None:
            max_messages = self.config.max_messages

        original_tokens = self._count_total_tokens(messages)
        original_messages = len(messages)

        transformed_messages = messages.copy()

        # Apply transformations
        if max_messages and len(transformed_messages) > max_messages:
            transformed_messages = self._compress_message_count(
                transformed_messages, max_messages
            )

        if (
            max_tokens
            and self._count_total_tokens(transformed_messages)
            > max_tokens
        ):
            transformed_messages = self._compress_tokens(
                transformed_messages, max_tokens
            )

        compressed_tokens = self._count_total_tokens(
            transformed_messages
        )
        compressed_messages = len(transformed_messages)

        compression_ratio = (
            compressed_tokens / original_tokens
            if original_tokens > 0
            else 1.0
        )

        return TransformResult(
            messages=transformed_messages,
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            original_message_count=original_messages,
            compressed_message_count=compressed_messages,
            compression_ratio=compression_ratio,
            was_compressed=compressed_tokens < original_tokens
            or compressed_messages < original_messages,
        )

    def _compress_message_count(
        self, messages: List[Dict[str, Any]], max_messages: int
    ) -> List[Dict[str, Any]]:
        """
        Compress message count using middle-out strategy.

        Args:
            messages: List of messages to compress
            max_messages: Maximum number of messages to keep

        Returns:
            Compressed list of messages
        """
        if len(messages) <= max_messages:
            return messages

        # Always preserve system messages at the beginning
        system_messages = []
        other_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_messages.append(msg)
            else:
                other_messages.append(msg)

        # Calculate how many non-system messages we can keep
        available_slots = max_messages - len(system_messages)
        if available_slots <= 0:
            # If we can't fit any non-system messages, just return system messages
            return system_messages[:max_messages]

        # Preserve recent messages
        preserve_recent = min(
            self.config.preserve_recent_messages, len(other_messages)
        )
        recent_messages = (
            other_messages[-preserve_recent:]
            if preserve_recent > 0
            else []
        )

        # Calculate remaining slots for middle messages
        remaining_slots = available_slots - len(recent_messages)
        if remaining_slots <= 0:
            # Only keep system messages and recent messages
            result = system_messages + recent_messages
            return result[:max_messages]

        # Get messages from the beginning (excluding recent ones)
        early_messages = (
            other_messages[:-preserve_recent]
            if preserve_recent > 0
            else other_messages
        )

        # If we have enough slots for all early messages
        if len(early_messages) <= remaining_slots:
            result = (
                system_messages + early_messages + recent_messages
            )
            return result[:max_messages]

        # Apply middle-out compression to early messages
        compressed_early = self._middle_out_compress(
            early_messages, remaining_slots
        )

        result = system_messages + compressed_early + recent_messages
        return result[:max_messages]

    def _compress_tokens(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Compress messages to fit within token limit using middle-out strategy.

        Args:
            messages: List of messages to compress
            max_tokens: Maximum token count

        Returns:
            Compressed list of messages
        """
        current_tokens = self._count_total_tokens(messages)

        if current_tokens <= max_tokens:
            return messages

        # First try to compress message count if we have too many messages
        if (
            len(messages) > 50
        ):  # Arbitrary threshold for when to try message count compression first
            messages = self._compress_message_count(
                messages, len(messages) // 2
            )

        current_tokens = self._count_total_tokens(messages)
        if current_tokens <= max_tokens:
            return messages

        # Apply middle-out compression with token awareness
        return self._middle_out_compress_tokens(messages, max_tokens)

    def _middle_out_compress(
        self, messages: List[Dict[str, Any]], target_count: int
    ) -> List[Dict[str, Any]]:
        """
        Apply middle-out compression to reduce message count.

        Args:
            messages: Messages to compress
            target_count: Target number of messages

        Returns:
            Compressed messages
        """
        if len(messages) <= target_count:
            return messages

        # Keep first half and last half
        keep_count = target_count // 2
        first_half = messages[:keep_count]
        last_half = messages[-keep_count:]

        # Combine first half, last half, and if odd number, add the middle message
        result = first_half + last_half

        if target_count % 2 == 1 and len(messages) > keep_count * 2:
            middle_index = len(messages) // 2
            result.insert(keep_count, messages[middle_index])

        return result[:target_count]

    def _middle_out_compress_tokens(
        self, messages: List[Dict[str, Any]], max_tokens: int
    ) -> List[Dict[str, Any]]:
        """
        Apply middle-out compression with token awareness.

        Args:
            messages: Messages to compress
            max_tokens: Maximum token count

        Returns:
            Compressed messages
        """
        # Start by keeping all messages and remove from middle until under token limit
        current_messages = messages.copy()

        while (
            self._count_total_tokens(current_messages) > max_tokens
            and len(current_messages) > 2
        ):
            # Remove from the middle
            if len(current_messages) <= 2:
                break

            # Find the middle message (avoiding system messages if possible)
            middle_index = len(current_messages) // 2

            # Try to avoid removing system messages
            if current_messages[middle_index].get("role") == "system":
                # Look for a non-system message near the middle
                for offset in range(
                    1, len(current_messages) // 4 + 1
                ):
                    if (
                        middle_index - offset >= 0
                        and current_messages[
                            middle_index - offset
                        ].get("role")
                        != "system"
                    ):
                        middle_index = middle_index - offset
                        break
                    if (
                        middle_index + offset < len(current_messages)
                        and current_messages[
                            middle_index + offset
                        ].get("role")
                        != "system"
                    ):
                        middle_index = middle_index + offset
                        break

            # Remove the middle message
            current_messages.pop(middle_index)

        return current_messages

    def _count_total_tokens(
        self, messages: List[Dict[str, Any]]
    ) -> int:
        """Count total tokens in a list of messages."""
        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += count_tokens(
                    content, self.config.model_name
                )
            elif isinstance(content, (list, dict)):
                # Handle structured content
                total_tokens += count_tokens(
                    str(content), self.config.model_name
                )
        return total_tokens

    def _get_model_context_limit(
        self, model_name: str
    ) -> Optional[int]:
        """
        Get the context token limit for a given model.

        Args:
            model_name: Name of the model

        Returns:
            Token limit or None if unknown
        """
        # Common model context limits (in tokens)
        model_limits = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4.1": 128000,
            "gpt-4o-mini": 128000,
            "gpt-3.5-turbo": 16385,
            "claude-3-opus": 200000,
            "claude-3-sonnet": 200000,
            "claude-3-haiku": 200000,
            "claude-3-5-sonnet": 200000,
            "claude-2": 100000,
            "gemini-pro": 32768,
            "gemini-pro-vision": 16384,
            "llama-2-7b": 4096,
            "llama-2-13b": 4096,
            "llama-2-70b": 4096,
        }

        # Check for exact match first
        if model_name in model_limits:
            return model_limits[model_name]

        # Check for partial matches
        for model_key, limit in model_limits.items():
            if model_key in model_name.lower():
                return limit

        # Default fallback
        logger.warning(
            f"Unknown model '{model_name}', using default context limit of 4096 tokens"
        )
        return 4096

    def _get_model_message_limit(
        self, model_name: str
    ) -> Optional[int]:
        """
        Get the message count limit for a given model.

        Args:
            model_name: Name of the model

        Returns:
            Message limit or None if no limit
        """
        # Models with known message limits
        message_limits = {
            "claude-3-opus": 1000,
            "claude-3-sonnet": 1000,
            "claude-3-haiku": 1000,
            "claude-3-5-sonnet": 1000,
            "claude-2": 1000,
        }

        # Check for exact match first
        if model_name in message_limits:
            return message_limits[model_name]

        # Check for partial matches
        for model_key, limit in message_limits.items():
            if model_key in model_name.lower():
                return limit

        return None  # No known limit


def create_default_transforms(
    enabled: bool = True,
    method: str = "middle-out",
    model_name: str = "gpt-4",
) -> MessageTransforms:
    """
    Create MessageTransforms with default configuration.

    Args:
        enabled: Whether transforms are enabled
        method: Transform method to use
        model_name: Model name for context limits

    Returns:
        Configured MessageTransforms instance
    """
    config = TransformConfig(
        enabled=enabled, method=method, model_name=model_name
    )
    return MessageTransforms(config)


def apply_transforms_to_messages(
    messages: List[Dict[str, Any]],
    transforms_config: Optional[TransformConfig] = None,
    model_name: str = "gpt-4",
) -> TransformResult:
    """
    Convenience function to apply transforms to messages.

    Args:
        messages: List of message dictionaries
        transforms_config: Optional transform configuration
        model_name: Model name for context determination

    Returns:
        TransformResult with processed messages
    """
    if transforms_config is None:
        transforms = create_default_transforms(
            enabled=True, model_name=model_name
        )
    else:
        transforms = MessageTransforms(transforms_config)

    return transforms.transform_messages(messages, model_name)


def handle_transforms(
    transforms: MessageTransforms,
    short_memory: Conversation = None,
    model_name: Optional[str] = "gpt-4.1",
) -> str:
    """
    Handle message transforms and return a formatted task prompt.

    Applies message transforms to the provided messages using the given
    MessageTransforms instance. If compression occurs, logs the results.
    Returns the formatted string of messages after transforms, or the
    original message history as a string if no transforms are enabled.

    Args:
        messages: List of message dictionaries to process.
        transforms: MessageTransforms instance to apply.
        short_memory: Object with methods to return messages as dictionary or string.
        model_name: Name of the model for context.

    Returns:
        Formatted string of messages for the task prompt.
    """
    # Get messages as dictionary format for transforms
    messages_dict = short_memory.return_messages_as_dictionary()

    # Apply transforms
    transform_result = transforms.transform_messages(
        messages_dict, model_name
    )

    # Log transform results if compression occurred
    if transform_result.was_compressed:
        logger.info(
            f"Applied message transforms: {transform_result.original_message_count} -> "
            f"{transform_result.compressed_message_count} messages, "
            f"{transform_result.original_token_count} -> {transform_result.compressed_token_count} tokens "
            f"(ratio: {transform_result.compression_ratio:.2f})"
        )

    # Convert transformed messages back to string format
    formatted_messages = [
        f"{message['role']}: {message['content']}"
        for message in transform_result.messages
    ]
    task_prompt = "\n\n".join(formatted_messages)

    return task_prompt
