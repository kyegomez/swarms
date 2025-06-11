import time
from typing import Any, Dict, List, Optional

from loguru import logger
from swarms.utils.litellm_tokenizer import count_tokens
from pydantic import BaseModel, Field, field_validator


class RAGConfig(BaseModel):
    """Configuration class for RAG operations"""

    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for memory retrieval",
    )
    max_results: int = Field(
        default=5,
        gt=0,
        description="Maximum number of results to return from memory",
    )
    context_window_tokens: int = Field(
        default=2000,
        gt=0,
        description="Maximum number of tokens in the context window",
    )
    auto_save_to_memory: bool = Field(
        default=True,
        description="Whether to automatically save responses to memory",
    )
    save_every_n_loops: int = Field(
        default=5, gt=0, description="Save to memory every N loops"
    )
    min_content_length: int = Field(
        default=50,
        gt=0,
        description="Minimum content length to save to memory",
    )
    query_every_loop: bool = Field(
        default=False,
        description="Whether to query memory every loop",
    )
    enable_conversation_summaries: bool = Field(
        default=True,
        description="Whether to enable conversation summaries",
    )
    relevance_keywords: Optional[List[str]] = Field(
        default=None, description="Keywords to check for relevance"
    )

    @field_validator("relevance_keywords", mode="before")
    def set_default_keywords(cls, v):
        if v is None:
            return [
                "important",
                "key",
                "critical",
                "summary",
                "conclusion",
            ]
        return v

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True
        json_schema_extra = {
            "example": {
                "similarity_threshold": 0.7,
                "max_results": 5,
                "context_window_tokens": 2000,
                "auto_save_to_memory": True,
                "save_every_n_loops": 5,
                "min_content_length": 50,
                "query_every_loop": False,
                "enable_conversation_summaries": True,
                "relevance_keywords": [
                    "important",
                    "key",
                    "critical",
                    "summary",
                    "conclusion",
                ],
            }
        }


class AgentRAGHandler:
    """
    Handles all RAG (Retrieval-Augmented Generation) operations for agents.
    Provides memory querying, storage, and context management capabilities.
    """

    def __init__(
        self,
        long_term_memory: Optional[Any] = None,
        config: Optional[RAGConfig] = None,
        agent_name: str = "Unknown",
        max_context_length: int = 158_000,
        verbose: bool = False,
    ):
        """
        Initialize the RAG handler.

        Args:
            long_term_memory: The long-term memory store (must implement add() and query() methods)
            config: RAG configuration settings
            agent_name: Name of the agent using this handler
            verbose: Enable verbose logging
        """
        self.long_term_memory = long_term_memory
        self.config = config or RAGConfig()
        self.agent_name = agent_name
        self.verbose = verbose
        self.max_context_length = max_context_length

        self._loop_counter = 0
        self._conversation_history = []
        self._important_memories = []

        # Validate memory interface
        if (
            self.long_term_memory
            and not self._validate_memory_interface()
        ):
            logger.warning(
                "Long-term memory doesn't implement required interface"
            )

    def _validate_memory_interface(self) -> bool:
        """Validate that the memory object has required methods"""
        required_methods = ["add", "query"]
        for method in required_methods:
            if not hasattr(self.long_term_memory, method):
                logger.error(
                    f"Memory object missing required method: {method}"
                )
                return False
        return True

    def is_enabled(self) -> bool:
        """Check if RAG is enabled (has valid memory store)"""
        return self.long_term_memory is not None

    def query_memory(
        self,
        query: str,
        context_type: str = "general",
        loop_count: Optional[int] = None,
    ) -> str:
        """
        Query the long-term memory and return formatted context.

        Args:
            query: The query string to search for
            context_type: Type of context being queried (for logging)
            loop_count: Current loop number (for logging)

        Returns:
            Formatted string of relevant memories, empty string if no results
        """
        if not self.is_enabled():
            return ""

        try:
            if self.verbose:
                logger.info(
                    f"🔍 [{self.agent_name}] Querying RAG for {context_type}: {query[:100]}..."
                )

            # Query the memory store
            results = self.long_term_memory.query(
                query=query,
                top_k=self.config.max_results,
                similarity_threshold=self.config.similarity_threshold,
            )

            if not results:
                if self.verbose:
                    logger.info(
                        f"No relevant memories found for query: {context_type}"
                    )
                return ""

            # Format results for context
            formatted_context = self._format_memory_results(
                results, context_type, loop_count
            )

            # Ensure context fits within token limits
            if (
                count_tokens(formatted_context)
                > self.config.context_window_tokens
            ):
                formatted_context = self._truncate_context(
                    formatted_context
                )

            if self.verbose:
                logger.info(
                    f"✅ Retrieved {len(results)} relevant memories for {context_type}"
                )

            return formatted_context

        except Exception as e:
            logger.error(f"Error querying long-term memory: {e}")
            return ""

    def _format_memory_results(
        self,
        results: List[Any],
        context_type: str,
        loop_count: Optional[int] = None,
    ) -> str:
        """Format memory results into a structured context string"""
        if not results:
            return ""

        loop_info = f" (Loop {loop_count})" if loop_count else ""
        header = (
            f"📚 Relevant Knowledge - {context_type.title()}{loop_info}:\n"
            + "=" * 50
            + "\n"
        )

        formatted_sections = [header]

        for i, result in enumerate(results, 1):
            (
                content,
                score,
                source,
                metadata,
            ) = self._extract_result_fields(result)

            section = f"""
[Memory {i}] Relevance: {score} | Source: {source}
{'-' * 40}
{content}
{'-' * 40}
"""
            formatted_sections.append(section)

        formatted_sections.append(f"\n{'='*50}\n")
        return "\n".join(formatted_sections)

    def _extract_result_fields(self, result: Any) -> tuple:
        """Extract content, score, source, and metadata from a result object"""
        if isinstance(result, dict):
            content = result.get(
                "content", result.get("text", str(result))
            )
            score = result.get(
                "score", result.get("similarity", "N/A")
            )
            metadata = result.get("metadata", {})
            source = metadata.get(
                "source", result.get("source", "Unknown")
            )
        else:
            content = str(result)
            score = "N/A"
            source = "Unknown"
            metadata = {}

        return content, score, source, metadata

    def _truncate_context(self, content: str) -> str:
        """Truncate content to fit within token limits using smart truncation"""
        max_chars = (
            self.config.context_window_tokens * 3
        )  # Rough token-to-char ratio

        if len(content) <= max_chars:
            return content

        # Try to cut at section boundaries first
        sections = content.split("=" * 50)
        if len(sections) > 2:  # Header + sections + footer
            truncated_sections = [sections[0]]  # Keep header
            current_length = len(sections[0])

            for section in sections[1:-1]:  # Skip footer
                if current_length + len(section) > max_chars * 0.9:
                    break
                truncated_sections.append(section)
                current_length += len(section)

            truncated_sections.append(
                f"\n[... {len(sections) - len(truncated_sections)} more memories truncated for length ...]\n"
            )
            truncated_sections.append(sections[-1])  # Keep footer
            return "=" * (50).join(truncated_sections)

        # Fallback: simple truncation at sentence boundary
        truncated = content[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:
            truncated = truncated[: last_period + 1]

        return (
            truncated + "\n\n[... content truncated for length ...]"
        )

    def should_save_response(
        self,
        response: str,
        loop_count: int,
        has_tool_usage: bool = False,
    ) -> bool:
        """
        Determine if a response should be saved to long-term memory.

        Args:
            response: The response text to evaluate
            loop_count: Current loop number
            has_tool_usage: Whether tools were used in this response

        Returns:
            Boolean indicating whether to save the response
        """
        if (
            not self.is_enabled()
            or not self.config.auto_save_to_memory
        ):
            return False

        # Content length check
        if len(response.strip()) < self.config.min_content_length:
            return False

        save_conditions = [
            # Substantial content
            len(response) > 200,
            # Contains important keywords
            any(
                keyword in response.lower()
                for keyword in self.config.relevance_keywords
            ),
            # Periodic saves
            loop_count % self.config.save_every_n_loops == 0,
            # Tool usage indicates potentially important information
            has_tool_usage,
            # Complex responses (multiple sentences)
            response.count(".") >= 3,
            # Contains structured data or lists
            any(
                marker in response
                for marker in ["- ", "1. ", "2. ", "* ", "```"]
            ),
        ]

        return any(save_conditions)

    def save_to_memory(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        content_type: str = "response",
    ) -> bool:
        """
        Save content to long-term memory with metadata.

        Args:
            content: The content to save
            metadata: Additional metadata to store
            content_type: Type of content being saved

        Returns:
            Boolean indicating success
        """
        if not self.is_enabled():
            return False

        if (
            not content
            or len(content.strip()) < self.config.min_content_length
        ):
            return False

        try:
            # Create default metadata
            default_metadata = {
                "timestamp": time.time(),
                "agent_name": self.agent_name,
                "content_type": content_type,
                "loop_count": self._loop_counter,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Merge with provided metadata
            if metadata:
                default_metadata.update(metadata)

            if self.verbose:
                logger.info(
                    f"💾 [{self.agent_name}] Saving to long-term memory: {content[:100]}..."
                )

            success = self.long_term_memory.add(
                content, metadata=default_metadata
            )

            if success and self.verbose:
                logger.info(
                    f"✅ Successfully saved {content_type} to long-term memory"
                )

            # Track important memories
            if content_type in [
                "final_response",
                "conversation_summary",
            ]:
                self._important_memories.append(
                    {
                        "content": content[:200],
                        "timestamp": time.time(),
                        "type": content_type,
                    }
                )

            return success

        except Exception as e:
            logger.error(f"Error saving to long-term memory: {e}")
            return False

    def create_conversation_summary(
        self,
        task: str,
        final_response: str,
        total_loops: int,
        tools_used: List[str] = None,
    ) -> str:
        """Create a comprehensive summary of the conversation"""
        tools_info = (
            f"Tools Used: {', '.join(tools_used)}"
            if tools_used
            else "Tools Used: None"
        )

        summary = f"""
CONVERSATION SUMMARY
====================
Agent: {self.agent_name}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

ORIGINAL TASK:
{task}

FINAL RESPONSE:
{final_response}

EXECUTION DETAILS:
- Total Reasoning Loops: {total_loops}
- {tools_info}
- Memory Queries Made: {len(self._conversation_history)}

KEY INSIGHTS:
{self._extract_key_insights(final_response)}
====================
"""
        return summary

    def _extract_key_insights(self, response: str) -> str:
        """Extract key insights from the response for summary"""
        # Simple keyword-based extraction
        insights = []
        sentences = response.split(".")

        for sentence in sentences:
            if any(
                keyword in sentence.lower()
                for keyword in self.config.relevance_keywords[:5]
            ):
                insights.append(sentence.strip())

        if insights:
            return "\n- " + "\n- ".join(
                insights[:3]
            )  # Top 3 insights
        return "No specific insights extracted"

    def handle_loop_memory_operations(
        self,
        task: str,
        response: str,
        loop_count: int,
        conversation_context: str = "",
        has_tool_usage: bool = False,
    ) -> str:
        """
        Handle all memory operations for a single loop iteration.

        Args:
            task: Original task
            response: Current response
            loop_count: Current loop number
            conversation_context: Current conversation context
            has_tool_usage: Whether tools were used

        Returns:
            Retrieved context string (empty if no relevant memories)
        """
        self._loop_counter = loop_count
        retrieved_context = ""

        # 1. Query memory if enabled for this loop
        if self.config.query_every_loop and loop_count > 1:
            query_context = f"Task: {task}\nCurrent Context: {conversation_context[-500:]}"
            retrieved_context = self.query_memory(
                query_context,
                context_type=f"loop_{loop_count}",
                loop_count=loop_count,
            )

        # 2. Save response if criteria met
        if self.should_save_response(
            response, loop_count, has_tool_usage
        ):
            self.save_to_memory(
                content=response,
                metadata={
                    "task_preview": task[:200],
                    "loop_count": loop_count,
                    "has_tool_usage": has_tool_usage,
                },
                content_type="loop_response",
            )

        return retrieved_context

    def handle_initial_memory_query(self, task: str) -> str:
        """Handle the initial memory query before reasoning loops begin"""
        if not self.is_enabled():
            return ""

        return self.query_memory(task, context_type="initial_task")

    def handle_final_memory_consolidation(
        self,
        task: str,
        final_response: str,
        total_loops: int,
        tools_used: List[str] = None,
    ) -> bool:
        """Handle final memory consolidation after all loops complete"""
        if (
            not self.is_enabled()
            or not self.config.enable_conversation_summaries
        ):
            return False

        # Create and save conversation summary
        summary = self.create_conversation_summary(
            task, final_response, total_loops, tools_used
        )

        return self.save_to_memory(
            content=summary,
            metadata={
                "task": task[:200],
                "total_loops": total_loops,
                "tools_used": tools_used or [],
            },
            content_type="conversation_summary",
        )

    def search_memories(
        self,
        query: str,
        top_k: int = None,
        similarity_threshold: float = None,
    ) -> List[Dict]:
        """
        Search long-term memory and return raw results.

        Args:
            query: Search query
            top_k: Number of results to return (uses config default if None)
            similarity_threshold: Similarity threshold (uses config default if None)

        Returns:
            List of memory results
        """
        if not self.is_enabled():
            return []

        try:
            results = self.long_term_memory.query(
                query=query,
                top_k=top_k or self.config.max_results,
                similarity_threshold=similarity_threshold
                or self.config.similarity_threshold,
            )
            return results if results else []
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage and operations"""
        return {
            "is_enabled": self.is_enabled(),
            "config": self.config.__dict__,
            "loops_processed": self._loop_counter,
            "important_memories_count": len(self._important_memories),
            "last_important_memories": (
                self._important_memories[-3:]
                if self._important_memories
                else []
            ),
            "memory_store_type": (
                type(self.long_term_memory).__name__
                if self.long_term_memory
                else None
            ),
        }

    def clear_session_data(self):
        """Clear session-specific data (not the long-term memory store)"""
        self._loop_counter = 0
        self._conversation_history.clear()
        self._important_memories.clear()

        if self.verbose:
            logger.info(f"[{self.agent_name}] Session data cleared")

    def update_config(self, **kwargs):
        """Update RAG configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                if self.verbose:
                    logger.info(
                        f"Updated RAG config: {key} = {value}"
                    )
            else:
                logger.warning(f"Unknown config parameter: {key}")


# # Example memory interface that your RAG implementation should follow
# class ExampleMemoryInterface:
#     """Example interface for long-term memory implementations"""

#     def add(self, content: str, metadata: Dict = None) -> bool:
#         """
#         Add content to the memory store.

#         Args:
#             content: Text content to store
#             metadata: Additional metadata dictionary

#         Returns:
#             Boolean indicating success
#         """
#         # Your vector database implementation here
#         return True

#     def query(
#         self,
#         query: str,
#         top_k: int = 5,
#         similarity_threshold: float = 0.7
#     ) -> List[Dict]:
#         """
#         Query the memory store for relevant content.

#         Args:
#             query: Search query string
#             top_k: Maximum number of results to return
#             similarity_threshold: Minimum similarity score

#         Returns:
#             List of dictionaries with keys: 'content', 'score', 'metadata'
#         """
#         # Your vector database query implementation here
#         return [
#             {
#                 'content': 'Example memory content',
#                 'score': 0.85,
#                 'metadata': {'source': 'example', 'timestamp': time.time()}
#             }
#         ]
