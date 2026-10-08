import math
from typing import Any, Callable, List, Optional, Union

from litellm import embedding
from tenacity import retry, stop_after_attempt, wait_exponential

from swarms.structs.ma_blocks import find_agent_by_name
from swarms.structs.omni_agent_types import AgentType
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="agent_router")


class AgentRouter:
    """
    Initialize the AgentRouter using LiteLLM embeddings for agent matching.

    Args:
        embedding_model (str): The embedding model to use for generating embeddings.
            Examples: 'text-embedding-ada-002', 'text-embedding-3-small', 'text-embedding-3-large',
            'cohere/embed-english-v3.0', 'huggingface/microsoft/codebert-base', etc.
        n_agents (int): Number of agents to return in queries.
        api_key (str, optional): API key for the embedding service. If not provided,
            will use environment variables.
        api_base (str, optional): Custom API base URL for the embedding service.
        agents (List[AgentType], optional): List of agents to initialize the router with.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-ada-002",
        n_agents: int = 1,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        agents: Optional[List[AgentType]] = None,
    ):
        self.embedding_model = embedding_model
        self.n_agents = n_agents
        self.api_key = api_key
        self.api_base = api_base
        self.agents: List[AgentType] = []
        self.agent_embeddings: List[List[float]] = []
        self.agent_metadata: List[dict] = []

        # Add agents if provided during initialization
        if agents:
            self.add_agents(agents)

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text using the specified model.

        Args:
            text (str): The text to generate embedding for.

        Returns:
            List[float]: The embedding vector as a list of floats.
        """
        try:
            # Prepare parameters for the embedding call
            params = {"model": self.embedding_model, "input": [text]}

            if self.api_key:
                params["api_key"] = self.api_key
            if self.api_base:
                params["api_base"] = self.api_base

            response = embedding(**params)

            # Handle different response structures from litellm
            if hasattr(response, "data") and response.data:
                if hasattr(response.data[0], "embedding"):
                    embedding_vector = response.data[0].embedding
                elif (
                    isinstance(response.data[0], dict)
                    and "embedding" in response.data[0]
                ):
                    embedding_vector = response.data[0]["embedding"]
                else:
                    logger.error(
                        f"Unexpected response structure: {response.data[0]}"
                    )
                    raise ValueError(
                        f"Unexpected embedding response structure: {type(response.data[0])}"
                    )
            else:
                logger.error(
                    f"Unexpected response structure: {response}"
                )
                raise ValueError(
                    f"Unexpected embedding response structure: {type(response)}"
                )

            return embedding_vector

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1 (List[float]): First vector.
            vec2 (List[float]): Second vector.

        Returns:
            float: Cosine similarity between the vectors.
        """
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def add_agent(self, agent: AgentType) -> None:
        """
        Add an agent to the embedding-based agent router.

        Args:
            agent (Agent): The agent to add.

        Raises:
            Exception: If there's an error adding the agent to the router.
        """
        try:
            agent_text = f"{agent.name} {agent.description} {agent.system_prompt}"

            # Generate embedding for the agent
            agent_embedding = self._generate_embedding(agent_text)

            # Store agent and its embedding
            self.agents.append(agent)
            self.agent_embeddings.append(agent_embedding)
            self.agent_metadata.append(
                {"name": agent.name, "text": agent_text}
            )

            logger.info(
                f"Added agent {agent.name} to the embedding-based router."
            )
        except Exception as e:
            logger.error(
                f"Error adding agent {agent.name} to the router: {str(e)}"
            )
            raise

    def add_agents(
        self, agents: List[Union[AgentType, Callable, Any]]
    ) -> None:
        """
        Add multiple agents to the vector database.

        Args:
            agents (List[Union[Agent, Callable, Any]]): List of agents to add.
        """
        for agent in agents:
            self.add_agent(agent)

    def update_agent_history(self, agent_name: str) -> None:
        """
        Update the agent's entry in the router with its interaction history.

        Args:
            agent_name (str): The name of the agent to update.
        """
        agent = find_agent_by_name(self.agents, agent_name)

        history = agent.short_memory.return_history_as_string()
        history_text = " ".join(history)
        updated_text = f"{agent.name} {agent.description} {agent.system_prompt} {history_text}"

        try:
            agent_index = self.agents.index(agent)
        except ValueError:
            logger.warning(
                f"Agent {agent_name} not found in the agents list."
            )
            return

        updated_embedding = self._generate_embedding(updated_text)
        self.agent_embeddings[agent_index] = updated_embedding
        self.agent_metadata[agent_index] = {
            "name": agent_name,
            "text": updated_text,
        }

        logger.info(
            f"Updated agent {agent_name} with interaction history."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def find_best_agent(
        self, task: str, *args, **kwargs
    ) -> Optional[AgentType]:
        """
        Find the best agent for a given task using cosine similarity.

        Args:
            task (str): The task description.
            *args: Additional arguments (unused, kept for compatibility).
            **kwargs: Additional keyword arguments (unused, kept for compatibility).

        Returns:
            Optional[Agent]: The best matching agent, if found.

        Raises:
            Exception: If there's an error finding the best agent.
        """
        try:
            if not self.agents or not self.agent_embeddings:
                logger.warning("No agents available in the router.")
                return None

            # Generate embedding for the task
            task_embedding = self._generate_embedding(task)

            # Calculate cosine similarities
            similarities = []
            for agent_embedding in self.agent_embeddings:
                similarity = self._cosine_similarity(
                    task_embedding, agent_embedding
                )
                similarities.append(similarity)

            # Find the best matching agent(s)
            if similarities:
                # Get index of the best similarity
                best_index = similarities.index(max(similarities))
                best_agent = self.agents[best_index]
                best_similarity = similarities[best_index]

                logger.info(
                    f"Found best matching agent: {best_agent.name} (similarity: {best_similarity:.4f})"
                )
                return best_agent
            else:
                logger.warning(
                    "No matching agent found for the given task."
                )

            return None
        except Exception as e:
            logger.error(f"Error finding best agent: {str(e)}")
            raise

    def run(self, task: str) -> Optional[AgentType]:
        """
        Run the agent router on a given task.
        """
        return self.find_best_agent(task)
