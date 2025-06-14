import json
import os
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator
from pydantic.v1 import validator
from litellm import embedding

from swarms.utils.auto_download_check_packages import (
    auto_check_and_download_package,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="swarm_matcher")


class SwarmType(BaseModel):
    name: str
    description: str
    embedding: Optional[List[float]] = Field(
        default=None, exclude=True
    )


api_key = os.getenv("OPENAI_API_KEY")


class SwarmMatcherConfig(BaseModel):
    backend: Literal["local", "openai"] = "local"
    model_name: str = (
        "sentence-transformers/all-MiniLM-L6-v2"  # For local embeddings
    )
    openai_model: str = (
        "text-embedding-3-small"  # Default to newer OpenAI model
    )
    embedding_dim: int = 512  # For local embeddings
    openai_dimensions: Optional[int] = (
        None  # For OpenAI text-embedding-3-* models
    )
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    cache_embeddings: bool = True
    max_sequence_length: int = Field(default=512, ge=64, le=2048)
    device: str = "cpu"  # Only used for local embeddings
    batch_size: int = Field(default=32, ge=1)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    metadata: Optional[Dict] = Field(
        default_factory=dict
    )  # For OpenAI embedding calls

    class Config:
        validate_assignment = True

    @validator("openai_dimensions")
    def validate_dimensions(cls, v, values):
        if values.get("backend") == "openai":
            if (
                values.get("openai_model", "").startswith(
                    "text-embedding-3"
                )
                and v is None
            ):
                # Default to 1536 for text-embedding-3-small/large if not specified
                return 1536
        return v

    @field_validator("openai_model")
    def validate_model(cls, v, values):
        if values.get("backend") == "openai":
            valid_models = [
                "text-embedding-3-small",
                "text-embedding-3-large",
                "text-embedding-ada-002",
            ]
            if v not in valid_models:
                raise ValueError(
                    f"OpenAI model must be one of: {', '.join(valid_models)}"
                )
        return v


class SwarmMatcher:
    """
    A class for matching tasks to swarm types based on their descriptions using semantic similarity.

    This class uses transformer models to generate embeddings for both task descriptions and swarm type descriptions.
    It then calculates similarity scores to find the most appropriate swarm type for a given task.

    Features:
    - Supports both local transformer models and OpenAI embeddings
    - Implements embedding caching for improved performance
    - Provides batch processing capabilities
    - Includes retry mechanisms for API calls
    - Supports saving/loading swarm type configurations
    """

    def __init__(self, config: SwarmMatcherConfig):
        """
        Initializes the SwarmMatcher with a configuration.

        Args:
            config (SwarmMatcherConfig): Configuration object specifying model settings,
                                        similarity thresholds, and other parameters.

        Raises:
            ImportError: If required dependencies (torch, transformers) are not available
            Exception: If model initialization fails
        """
        try:
            self.config = config
            if self.config.backend == "local":
                transformers = self._setup_dependencies()
                self._setup_model_and_tokenizer(transformers)
            self._initialize_state()
            self.initialize_swarm_types()
            logger.debug("SwarmMatcher initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SwarmMatcher: {str(e)}")
            raise

    def _setup_dependencies(self):
        """Set up required dependencies for the SwarmMatcher."""
        try:
            import numpy as np
            import torch
        except ImportError:
            auto_check_and_download_package(
                "torch", package_manager="pip", upgrade=True
            )
            import numpy as np
            import torch

        try:
            import transformers
        except ImportError:
            auto_check_and_download_package(
                "transformers", package_manager="pip", upgrade=True
            )
            import transformers

        self.torch = torch
        self.np = np
        return transformers

    def _setup_model_and_tokenizer(self, transformers):
        """Initialize the model and tokenizer."""
        self.device = self.torch.device(self.config.device)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.config.model_name
        )
        self.model = transformers.AutoModel.from_pretrained(
            self.config.model_name
        ).to(self.device)

    def _initialize_state(self):
        """Initialize internal state variables."""
        self.swarm_types: List[SwarmType] = []
        self._embedding_cache = (
            {} if self.config.cache_embeddings else None
        )

    def _get_cached_embedding(
        self, text: str
    ) -> Optional[np.ndarray]:
        """
        Retrieves a cached embedding if available.

        Args:
            text (str): The text to look up in the cache

        Returns:
            Optional[np.ndarray]: The cached embedding if found, None otherwise
        """
        if self._embedding_cache is not None:
            return self._embedding_cache.get(text)
        return None

    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """
        Stores an embedding in the cache for future use.

        Args:
            text (str): The text associated with the embedding
            embedding (np.ndarray): The embedding vector to cache
        """
        if self._embedding_cache is not None:
            self._embedding_cache[text] = embedding

    def _get_openai_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI's API via litellm."""
        try:
            params = {
                "model": self.config.openai_model,
                "input": [text],
            }

            # Add dimensions parameter for text-embedding-3-* models
            if (
                self.config.openai_model.startswith(
                    "text-embedding-3"
                )
                and self.config.openai_dimensions
            ):
                params["dimensions"] = self.config.openai_dimensions

            response = embedding(**params)
            response = response.model_dump()

            # Handle the response format
            if "data" in response and len(response["data"]) > 0:
                embedding_data = response["data"][0]["embedding"]
            else:
                raise ValueError(
                    f"Unexpected response format from OpenAI API: {response}"
                )

            embedding_array = np.array(embedding_data)

            # Log usage information if available
            if "usage" in response:
                logger.debug(
                    f"OpenAI API usage - Prompt tokens: {response['usage'].get('prompt_tokens', 'N/A')}, "
                    f"Total tokens: {response['usage'].get('total_tokens', 'N/A')}"
                )

            return embedding_array
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {str(e)}")
            raise

    def _get_openai_embeddings_batch(
        self, texts: List[str]
    ) -> np.ndarray:
        """Get embeddings for a batch of texts using OpenAI's API via litellm."""
        try:
            params = {
                "model": self.config.openai_model,
                "input": texts,
            }

            # Add dimensions parameter for text-embedding-3-* models
            if (
                self.config.openai_model.startswith(
                    "text-embedding-3"
                )
                and self.config.openai_dimensions
            ):
                params["dimensions"] = self.config.openai_dimensions

            response = embedding(**params)
            response = response.model_dump()

            # Handle the response format
            if "data" in response:
                embeddings = [
                    data["embedding"] for data in response["data"]
                ]
            else:
                raise ValueError(
                    f"Unexpected response format from OpenAI API: {response}"
                )

            # Log usage information if available
            if "usage" in response:
                logger.debug(
                    f"Batch OpenAI API usage - Prompt tokens: {response['usage'].get('prompt_tokens', 'N/A')}, "
                    f"Total tokens: {response['usage'].get('total_tokens', 'N/A')}"
                )

            return np.array(embeddings)
        except Exception as e:
            logger.error(
                f"Error getting OpenAI embeddings batch: {str(e)}"
            )
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a given text using the configured model.

        This method first checks the cache for an existing embedding. If not found,
        it generates a new embedding using either the local transformer model or OpenAI API.

        Args:
            text (str): The text for which to generate an embedding

        Returns:
            np.ndarray: The embedding vector for the text

        Raises:
            Exception: If embedding generation fails
        """
        # Check cache first
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            return cached_embedding

        logger.debug(f"Getting embedding for text: {text[:50]}...")
        try:
            if self.config.backend == "openai":
                embedding = self._get_openai_embedding(text)
            else:
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_sequence_length,
                )
                # Move inputs to device
                inputs = {
                    k: v.to(self.device) for k, v in inputs.items()
                }

                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                embedding = (
                    outputs.last_hidden_state.mean(dim=1)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            # Cache the embedding
            self._cache_embedding(text, embedding)

            logger.debug("Embedding generated successfully")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batch for improved efficiency.

        This method processes texts in batches, utilizing the cache where possible
        and generating new embeddings only for uncached texts.

        Args:
            texts (List[str]): List of texts to generate embeddings for

        Returns:
            np.ndarray: Array of embeddings, one for each input text

        Raises:
            Exception: If batch processing fails
        """
        embeddings = []
        batch_texts = []

        for text in texts:
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                batch_texts.append(text)

        if batch_texts:
            if self.config.backend == "openai":
                batch_embeddings = self._get_openai_embeddings_batch(
                    batch_texts
                )
                for text, embedding in zip(
                    batch_texts, batch_embeddings
                ):
                    self._cache_embedding(text, embedding)
                    embeddings.append(embedding)
            else:
                for i in range(
                    0, len(batch_texts), self.config.batch_size
                ):
                    batch = batch_texts[
                        i : i + self.config.batch_size
                    ]
                    inputs = self.tokenizer(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_sequence_length,
                    )
                    inputs = {
                        k: v.to(self.device)
                        for k, v in inputs.items()
                    }

                    with self.torch.no_grad():
                        outputs = self.model(**inputs)
                    batch_embeddings = (
                        outputs.last_hidden_state.mean(dim=1)
                        .cpu()
                        .numpy()
                    )

                    for text, embedding in zip(
                        batch, batch_embeddings
                    ):
                        self._cache_embedding(text, embedding)
                        embeddings.append(embedding)

        return np.array(embeddings)

    def add_swarm_type(self, swarm_type: SwarmType):
        """
        Adds a swarm type to the matcher's registry.

        Generates and stores an embedding for the swarm type's description.

        Args:
            swarm_type (SwarmType): The swarm type to add

        Raises:
            Exception: If embedding generation or storage fails
        """
        logger.debug(f"Adding swarm type: {swarm_type.name}")
        try:
            embedding = self.get_embedding(swarm_type.description)
            swarm_type.embedding = embedding.tolist()
            self.swarm_types.append(swarm_type)
            logger.info(f"Added swarm type: {swarm_type.name}")
        except Exception as e:
            logger.error(
                f"Error adding swarm type {swarm_type.name}: {str(e)}"
            )
            raise

    def find_best_match(self, task: str) -> Tuple[str, float]:
        """
        Finds the best matching swarm type for a given task.

        Uses semantic similarity to compare the task against all registered swarm types
        and returns the best match along with its confidence score.

        Args:
            task (str): The task description to match

        Returns:
            Tuple[str, float]: A tuple containing:
                - The name of the best matching swarm type
                - The similarity score (between 0 and 1)

        Raises:
            Exception: If matching process fails
        """
        logger.debug(f"Finding best match for task: {task[:50]}...")
        try:
            task_embedding = self.get_embedding(task)
            best_match = None
            best_score = -float("inf")

            # Get all swarm type embeddings in batch
            swarm_descriptions = [
                st.description for st in self.swarm_types
            ]
            swarm_embeddings = self.get_embeddings_batch(
                swarm_descriptions
            )

            # Calculate similarity scores in batch
            scores = np.dot(task_embedding, swarm_embeddings.T)
            best_idx = np.argmax(scores)
            best_score = float(scores[best_idx])
            best_match = self.swarm_types[best_idx]

            if best_score < self.config.similarity_threshold:
                logger.warning(
                    f"Best match score {best_score} is below threshold {self.config.similarity_threshold}"
                )

            logger.info(
                f"Best match for task: {best_match.name} (score: {best_score})"
            )
            return best_match.name, best_score
        except Exception as e:
            logger.error(
                f"Error finding best match for task: {str(e)}"
            )
            raise

    def find_top_k_matches(
        self, task: str, k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Finds the top k matching swarm types for a given task.

        Returns all matches that exceed the similarity threshold, sorted by score.

        Args:
            task (str): The task for which to find matches.
            k (int): Number of top matches to return.

        Returns:
            List[Tuple[str, float]]: List of tuples containing swarm names and their scores.
        """
        logger.debug(
            f"Finding top {k} matches for task: {task[:50]}..."
        )
        try:
            task_embedding = self.get_embedding(task)
            swarm_descriptions = [
                st.description for st in self.swarm_types
            ]
            swarm_embeddings = self.get_embeddings_batch(
                swarm_descriptions
            )

            # Calculate similarity scores in batch
            scores = np.dot(task_embedding, swarm_embeddings.T)
            top_k_indices = np.argsort(scores)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                score = float(scores[idx])
                if score >= self.config.similarity_threshold:
                    results.append(
                        (self.swarm_types[idx].name, score)
                    )

            logger.info(
                f"Found {len(results)} matches above threshold"
            )
            return results
        except Exception as e:
            logger.error(f"Error finding top matches: {str(e)}")
            raise

    def auto_select_swarm(self, task: str) -> str:
        """
        Automatically selects the best swarm type for a given task based on their descriptions.

        Args:
            task (str): The task for which to select a swarm type.

        Returns:
            str: The name of the selected swarm type.
        """
        logger.debug(f"Auto-selecting swarm for task: {task[:50]}...")
        best_match, score = self.find_best_match(task)
        logger.info(f"Task: {task}")
        logger.info(f"Selected Swarm Type: {best_match}")
        logger.info(f"Confidence Score: {score:.2f}")
        return best_match

    def run_multiple(self, tasks: List[str], *args, **kwargs) -> str:
        swarms = []

        for task in tasks:
            output = self.auto_select_swarm(task)

            # Append
            swarms.append(output)

        return swarms

    def save_swarm_types(self, filename: str):
        """
        Saves the registered swarm types to a JSON file.

        Args:
            filename (str): The name of the file to which to save the swarm types.
        """
        try:
            with open(filename, "w") as f:
                json.dump([st.dict() for st in self.swarm_types], f)
            logger.info(f"Saved swarm types to {filename}")
        except Exception as e:
            logger.error(f"Error saving swarm types: {str(e)}")
            raise

    def load_swarm_types(self, filename: str):
        """
        Loads swarm types from a JSON file.

        Args:
            filename (str): The name of the file from which to load the swarm types.
        """
        try:
            with open(filename, "r") as f:
                swarm_types_data = json.load(f)
            self.swarm_types = [
                SwarmType(**st) for st in swarm_types_data
            ]
            logger.info(f"Loaded swarm types from {filename}")
        except Exception as e:
            logger.error(f"Error loading swarm types: {str(e)}")
            raise

    def initialize_swarm_types(self):
        logger.debug("Initializing swarm types")
        swarm_types = [
            SwarmType(
                name="AgentRearrange",
                description="Optimize agent order and rearrange flow for multi-step tasks, ensuring efficient task allocation and minimizing bottlenecks. Keywords: orchestration, coordination, pipeline optimization, task scheduling, resource allocation, workflow management, agent organization, process optimization",
            ),
            SwarmType(
                name="MixtureOfAgents",
                description="Combine diverse expert agents for comprehensive analysis, fostering a collaborative approach to problem-solving and leveraging individual strengths. Keywords: multi-agent system, expert collaboration, distributed intelligence, collective problem solving, agent specialization, team coordination, hybrid approaches, knowledge synthesis",
            ),
            SwarmType(
                name="SpreadSheetSwarm",
                description="Collaborative data processing and analysis in a spreadsheet-like environment, facilitating real-time data sharing and visualization. Keywords: data analysis, tabular processing, collaborative editing, data transformation, spreadsheet operations, data visualization, real-time collaboration, structured data",
            ),
            SwarmType(
                name="SequentialWorkflow",
                description="Execute tasks in a step-by-step, sequential process workflow, ensuring a logical and methodical approach to task execution. Keywords: linear processing, waterfall methodology, step-by-step execution, ordered tasks, sequential operations, process flow, systematic approach, staged execution",
            ),
            SwarmType(
                name="ConcurrentWorkflow",
                description="Process multiple tasks or data sources concurrently in parallel, maximizing productivity and reducing processing time. Keywords: parallel processing, multi-threading, asynchronous execution, distributed computing, concurrent operations, simultaneous tasks, parallel workflows, scalable processing",
            ),
            SwarmType(
                name="HierarchicalSwarm",
                description="Organize agents in a hierarchical structure with clear reporting lines and delegation of responsibilities. Keywords: management hierarchy, organizational structure, delegation, supervision, chain of command, tiered organization, structured coordination, leadership roles",
            ),
            SwarmType(
                name="AdaptiveSwarm",
                description="Dynamically adjust agent behavior and swarm configuration based on task requirements and performance feedback. Keywords: dynamic adaptation, self-optimization, feedback loops, learning systems, flexible configuration, responsive behavior, adaptive algorithms, real-time adjustment",
            ),
            SwarmType(
                name="ConsensusSwarm",
                description="Achieve group decisions through consensus mechanisms and voting protocols among multiple agents. Keywords: group decision making, voting systems, collective intelligence, agreement protocols, democratic processes, collaborative decisions, consensus building",
            ),
            SwarmType(
                name="DeepResearchSwarm",
                description="Conduct in-depth research and analysis by coordinating multiple agents to explore, synthesize, and validate information from various sources. Keywords: research methodology, information synthesis, data validation, comprehensive analysis, knowledge discovery, systematic investigation",
            ),
            SwarmType(
                name="CouncilAsAJudge",
                description="Evaluate and judge solutions or decisions through a council of expert agents acting as arbitrators. Keywords: evaluation, judgment, arbitration, expert assessment, quality control, decision validation, peer review, consensus building",
            ),
            SwarmType(
                name="MALT",
                description="Multi-Agent Language Tasks framework for coordinating language-based operations across multiple specialized agents. Keywords: language processing, task coordination, linguistic analysis, communication protocols, semantic understanding, natural language tasks",
            ),
            SwarmType(
                name="GroupChat",
                description="Enable dynamic multi-agent conversations and collaborative problem-solving through structured group discussions. Keywords: collaborative dialogue, group interaction, team communication, collective problem-solving, discussion facilitation, knowledge sharing",
            ),
            SwarmType(
                name="MultiAgentRouter",
                description="Intelligently route tasks and information between agents based on their specializations and current workload. Keywords: task distribution, load balancing, intelligent routing, agent specialization, workflow optimization, resource allocation",
            ),
            SwarmType(
                name="MajorityVoting",
                description="Make decisions through democratic voting mechanisms where multiple agents contribute their opinions and votes. Keywords: collective decision-making, democratic process, vote aggregation, opinion pooling, consensus building, collaborative choice",
            ),
        ]

        try:
            for swarm_type in swarm_types:
                self.add_swarm_type(swarm_type)
        except Exception as e:
            logger.error(f"Error initializing swarm types: {str(e)}")
            raise


def swarm_matcher(task: Union[str, List[str]], *args, **kwargs):
    """
    Runs the SwarmMatcher example with predefined tasks and swarm types.
    """
    if isinstance(task, list):
        task = "".join(task)
    else:
        task = task

    config = SwarmMatcherConfig()
    matcher = SwarmMatcher(config)

    # matcher.save_swarm_types(f"swarm_logs/{uuid4().hex}.json")

    swarm_type = matcher.auto_select_swarm(task)

    logger.info(f"{swarm_type}")

    return swarm_type


# # Example usage
# if __name__ == "__main__":
#     # Create configuration
#     config = SwarmMatcherConfig(
#         backend="openai",  # Using local embeddings for this example
#         similarity_threshold=0.6,  # Increase threshold for more strict matching
#         cache_embeddings=True,
#     )

#     # Initialize matcher
#     matcher = SwarmMatcher(config)

#     task = "I need to concurrently run 1000 tasks"

#     print(matcher.auto_select_swarm(task))
