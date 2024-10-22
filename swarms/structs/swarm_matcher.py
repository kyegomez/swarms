from typing import List, Tuple, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pydantic import BaseModel, Field
from loguru import logger
import json
from tenacity import retry, stop_after_attempt, wait_exponential

# Ensure you have the necessary libraries installed:
# pip install torch transformers pydantic loguru tenacity


class SwarmType(BaseModel):
    name: str
    description: str
    embedding: Optional[List[float]] = Field(
        default=None, exclude=True
    )


class SwarmMatcherConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = (
        512  # Dimension of the sentence-transformers model
    )


class SwarmMatcher:
    """
    A class for matching tasks to swarm types based on their descriptions.
    It utilizes a transformer model to generate embeddings for task and swarm type descriptions,
    and then calculates the dot product to find the best match.
    """

    def __init__(self, config: SwarmMatcherConfig):
        """
        Initializes the SwarmMatcher with a configuration.

        Args:
            config (SwarmMatcherConfig): The configuration for the SwarmMatcher.
        """
        logger.add("swarm_matcher_debug.log", level="DEBUG")
        logger.debug("Initializing SwarmMatcher")
        try:
            self.config = config
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name
            )
            self.model = AutoModel.from_pretrained(config.model_name)
            self.swarm_types: List[SwarmType] = []
            logger.debug("SwarmMatcher initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SwarmMatcher: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generates an embedding for a given text using the configured model.

        Args:
            text (str): The text for which to generate an embedding.

        Returns:
            np.ndarray: The embedding vector for the text.
        """
        logger.debug(f"Getting embedding for text: {text[:50]}...")
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = (
                outputs.last_hidden_state.mean(dim=1)
                .squeeze()
                .numpy()
            )
            logger.debug("Embedding generated successfully")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def add_swarm_type(self, swarm_type: SwarmType):
        """
        Adds a swarm type to the list of swarm types, generating an embedding for its description.

        Args:
            swarm_type (SwarmType): The swarm type to add.
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
        Finds the best match for a given task among the registered swarm types.

        Args:
            task (str): The task for which to find the best match.

        Returns:
            Tuple[str, float]: A tuple containing the name of the best matching swarm type and the score.
        """
        logger.debug(f"Finding best match for task: {task[:50]}...")
        try:
            task_embedding = self.get_embedding(task)
            best_match = None
            best_score = -float("inf")
            for swarm_type in self.swarm_types:
                score = np.dot(
                    task_embedding, np.array(swarm_type.embedding)
                )
                if score > best_score:
                    best_score = score
                    best_match = swarm_type
            logger.info(
                f"Best match for task: {best_match.name} (score: {best_score})"
            )
            return best_match.name, float(best_score)
        except Exception as e:
            logger.error(
                f"Error finding best match for task: {str(e)}"
            )
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


def initialize_swarm_types(matcher: SwarmMatcher):
    logger.debug("Initializing swarm types")
    swarm_types = [
        SwarmType(
            name="AgentRearrange",
            description="Optimize agent order and rearrange flow for multi-step tasks, ensuring efficient task allocation and minimizing bottlenecks",
        ),
        SwarmType(
            name="MixtureOfAgents",
            description="Combine diverse expert agents for comprehensive analysis, fostering a collaborative approach to problem-solving and leveraging individual strengths",
        ),
        SwarmType(
            name="SpreadSheetSwarm",
            description="Collaborative data processing and analysis in a spreadsheet-like environment, facilitating real-time data sharing and visualization",
        ),
        SwarmType(
            name="SequentialWorkflow",
            description="Execute tasks in a step-by-step, sequential process workflow, ensuring a logical and methodical approach to task execution",
        ),
        SwarmType(
            name="ConcurrentWorkflow",
            description="Process multiple tasks or data sources concurrently in parallel, maximizing productivity and reducing processing time",
        ),
    ]

    for swarm_type in swarm_types:
        matcher.add_swarm_type(swarm_type)
    logger.debug("Swarm types initialized")


def swarm_matcher(task: str, *args, **kwargs):
    """
    Runs the SwarmMatcher example with predefined tasks and swarm types.
    """
    config = SwarmMatcherConfig()
    matcher = SwarmMatcher(config)
    initialize_swarm_types(matcher)

    matcher.save_swarm_types("swarm_types.json")

    swarm_type = matcher.auto_select_swarm(task)

    logger.info(f"{swarm_type}")

    return swarm_type
