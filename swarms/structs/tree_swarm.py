import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, List, Optional

from litellm import embedding
from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="tree_swarm")


def extract_keywords(prompt: str, top_n: int = 5) -> List[str]:
    """
    A simplified keyword extraction function using basic word splitting instead of NLTK tokenization.

    Args:
        prompt (str): The text prompt to extract keywords from
        top_n (int): Maximum number of keywords to return

    Returns:
        List[str]: List of extracted keywords
    """
    words = prompt.lower().split()
    filtered_words = [word for word in words if word.isalnum()]
    word_counts = Counter(filtered_words)
    return [word for word, _ in word_counts.most_common(top_n)]


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1 (List[float]): First vector
        vec2 (List[float]): Second vector

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Calculate norms
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


# Pydantic Models for Logging
class AgentLogInput(BaseModel):
    """
    Input log model for tracking agent task execution.

    Attributes:
        log_id (str): Unique identifier for the log entry
        agent_name (str): Name of the agent executing the task
        task (str): Description of the task being executed
        timestamp (datetime): When the task was started
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), alias="id"
    )
    agent_name: str
    task: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class AgentLogOutput(BaseModel):
    """
    Output log model for tracking agent task completion.

    Attributes:
        log_id (str): Unique identifier for the log entry
        agent_name (str): Name of the agent that completed the task
        result (Any): Result/output from the task execution
        timestamp (datetime): When the task was completed
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), alias="id"
    )
    agent_name: str
    result: Any
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class TreeLog(BaseModel):
    """
    Tree execution log model for tracking tree-level operations.

    Attributes:
        log_id (str): Unique identifier for the log entry
        tree_name (str): Name of the tree that executed the task
        task (str): Description of the task that was executed
        selected_agent (str): Name of the agent selected for the task
        timestamp (datetime): When the task was executed
        result (Any): Result/output from the task execution
    """

    log_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), alias="id"
    )
    tree_name: str
    task: str
    selected_agent: str
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    result: Any


class TreeAgent(Agent):
    """
    A specialized Agent class that contains information about the system prompt's
    locality and allows for dynamic chaining of agents in trees.
    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        system_prompt: str = None,
        model_name: str = "gpt-4.1",
        agent_name: Optional[str] = None,
        embedding_model_name: str = "text-embedding-ada-002",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a TreeAgent with litellm embedding capabilities.

        Args:
            name (str): Name of the agent
            description (str): Description of the agent
            system_prompt (str): System prompt for the agent
            model_name (str): Name of the language model to use
            agent_name (Optional[str]): Alternative name for the agent
            embedding_model_name (str): Name of the embedding model to use
            verbose (bool): Whether to enable verbose logging
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        agent_name = agent_name
        super().__init__(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model_name=model_name,
            agent_name=agent_name,
            *args,
            **kwargs,
        )
        self.embedding_model_name = embedding_model_name
        self.verbose = verbose

        # Generate system prompt embedding using litellm
        if system_prompt:
            self.system_prompt_embedding = self._get_embedding(
                system_prompt
            )
        else:
            self.system_prompt_embedding = None

        # Automatically extract keywords from system prompt
        self.relevant_keywords = (
            extract_keywords(system_prompt) if system_prompt else []
        )

        # Distance is now calculated based on similarity between agents' prompts
        self.distance = None  # Will be dynamically calculated later

    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a given text using litellm.

        Args:
            text (str): Text to embed

        Returns:
            List[float]: Embedding vector
        """
        try:
            response = embedding(
                model=self.embedding_model_name, input=[text]
            )
            if self.verbose:
                logger.info(f"Embedding type: {type(response)}")
            # print(response)
            # Handle different response structures from litellm
            if hasattr(response, "data") and response.data:
                if hasattr(response.data[0], "embedding"):
                    return response.data[0].embedding
                elif (
                    isinstance(response.data[0], dict)
                    and "embedding" in response.data[0]
                ):
                    return response.data[0]["embedding"]
                else:
                    if self.verbose:
                        logger.error(
                            f"Unexpected response structure: {response.data[0]}"
                        )
                    return [0.0] * 1536
            else:
                if self.verbose:
                    logger.error(
                        f"Unexpected response structure: {response}"
                    )
                return [0.0] * 1536
        except Exception as e:
            if self.verbose:
                logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default OpenAI embedding dimension

    def calculate_distance(self, other_agent: "TreeAgent") -> float:
        """
        Calculate the distance between this agent and another agent using embedding similarity.

        Args:
            other_agent (TreeAgent): Another agent in the tree.

        Returns:
            float: Distance score between 0 and 1, with 0 being close and 1 being far.
        """
        if (
            not self.system_prompt_embedding
            or not other_agent.system_prompt_embedding
        ):
            return 1.0  # Maximum distance if embeddings are not available

        similarity = cosine_similarity(
            self.system_prompt_embedding,
            other_agent.system_prompt_embedding,
        )
        distance = (
            1 - similarity
        )  # Closer agents have a smaller distance
        return distance

    def run_task(
        self, task: str, img: str = None, *args, **kwargs
    ) -> Any:
        """
        Execute a task and log the input and output.

        Args:
            task (str): The task to execute
            img (str): Optional image input
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: Result of the task execution
        """
        input_log = AgentLogInput(
            agent_name=self.agent_name,
            task=task,
            timestamp=datetime.now(timezone.utc),
        )
        if self.verbose:
            logger.info(f"Running task on {self.agent_name}: {task}")
            logger.debug(f"Input Log: {input_log.json()}")

        result = self.run(task=task, img=img, *args, **kwargs)

        output_log = AgentLogOutput(
            agent_name=self.agent_name,
            result=result,
            timestamp=datetime.now(timezone.utc),
        )
        if self.verbose:
            logger.info(
                f"Task result from {self.agent_name}: {result}"
            )
            logger.debug(f"Output Log: {output_log.json()}")

        return result

    def is_relevant_for_task(
        self, task: str, threshold: float = 0.7
    ) -> bool:
        """
        Checks if the agent is relevant for the given task using both keyword matching and embedding similarity.

        Args:
            task (str): The task or query for which we need to find a relevant agent.
            threshold (float): The cosine similarity threshold for embedding-based matching.

        Returns:
            bool: True if the agent is relevant, False otherwise.
        """
        # Check if any of the relevant keywords are present in the task (case-insensitive)
        keyword_match = any(
            keyword.lower() in task.lower()
            for keyword in self.relevant_keywords
        )

        # Perform embedding similarity match if keyword match is not found
        if not keyword_match and self.system_prompt_embedding:
            task_embedding = self._get_embedding(task)
            similarity = cosine_similarity(
                self.system_prompt_embedding, task_embedding
            )
            if self.verbose:
                logger.info(
                    f"Semantic similarity between task and {self.agent_name}: {similarity:.2f}"
                )
            return similarity >= threshold

        return True  # Return True if keyword match is found


class Tree:
    def __init__(
        self,
        tree_name: str,
        agents: List[TreeAgent],
        verbose: bool = False,
    ):
        """
        Initializes a tree of agents.

        Args:
            tree_name (str): The name of the tree.
            agents (List[TreeAgent]): A list of agents in the tree.
            verbose (bool): Whether to enable verbose logging
        """
        self.tree_name = tree_name
        self.agents = agents
        self.verbose = verbose
        # Pass verbose to all agents
        for agent in self.agents:
            agent.verbose = verbose
        self.calculate_agent_distances()

    def calculate_agent_distances(self):
        """
        Automatically calculate and assign distances between agents in the tree based on prompt similarity.

        This method computes the semantic distance between consecutive agents using their system prompt
        embeddings and sorts the agents by distance for optimal task routing.
        """
        if self.verbose:
            logger.info(
                f"Calculating distances between agents in tree '{self.tree_name}'"
            )
        for i, agent in enumerate(self.agents):
            if i > 0:
                agent.distance = agent.calculate_distance(
                    self.agents[i - 1]
                )
            else:
                agent.distance = 0  # First agent is closest

        # Sort agents by distance after calculation
        self.agents.sort(key=lambda agent: agent.distance)

    def find_relevant_agent(self, task: str) -> Optional[TreeAgent]:
        """
        Finds the most relevant agent in the tree for the given task based on its system prompt.
        Uses both keyword and semantic similarity matching.

        Args:
            task (str): The task or query for which we need to find a relevant agent.

        Returns:
            Optional[TreeAgent]: The most relevant agent, or None if no match found.
        """
        if self.verbose:
            logger.info(
                f"Searching relevant agent in tree '{self.tree_name}' for task: {task}"
            )
        for agent in self.agents:
            if agent.is_relevant_for_task(task):
                return agent
        if self.verbose:
            logger.warning(
                f"No relevant agent found in tree '{self.tree_name}' for task: {task}"
            )
        return None

    def log_tree_execution(
        self, task: str, selected_agent: TreeAgent, result: Any
    ) -> None:
        """
        Logs the execution details of a tree, including selected agent and result.
        """
        tree_log = TreeLog(
            tree_name=self.tree_name,
            task=task,
            selected_agent=selected_agent.agent_name,
            timestamp=datetime.now(timezone.utc),
            result=result,
        )
        if self.verbose:
            logger.info(
                f"Tree '{self.tree_name}' executed task with agent '{selected_agent.agent_name}'"
            )
            logger.debug(f"Tree Log: {tree_log.json()}")


class ForestSwarm:
    def __init__(
        self,
        name: str = "default-forest-swarm",
        description: str = "Standard forest swarm",
        trees: List[Tree] = [],
        shared_memory: Any = None,
        rules: str = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a ForestSwarm with multiple trees of agents.

        Args:
            name (str): Name of the forest swarm
            description (str): Description of the forest swarm
            trees (List[Tree]): A list of trees in the structure
            shared_memory (Any): Shared memory object for inter-tree communication
            rules (str): Rules governing the forest swarm behavior
            verbose (bool): Whether to enable verbose logging
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        self.name = name
        self.description = description
        self.trees = trees
        self.shared_memory = shared_memory
        self.verbose = verbose
        # Pass verbose to all trees
        for tree in self.trees:
            tree.verbose = verbose
        self.save_file_path = f"forest_swarm_{uuid.uuid4().hex}.json"
        self.conversation = Conversation(
            time_enabled=False,
            save_filepath=self.save_file_path,
            rules=rules,
        )

    def find_relevant_tree(self, task: str) -> Optional[Tree]:
        """
        Find the most relevant tree based on the given task.

        Args:
            task (str): The task or query for which we need to find a relevant tree

        Returns:
            Optional[Tree]: The most relevant tree, or None if no match found
        """
        if self.verbose:
            logger.info(
                f"Searching for the most relevant tree for task: {task}"
            )
        for tree in self.trees:
            if tree.find_relevant_agent(task):
                return tree
        if self.verbose:
            logger.warning(f"No relevant tree found for task: {task}")
        return None

    def run(self, task: str, img: str = None, *args, **kwargs) -> Any:
        """
        Execute the given task by finding the most relevant tree and agent within that tree.

        Args:
            task (str): The task or query to be executed
            img (str): Optional image input for vision-enabled tasks
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Any: The result of the task after it has been processed by the agents
        """
        try:
            if self.verbose:
                logger.info(
                    f"Running task across MultiAgentTreeStructure: {task}"
                )
            relevant_tree = self.find_relevant_tree(task)
            if relevant_tree:
                agent = relevant_tree.find_relevant_agent(task)
                if agent:
                    result = agent.run_task(
                        task, img=img, *args, **kwargs
                    )
                    relevant_tree.log_tree_execution(
                        task, agent, result
                    )
                    return result
            else:
                if self.verbose:
                    logger.error(
                        "Task could not be completed: No relevant agent or tree found."
                    )
                return "No relevant agent found to handle this task."
        except Exception as error:
            if self.verbose:
                logger.error(
                    f"Error detected in the ForestSwarm, check your inputs and try again ;) {error}"
                )

    def batched_run(
        self,
        tasks: List[str],
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Execute the given tasks by finding the most relevant tree and agent within that tree.

        Args:
            tasks: List[str]: The tasks to be executed
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        return [self.run(task, *args, **kwargs) for task in tasks]


# # Example Usage:

# # Create agents with varying system prompts and dynamically generated distances/keywords
# agents_tree1 = [
#     TreeAgent(
#         system_prompt="Stock Analysis Agent",
#         agent_name="Stock Analysis Agent",
#     ),
#     TreeAgent(
#         system_prompt="Financial Planning Agent",
#         agent_name="Financial Planning Agent",
#     ),
#     TreeAgent(
#         agent_name="Retirement Strategy Agent",
#         system_prompt="Retirement Strategy Agent",
#     ),
# ]

# agents_tree2 = [
#     TreeAgent(
#         system_prompt="Tax Filing Agent",
#         agent_name="Tax Filing Agent",
#     ),
#     TreeAgent(
#         system_prompt="Investment Strategy Agent",
#         agent_name="Investment Strategy Agent",
#     ),
#     TreeAgent(
#         system_prompt="ROTH IRA Agent", agent_name="ROTH IRA Agent"
#     ),
# ]

# # Create trees
# tree1 = Tree(tree_name="Financial Tree", agents=agents_tree1)
# tree2 = Tree(tree_name="Investment Tree", agents=agents_tree2)

# # Create the ForestSwarm
# multi_agent_structure = ForestSwarm(trees=[tree1, tree2])

# # Run a task
# task = "Our company is incorporated in delaware, how do we do our taxes for free?"
# output = multi_agent_structure.run(task)
# print(output)
