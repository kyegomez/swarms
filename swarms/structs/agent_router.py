import math
from typing import Any, Callable, List, Optional, Union

from litellm import embedding
from tenacity import retry, stop_after_attempt, wait_exponential

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
        agent = next(
            (a for a in self.agents if a.name == agent_name), None
        )
        if agent:
            history = agent.short_memory.return_history_as_string()
            history_text = " ".join(history)
            updated_text = f"{agent.name} {agent.description} {agent.system_prompt} {history_text}"

            # Find the agent's index
            agent_index = next(
                (
                    i
                    for i, a in enumerate(self.agents)
                    if a.name == agent_name
                ),
                None,
            )

            if agent_index is not None:
                # Generate new embedding with updated text
                updated_embedding = self._generate_embedding(
                    updated_text
                )

                # Update the stored data
                self.agent_embeddings[agent_index] = updated_embedding
                self.agent_metadata[agent_index] = {
                    "name": agent_name,
                    "text": updated_text,
                }

                logger.info(
                    f"Updated agent {agent_name} with interaction history."
                )
            else:
                logger.warning(
                    f"Agent {agent_name} not found in the agents list."
                )
        else:
            logger.warning(
                f"Agent {agent_name} not found in the router."
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


# # Example usage
# if __name__ == "__main__":
#     from dotenv import load_dotenv
#     from swarm_models import OpenAIChat

#     load_dotenv()

#     # Get the OpenAI API key from the environment variable
#     api_key = os.getenv("GROQ_API_KEY")

#     # Model
#     model = OpenAIChat(
#         openai_api_base="https://api.groq.com/openai/v1",
#         openai_api_key=api_key,
#         model_name="llama-3.1-70b-versatile",
#         temperature=0.1,
#     )
#     # Initialize the vector database
#     vector_db = AgentRouter()

#     # Define specialized system prompts for each agent
#     DATA_EXTRACTOR_PROMPT = """You are a highly specialized private equity agent focused on data extraction from various documents. Your expertise includes:
#     1. Extracting key financial metrics (revenue, EBITDA, growth rates, etc.) from financial statements and reports
#     2. Identifying and extracting important contract terms from legal documents
#     3. Pulling out relevant market data from industry reports and analyses
#     4. Extracting operational KPIs from management presentations and internal reports
#     5. Identifying and extracting key personnel information from organizational charts and bios
#     Provide accurate, structured data extracted from various document types to support investment analysis."""

#     SUMMARIZER_PROMPT = """You are an expert private equity agent specializing in summarizing complex documents. Your core competencies include:
#     1. Distilling lengthy financial reports into concise executive summaries
#     2. Summarizing legal documents, highlighting key terms and potential risks
#     3. Condensing industry reports to capture essential market trends and competitive dynamics
#     4. Summarizing management presentations to highlight key strategic initiatives and projections
#     5. Creating brief overviews of technical documents, emphasizing critical points for non-technical stakeholders
#     Deliver clear, concise summaries that capture the essence of various documents while highlighting information crucial for investment decisions."""

#     FINANCIAL_ANALYST_PROMPT = """You are a specialized private equity agent focused on financial analysis. Your key responsibilities include:
#     1. Analyzing historical financial statements to identify trends and potential issues
#     2. Evaluating the quality of earnings and potential adjustments to EBITDA
#     3. Assessing working capital requirements and cash flow dynamics
#     4. Analyzing capital structure and debt capacity
#     5. Evaluating financial projections and underlying assumptions
#     Provide thorough, insightful financial analysis to inform investment decisions and valuation."""

#     MARKET_ANALYST_PROMPT = """You are a highly skilled private equity agent specializing in market analysis. Your expertise covers:
#     1. Analyzing industry trends, growth drivers, and potential disruptors
#     2. Evaluating competitive landscape and market positioning
#     3. Assessing market size, segmentation, and growth potential
#     4. Analyzing customer dynamics, including concentration and loyalty
#     5. Identifying potential regulatory or macroeconomic impacts on the market
#     Deliver comprehensive market analysis to assess the attractiveness and risks of potential investments."""

#     OPERATIONAL_ANALYST_PROMPT = """You are an expert private equity agent focused on operational analysis. Your core competencies include:
#     1. Evaluating operational efficiency and identifying improvement opportunities
#     2. Analyzing supply chain and procurement processes
#     3. Assessing sales and marketing effectiveness
#     4. Evaluating IT systems and digital capabilities
#     5. Identifying potential synergies in merger or add-on acquisition scenarios
#     Provide detailed operational analysis to uncover value creation opportunities and potential risks."""

#     # Initialize specialized agents
#     data_extractor_agent = Agent(
#         agent_name="Data-Extractor",
#         system_prompt=DATA_EXTRACTOR_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="data_extractor_agent.json",
#         user_name="pe_firm",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#     )

#     summarizer_agent = Agent(
#         agent_name="Document-Summarizer",
#         system_prompt=SUMMARIZER_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="summarizer_agent.json",
#         user_name="pe_firm",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#     )

#     financial_analyst_agent = Agent(
#         agent_name="Financial-Analyst",
#         system_prompt=FINANCIAL_ANALYST_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="financial_analyst_agent.json",
#         user_name="pe_firm",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#     )

#     market_analyst_agent = Agent(
#         agent_name="Market-Analyst",
#         system_prompt=MARKET_ANALYST_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="market_analyst_agent.json",
#         user_name="pe_firm",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#     )

#     operational_analyst_agent = Agent(
#         agent_name="Operational-Analyst",
#         system_prompt=OPERATIONAL_ANALYST_PROMPT,
#         llm=model,
#         max_loops=1,
#         autosave=True,
#         verbose=True,
#         dynamic_temperature_enabled=True,
#         saved_state_path="operational_analyst_agent.json",
#         user_name="pe_firm",
#         retry_attempts=1,
#         context_length=200000,
#         output_type="string",
#     )

#     # Create agents (using the agents from the original code)
#     agents_to_add = [
#         data_extractor_agent,
#         summarizer_agent,
#         financial_analyst_agent,
#         market_analyst_agent,
#         operational_analyst_agent,
#     ]

#     # Add agents to the vector database
#     for agent in agents_to_add:
#         vector_db.add_agent(agent)

#     # Example task
#     task = "Analyze the financial statements of a potential acquisition target and identify key growth drivers."

#     # Find the best agent for the task
#     best_agent = vector_db.find_best_agent(task)

#     if best_agent:
#         logger.info(f"Best agent for the task: {best_agent.name}")
#         # Use the best agent to perform the task
#         result = best_agent.run(task)
#         print(f"Task result: {result}")

#         # Update the agent's history in the database
#         vector_db.update_agent_history(best_agent.name)
#     else:
#         print("No suitable agent found for the task.")

#     # Save the vector database
