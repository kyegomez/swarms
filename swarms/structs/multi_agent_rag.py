from dataclasses import dataclass
from typing import List, Optional

from swarms.memory.base_vectordatabase import AbstractVectorDatabase
from swarms.structs.agent import Agent


@dataclass
class MultiAgentRag:
    """
    Represents a multi-agent RAG (Relational Agent Graph) structure.
    
    Attributes:
        agents (List[Agent]): List of agents in the multi-agent RAG.
        db (AbstractVectorDatabase): Database used for querying.
        verbose (bool): Flag indicating whether to print verbose output.
    """
    agents: List[Agent]
    db: AbstractVectorDatabase
    verbose: bool = False
    
    
    def query_database(self, query: str):
        """
        Queries the database using the given query string.
        
        Args:
            query (str): The query string.
        
        Returns:
            List: The list of results from the database.
        """
        results = []
        for agent in self.agents:
            agent_results = agent.long_term_memory_prompt(query)
            results.extend(agent_results)
        return results
    
    
    def get_agent_by_id(self, agent_id) -> Optional[Agent]:
        """
        Retrieves an agent from the multi-agent RAG by its ID.
        
        Args:
            agent_id: The ID of the agent to retrieve.
        
        Returns:
            Agent or None: The agent with the specified ID, or None if not found.
        """
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def add_message(
        self,
        sender: Agent,
        message: str,
        *args,
        **kwargs
    ):
        """
        Adds a message to the database.
        
        Args:
            sender (Agent): The agent sending the message.
            message (str): The message to add.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            int: The ID of the added message.
        """
        doc = f"{sender.ai_name}: {message}"
        
        return self.db.add(doc)
    
    def query(
        self,
        message: str,
        *args,
        **kwargs
    ):
        """
        Queries the database using the given message.
        
        Args:
            message (str): The message to query.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            List: The list of results from the database.
        """
        return self.db.query(message)
    
    
