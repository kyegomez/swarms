import os
from typing import List, Optional

try:
    from feedo.memory import FeedoMemory
except ImportError:
    raise ImportError(
        "Please install the Feedo SDK to use this tool: "
        "`pip install feedo-sdk>=0.1.21`"
    )

class FeedoMemoryTools:
    """
    Provides tool functions for Swarms agents to interact with the 
    Feedo decentralized E2EE memory network.
    
    https://feedo.ink
    """
    def __init__(self, usage_key: str = None, private: bool = True):
        usage_key = usage_key or os.getenv("FEEDO_USAGE_KEY")
        if not usage_key:
            raise ValueError(
                "FEEDO_USAGE_KEY is required to initialize Feedo memory.\n"
                "You can generate a free testnet usage key at: https://feedo.ink"
            )
            
        self.memory = FeedoMemory(usage_key=usage_key, private=private)

    def add_memory(self, text: str, topic: Optional[str] = None) -> str:
        """
        Store important context or information in the decentralized long-term memory.
        
        Args:
            text: The information to remember.
            topic: An optional category or topic for the memory.
            
        Returns:
            A confirmation string with the memory ID.
        """
        metadata = {"topic": topic} if topic else {}
        mem_id = self.memory.add(text, metadata=metadata)
        return f"Successfully saved to Feedo memory with ID: {mem_id}"

    def search_memory(self, query: str) -> str:
        """
        Search the decentralized long-term memory for relevant past information.
        
        Args:
            query: The search query or question to look up in memory.
            
        Returns:
            A formatted string containing the relevant memory results.
        """
        results = self.memory.search(query, limit=5)
        if not results:
            return "No relevant memories found."
        
        return "\n".join([f"- {r.get('text')}" for r in results])
        
    def update_memory(self, memory_id: str, text: str) -> str:
        """
        Updates an existing memory entry by its ID.
        
        Args:
            memory_id: The ID of the memory to update.
            text: The new text to replace the old memory.
            
        Returns:
            A confirmation string with the new memory ID.
        """
        new_id = self.memory.update(memory_id, text)
        return f"Memory successfully updated. New ID: {new_id}"

    def delete_memory(self, memory_id: str) -> str:
        """
        Deletes a specific memory from the decentralized network.
        
        Args:
            memory_id: The ID of the memory to delete.
            
        Returns:
            A confirmation string.
        """
        self.memory.delete(memory_id)
        return f"Memory {memory_id} successfully deleted."

    def get_tools(self) -> List[callable]:
        """Returns the list of callable tools to pass to a Swarms Agent."""
        return [self.add_memory, self.search_memory, self.update_memory, self.delete_memory]
