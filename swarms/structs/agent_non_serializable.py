"""
Non-Serializable Properties Handler for Agent

This module provides helper functions to save and restore non-serializable properties
(tokenizer, long_term_memory, logger_handler, agent_output, executor) for the Agent class.

Usage:
    from swarms.structs.agent_non_serializable import restore_non_serializable_properties
    restore_non_serializable_properties(agent)
"""

from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
import logging

# Dummy/placeholder for long_term_memory and agent_output restoration
class DummyLongTermMemory:
    def __init__(self):
        self.memory = []
    def query(self, *args, **kwargs):
        # Return an empty list or a default value to avoid errors
        return []
    def save(self, path):
        # Optionally implement a no-op save for compatibility
        pass

class DummyAgentOutput:
    def __init__(self):
        self.output = None

def restore_non_serializable_properties(agent):
    """
    Restore non-serializable properties for the Agent instance after loading.
    This should be called after loading agent state from disk.
    """
    # Restore tokenizer if model_name is available
    if getattr(agent, "model_name", None):
        try:
            agent.tokenizer = AutoTokenizer.from_pretrained(agent.model_name)
        except Exception:
            agent.tokenizer = None
    else:
        agent.tokenizer = None

    # Restore long_term_memory (dummy for demo, replace with real backend as needed)
    if getattr(agent, "long_term_memory", None) is None or not hasattr(agent.long_term_memory, "query"):
        agent.long_term_memory = DummyLongTermMemory()

    # Restore logger_handler
    try:
        agent.logger_handler = logging.StreamHandler()
    except Exception:
        agent.logger_handler = None

    # Restore agent_output (dummy for demo, replace with real backend as needed)
    agent.agent_output = DummyAgentOutput()

    # Restore executor
    try:
        agent.executor = ThreadPoolExecutor()
    except Exception:
        agent.executor = None

    return agent
