from unittest.mock import Mock, MagicMock
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from datetime import datetime
import unittest
from swarms.schemas.agent_step_schemas import ManySteps, Step
from swarms.structs.agent import Agent

class TestAgentLogging(unittest.TestCase):
    def setUp(self):
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.count_tokens.return_value = 100
        
        self.mock_short_memory = MagicMock()
        self.mock_short_memory.get_memory_stats.return_value = {"message_count": 2}
        
        self.mock_long_memory = MagicMock()
        self.mock_long_memory.get_memory_stats.return_value = {"item_count": 5}
        
        self.agent = Agent(
            tokenizer=self.mock_tokenizer,
            short_memory=self.mock_short_memory,
            long_term_memory=self.mock_long_memory
        )
