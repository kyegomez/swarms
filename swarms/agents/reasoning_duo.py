from typing import List, Optional, Dict
from loguru import logger
from swarms.prompts.reasoning_prompt import REASONING_PROMPT
from swarms.structs.agent import Agent
from swarms.utils.output_types import OutputType
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
import uuid
import time
import threading

class ReasoningDuo:
    """
    ReasoningDuo is a class that encapsulates the functionality of two agents: a reasoning agent and a main agent.
    
    Implements an agent pool mechanism to reuse agent instances between calls.

    Attributes:
        model_name (str): The name of the model used for the reasoning agent.
        description (str): A description of the reasoning agent.
        model_names (list[str]): A list of model names for the agents.
        system_prompt (str): The system prompt for the main agent.
        reasoning_agent (Agent): An instance of the Agent class for reasoning tasks.
        main_agent (Agent): An instance of the Agent class for main tasks.
    """
    
    # Class-level agent pools
    _reasoning_agent_pool: Dict[str, Dict] = {}  # key: config_key, value: {agent, last_used, in_use}
    _main_agent_pool: Dict[str, Dict] = {}  # key: config_key, value: {agent, last_used, in_use}
    _pool_lock = threading.RLock()  # Thread-safe lock for pool access
    _pool_cleanup_interval = 300  # 5 minutes
    _pool_max_idle_time = 1800  # 30 minutes
    _last_cleanup_time = 0
    
    @classmethod
    def _generate_agent_config_key(cls, agent_type, model_name, system_prompt, **kwargs):
        """Generate a unique key for an agent configuration"""
        # Include essential parameters that affect agent behavior
        key_parts = [
            agent_type,
            model_name,
            system_prompt
        ]
        # Add other important configuration parameters
        for k in sorted(kwargs.keys()):
            if k in ['max_loops', 'dynamic_temperature_enabled', 'streaming', 'output_type']:
                key_parts.append(f"{k}={kwargs[k]}")
        
        return ":".join(str(part) for part in key_parts)
    
    @classmethod
    def _get_agent_from_pool(cls, agent_type, config_key, create_func):
        """Get an agent from the pool or create a new one if needed"""
        with cls._pool_lock:
            pool = cls._reasoning_agent_pool if agent_type == "reasoning" else cls._main_agent_pool
            
            # Periodic cleanup of idle agents
            current_time = time.time()
            if current_time - cls._last_cleanup_time > cls._pool_cleanup_interval:
                cls._cleanup_idle_agents()
                cls._last_cleanup_time = current_time
            
            # Try to find an available agent with matching configuration
            if config_key in pool and not pool[config_key]["in_use"]:
                pool[config_key]["in_use"] = True
                pool[config_key]["last_used"] = time.time()
                logger.debug(f"Reusing {agent_type} agent from pool with config: {config_key}")
                return pool[config_key]["agent"]
            
            # Create a new agent if none available
            logger.debug(f"Creating new {agent_type} agent with config: {config_key}")
            new_agent = create_func()
            pool[config_key] = {
                "agent": new_agent,
                "last_used": time.time(),
                "in_use": True
            }
            return new_agent
    
    @classmethod
    def _release_agent_to_pool(cls, agent_type, config_key):
        """Release an agent back to the pool"""
        with cls._pool_lock:
            pool = cls._reasoning_agent_pool if agent_type == "reasoning" else cls._main_agent_pool
            if config_key in pool:
                pool[config_key]["in_use"] = False
                pool[config_key]["last_used"] = time.time()
                logger.debug(f"Released {agent_type} agent back to pool: {config_key}")
    
    @classmethod
    def _cleanup_idle_agents(cls):
        """Clean up agents that have been idle for too long"""
        with cls._pool_lock:
            current_time = time.time()
            
            for pool in [cls._reasoning_agent_pool, cls._main_agent_pool]:
                keys_to_remove = []
                
                for key, data in pool.items():
                    # Only remove if not in use and idle for too long
                    if not data["in_use"] and (current_time - data["last_used"] > cls._pool_max_idle_time):
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    logger.debug(f"Removing idle agent from pool: {key}")
                    del pool[key]
    
    @classmethod
    def configure_pool(cls, cleanup_interval=None, max_idle_time=None):
        """Configure the agent pool parameters"""
        with cls._pool_lock:
            if cleanup_interval is not None:
                cls._pool_cleanup_interval = max(60, cleanup_interval)  # Minimum 1 minute
            if max_idle_time is not None:
                cls._pool_max_idle_time = max(300, max_idle_time)  # Minimum 5 minutes
    
    @classmethod
    def clear_pools(cls):
        """Clear all agent pools (useful for testing or memory management)"""
        with cls._pool_lock:
            cls._reasoning_agent_pool.clear()
            cls._main_agent_pool.clear()

    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        agent_name: str = "reasoning-agent-01",
        agent_description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_name: str = "gpt-4o-mini",
        description: str = "A highly intelligent and thoughtful AI designed to provide accurate and well-reasoned answers to the user's questions.",
        model_names: list[str] = ["gpt-4o-mini", "gpt-4.1"],
        system_prompt: str = "You are a helpful assistant that can answer questions and help with tasks.",
        output_type: OutputType = "dict-all-except-first",
        reasoning_model_name: Optional[str] = "gpt-4o",
        max_loops: int = 1,
        reuse_agents: bool = True,  # New parameter to control agent reuse
        *args,
        **kwargs,
    ):
        self.id = id
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.model_name = model_name
        self.description = description
        self.output_type = output_type
        self.reasoning_model_name = reasoning_model_name
        self.max_loops = max_loops
        self.reuse_agents = reuse_agents
        self.args = args
        self.kwargs = kwargs

        if self.reasoning_model_name is None:
            self.reasoning_model_name = model_names[0]

        self.conversation = Conversation()
        
        # Create a complete configuration for the reasoning agent
        reasoning_full_config = {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "system_prompt": REASONING_PROMPT,  
            "max_loops": 1,
            "model_name": self.reasoning_model_name,
            "dynamic_temperature_enabled": True
        }
        
        # Create a complete configuration for the main agent
        main_full_config = {
            "agent_name": self.agent_name,
            "description": self.agent_description,
            "system_prompt": system_prompt,
            "max_loops": 1,
            "model_name": model_names[1],
            "dynamic_temperature_enabled": True
        }
        
        
        for k, v in kwargs.items():
            if k not in ["system_prompt", "model_name"]:  
                reasoning_full_config[k] = v
                main_full_config[k] = v
        
        # To generate the configuration keys we need to extract the parameters (excluding those that have been explicitly passed)
        reasoning_extra_params = {
            k: v for k, v in reasoning_full_config.items() 
            if k not in ["system_prompt", "model_name"]
        }
        
        main_extra_params = {
            k: v for k, v in main_full_config.items() 
            if k not in ["system_prompt", "model_name"]
        }
        
        # generate the configuration keys
        self.reasoning_config_key = self._generate_agent_config_key(
            "reasoning", 
            self.reasoning_model_name, 
            REASONING_PROMPT,
            **reasoning_extra_params
        )
        
        self.main_config_key = self._generate_agent_config_key(
            "main", 
            model_names[1], 
            system_prompt,
            **main_extra_params
        )
        
        # Get the agent instance
        if self.reuse_agents:
            self.reasoning_agent = self._get_agent_from_pool(
                "reasoning", 
                self.reasoning_config_key,
                lambda: Agent(**reasoning_full_config)
            )
            
            self.main_agent = self._get_agent_from_pool(
                "main", 
                self.main_config_key,
                lambda: Agent(**main_full_config)
            )
        else:
            # If reuse is disabled, create a new agent directly
            self.reasoning_agent = Agent(**reasoning_full_config)
            self.main_agent = Agent(**main_full_config)

    def __del__(self):
        """Release agents back to the pool when instance is destroyed"""
        if hasattr(self, 'reuse_agents') and self.reuse_agents:
            if hasattr(self, 'reasoning_config_key'):
                self._release_agent_to_pool("reasoning", self.reasoning_config_key)
            if hasattr(self, 'main_config_key'):
                self._release_agent_to_pool("main", self.main_config_key)

    def step(self, task: str, img: Optional[str] = None):
        """
        Executes one step of reasoning and main agent processing.

        Args:
            task (str): The task to be processed.
            img (Optional[str]): Optional image input.
        """
        # For reasoning agent, use the current task (which may include conversation context)
        output_reasoner = self.reasoning_agent.run(task, img=img)
        self.conversation.add(
            role=self.reasoning_agent.agent_name,
            content=output_reasoner,
        )

        # For main agent, always use the full conversation context
        output_main = self.main_agent.run(
            task=self.conversation.get_str(), img=img
        )
        self.conversation.add(
            role=self.main_agent.agent_name, content=output_main
        )

    def run(self, task: str, img: Optional[str] = None):
        """
        Executes the reasoning and main agents on the provided task.

        Args:
            task (str): The task to be processed by the agents.
            img (Optional[str]): Optional image input.

        Returns:
            str: The output from the main agent after processing the task.
        """
        logger.info(
            f"Running task: {task} with max_loops: {self.max_loops}"
        )

        self.conversation.add(role="user", content=task)

        for loop_iteration in range(self.max_loops):
            logger.info(
                f"Loop iteration {loop_iteration + 1}/{self.max_loops}"
            )

            if loop_iteration == 0:
                # First iteration: use original task
                current_task = task
            else:
                # Subsequent iterations: use task with context of previous reasoning
                current_task = f"Continue reasoning and refining your analysis. Original task: {task}\n\nPrevious conversation context:\n{self.conversation.get_str()}"

            self.step(task=current_task, img=img)

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def batched_run(
        self, tasks: List[str], imgs: Optional[List[str]] = None
    ):
        """
        Executes the run method for a list of tasks.

        Args:
            tasks (list[str]): A list of tasks to be processed.
            imgs (Optional[List[str]]): Optional list of images corresponding to tasks.

        Returns:
            list: A list of outputs from the main agent for each task.
        """
        outputs = []

        # Handle case where imgs is None
        if imgs is None:
            imgs = [None] * len(tasks)

        for task, img in zip(tasks, imgs):
            logger.info(f"Processing task: {task}")
            outputs.append(self.run(task, img=img))

        return outputs