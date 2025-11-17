# Building Custom Swarms: A Comprehensive Guide for Swarm Engineers

## Introduction

As artificial intelligence and machine learning continue to grow in complexity and applicability, building systems that can harness multiple agents to solve complex tasks becomes more critical. Swarm engineering enables AI agents to collaborate and solve problems autonomously in diverse fields such as finance, marketing, operations, and even creative industries.

This comprehensive guide covers how to build a custom swarm system that integrates multiple agents into a cohesive system capable of solving tasks collaboratively. We'll cover everything from basic swarm structure to advanced features like conversation management, logging, error handling, and scalability.

By the end of this guide, you will have a complete understanding of:

- What swarms are and how they can be built

- How to create agents and integrate them into swarms

- How to implement proper conversation management for message storage

- Best practices for error handling, logging, and optimization

- How to make swarms scalable and production-ready


---

## Overview of Swarm Architecture

A **Swarm** refers to a collection of agents that collaborate to solve a problem. Each agent in the swarm performs part of the task, either independently or by communicating with other agents. Swarms are ideal for:

- **Scalability**: You can add or remove agents dynamically based on the task's complexity

- **Flexibility**: Each agent can be designed to specialize in different parts of the problem, offering modularity

- **Autonomy**: Agents in a swarm can operate autonomously, reducing the need for constant supervision

- **Conversation Management**: All interactions are tracked and stored for analysis and continuity


---

## Core Requirements for Swarm Classes

Every Swarm class must adhere to these fundamental requirements:

### Required Methods and Attributes

- **`run(task: str, img: str, *args, **kwargs)` method**: The primary execution method for tasks

- **`name`**: A descriptive name for the swarm

- **`description`**: A clear description of the swarm's purpose

- **`agents`**: A list of callables representing the agents

- **`conversation`**: A conversation structure for message storage and history management


### Required Agent Structure

Each Agent within the swarm must contain:

- **`agent_name`**: Unique identifier for the agent

- **`system_prompt`**: Instructions that guide the agent's behavior

- **`run` method**: Method to execute tasks assigned to the agent


---

## Setting Up the Foundation

### Required Dependencies

```python
from typing import List, Union, Any, Optional, Callable
from loguru import logger
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.structs.agent import Agent
import concurrent.futures
import os
```

### Custom Exception Handling

```python
class SwarmExecutionError(Exception):
    """Custom exception for handling swarm execution errors."""
    pass

class AgentValidationError(Exception):
    """Custom exception for agent validation errors."""
    pass
```

---

## Building the Custom Swarm Class

### Basic Swarm Structure

```python
class CustomSwarm(BaseSwarm):
    """
    A custom swarm class to manage and execute tasks with multiple agents.
    
    This swarm integrates conversation management for tracking all agent interactions,
    provides error handling, and supports both sequential and concurrent execution.

    Attributes:
        name (str): The name of the swarm.
        description (str): A brief description of the swarm's purpose.
        agents (List[Callable]): A list of callables representing the agents.
        conversation (Conversation): Conversation management for message storage.
        max_workers (int): Maximum number of concurrent workers for parallel execution.
        autosave_conversation (bool): Whether to automatically save conversation history.
    """

    def __init__(
        self,
        name: str,
        description: str,
        agents: List[Callable],
        max_workers: int = 4,
        autosave_conversation: bool = True,
        conversation_config: Optional[dict] = None,
    ):
        """
        Initialize the CustomSwarm with its name, description, and agents.

        Args:
            name (str): The name of the swarm.
            description (str): A description of the swarm.
            agents (List[Callable]): A list of callables that provide the agents for the swarm.
            max_workers (int): Maximum number of concurrent workers.
            autosave_conversation (bool): Whether to automatically save conversations.
            conversation_config (dict): Configuration for conversation management.
        """
        super().__init__(name=name, description=description, agents=agents)
        self.name = name
        self.description = description
        self.agents = agents
        self.max_workers = max_workers
        self.autosave_conversation = autosave_conversation
        
        # Initialize conversation management
        # See: https://docs.swarms.world/swarms/structs/conversation/
        conversation_config = conversation_config or {}
        self.conversation = Conversation(
            id=f"swarm_{name}_{int(time.time())}",
            name=f"{name}_conversation",
            autosave=autosave_conversation,
            save_enabled=True,
            time_enabled=True,
            **conversation_config
        )
        
        # Validate agents and log initialization
        self.validate_agents()
        logger.info(f"🚀 CustomSwarm '{self.name}' initialized with {len(self.agents)} agents")
        
        # Add swarm initialization to conversation history
        self.conversation.add(
            role="System",
            content=f"Swarm '{self.name}' initialized with {len(self.agents)} agents: {[getattr(agent, 'agent_name', 'Unknown') for agent in self.agents]}"
        )

    def validate_agents(self):
        """
        Validates that each agent has the required methods and attributes.
        
        Raises:
            AgentValidationError: If any agent fails validation.
        """
        for i, agent in enumerate(self.agents):
            # Check for required run method
            if not hasattr(agent, 'run'):
                raise AgentValidationError(f"Agent at index {i} does not have a 'run' method.")
            
            # Check for agent_name attribute
            if not hasattr(agent, 'agent_name'):
                logger.warning(f"Agent at index {i} does not have 'agent_name' attribute. Using 'Agent_{i}'")
                agent.agent_name = f"Agent_{i}"
            
            logger.info(f"✅ Agent '{agent.agent_name}' validated successfully.")

    def run(self, task: str, img: str = None, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a task using the swarm and its agents with conversation tracking.

        Args:
            task (str): The task description.
            img (str): The image input (optional).
            *args: Additional positional arguments for customization.
            **kwargs: Additional keyword arguments for fine-tuning behavior.

        Returns:
            Any: The result of the task execution, aggregated from all agents.
        """
        logger.info(f"🎯 Running task '{task}' across {len(self.agents)} agents in swarm '{self.name}'")
        
        # Add task to conversation history
        self.conversation.add(
            role="User",
            content=f"Task: {task}" + (f" | Image: {img}" if img else ""),
            category="input"
        )
        
        try:
            # Execute task across all agents
            results = self._execute_agents(task, img, *args, **kwargs)
            
            # Add results to conversation
            self.conversation.add(
                role="Swarm",
                content=f"Task completed successfully. Processed by {len(results)} agents.",
                category="output"
            )
            
            logger.success(f"✅ Task completed successfully by swarm '{self.name}'")
            return results
            
        except Exception as e:
            error_msg = f"❌ Task execution failed in swarm '{self.name}': {str(e)}"
            logger.error(error_msg)
            
            # Add error to conversation
            self.conversation.add(
                role="System",
                content=f"Error: {error_msg}",
                category="error"
            )
            
            raise SwarmExecutionError(error_msg)

    def _execute_agents(self, task: str, img: str = None, *args, **kwargs) -> List[Any]:
        """
        Execute the task across all agents with proper conversation tracking.
        
        Args:
            task (str): The task to execute.
            img (str): Optional image input.
            
        Returns:
            List[Any]: Results from all agents.
        """
        results = []
        
        for agent in self.agents:
            try:
                # Execute agent task
                result = agent.run(task, img, *args, **kwargs)
                results.append(result)
                
                # Add agent response to conversation
                self.conversation.add(
                    role=agent.agent_name,
                    content=result,
                    category="agent_output"
                )
                
                logger.info(f"✅ Agent '{agent.agent_name}' completed task successfully")
                
            except Exception as e:
                error_msg = f"Agent '{agent.agent_name}' failed: {str(e)}"
                logger.error(error_msg)
                
                # Add agent error to conversation
                self.conversation.add(
                    role=agent.agent_name,
                    content=f"Error: {error_msg}",
                    category="agent_error"
                )
                
                # Continue with other agents but log the failure
                results.append(f"FAILED: {error_msg}")
        
        return results
```

### Enhanced Swarm with Concurrent Execution

```python
    def run_concurrent(self, task: str, img: str = None, *args: Any, **kwargs: Any) -> List[Any]:
        """
        Execute a task using concurrent execution for better performance.
        
        Args:
            task (str): The task description.
            img (str): The image input (optional).
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            List[Any]: Results from all agents executed concurrently.
        """
        logger.info(f"🚀 Running task concurrently across {len(self.agents)} agents")
        
        # Add task to conversation
        self.conversation.add(
            role="User",
            content=f"Concurrent Task: {task}" + (f" | Image: {img}" if img else ""),
            category="input"
        )
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(self._run_single_agent, agent, task, img, *args, **kwargs): agent
                for agent in self.agents
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Add to conversation
                    self.conversation.add(
                        role=agent.agent_name,
                        content=result,
                        category="agent_output"
                    )
                    
                except Exception as e:
                    error_msg = f"Concurrent execution failed for agent '{agent.agent_name}': {str(e)}"
                    logger.error(error_msg)
                    results.append(f"FAILED: {error_msg}")
                    
                    # Add error to conversation
                    self.conversation.add(
                        role=agent.agent_name,
                        content=f"Error: {error_msg}",
                        category="agent_error"
                    )
        
        # Add completion summary
        self.conversation.add(
            role="Swarm",
            content=f"Concurrent task completed. {len(results)} agents processed.",
            category="output"
        )
        
        return results

    def _run_single_agent(self, agent: Callable, task: str, img: str = None, *args, **kwargs) -> Any:
        """
        Execute a single agent with error handling.
        
        Args:
            agent: The agent to execute.
            task (str): The task to execute.
            img (str): Optional image input.
            
        Returns:
            Any: The agent's result.
        """
        try:
            return agent.run(task, img, *args, **kwargs)
        except Exception as e:
            logger.error(f"Agent '{getattr(agent, 'agent_name', 'Unknown')}' execution failed: {str(e)}")
            raise
```

### Advanced Features

```python
    def run_with_retries(self, task: str, img: str = None, retries: int = 3, *args, **kwargs) -> List[Any]:
        """
        Execute a task with retry logic for failed agents.
        
        Args:
            task (str): The task to execute.
            img (str): Optional image input.
            retries (int): Number of retries for failed agents.
            
        Returns:
            List[Any]: Results from all agents with retry attempts.
        """
        logger.info(f"🔄 Running task with {retries} retries per agent")
        
        # Add task to conversation
        self.conversation.add(
            role="User",
            content=f"Task with retries ({retries}): {task}",
            category="input"
        )
        
        results = []
        
        for agent in self.agents:
            attempt = 0
            success = False
            
            while attempt <= retries and not success:
                try:
                    result = agent.run(task, img, *args, **kwargs)
                    results.append(result)
                    success = True
                    
                    # Add successful result to conversation
                    self.conversation.add(
                        role=agent.agent_name,
                        content=result,
                        category="agent_output"
                    )
                    
                    if attempt > 0:
                        logger.success(f"✅ Agent '{agent.agent_name}' succeeded on attempt {attempt + 1}")
                    
                except Exception as e:
                    attempt += 1
                    error_msg = f"Agent '{agent.agent_name}' failed on attempt {attempt}: {str(e)}"
                    logger.warning(error_msg)
                    
                    # Add retry attempt to conversation
                    self.conversation.add(
                        role=agent.agent_name,
                        content=f"Retry attempt {attempt}: {error_msg}",
                        category="agent_retry"
                    )
                    
                    if attempt > retries:
                        final_error = f"Agent '{agent.agent_name}' exhausted all {retries} retries"
                        logger.error(final_error)
                        results.append(f"FAILED: {final_error}")
                        
                        # Add final failure to conversation
                        self.conversation.add(
                            role=agent.agent_name,
                            content=final_error,
                            category="agent_error"
                        )
        
        return results

    def get_conversation_summary(self) -> dict:
        """
        Get a summary of the conversation history and agent performance.
        
        Returns:
            dict: Summary of conversation statistics and agent performance.
        """
        # Get conversation statistics
        message_counts = self.conversation.count_messages_by_role()
        
        # Count categories
        category_counts = {}
        for message in self.conversation.conversation_history:
            category = message.get("category", "uncategorized")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get token counts if available
        token_summary = self.conversation.export_and_count_categories()
        
        return {
            "swarm_name": self.name,
            "total_messages": len(self.conversation.conversation_history),
            "messages_by_role": message_counts,
            "messages_by_category": category_counts,
            "token_summary": token_summary,
            "conversation_id": self.conversation.id,
        }

    def export_conversation(self, filepath: str = None) -> str:
        """
        Export the conversation history to a file.
        
        Args:
            filepath (str): Optional custom filepath for export.
            
        Returns:
            str: The filepath where the conversation was saved.
        """
        if filepath is None:
            filepath = f"conversations/{self.name}_{self.conversation.id}.json"
        
        self.conversation.export_conversation(filepath)
        logger.info(f"📄 Conversation exported to: {filepath}")
        return filepath

    def display_conversation(self, detailed: bool = True):
        """
        Display the conversation history in a formatted way.
        
        Args:
            detailed (bool): Whether to show detailed information.
        """
        logger.info(f"💬 Displaying conversation for swarm: {self.name}")
        self.conversation.display_conversation(detailed=detailed)
```

---

## Creating Agents for Your Swarm

### Basic Agent Structure

```python
class CustomAgent:
    """
    A custom agent class that integrates with the swarm conversation system.
    
    Attributes:
        agent_name (str): The name of the agent.
        system_prompt (str): The system prompt guiding the agent's behavior.
        conversation (Optional[Conversation]): Shared conversation for context.
    """

    def __init__(
        self, 
        agent_name: str, 
        system_prompt: str,
        conversation: Optional[Conversation] = None
    ):
        """
        Initialize the agent with its name and system prompt.

        Args:
            agent_name (str): The name of the agent.
            system_prompt (str): The guiding prompt for the agent.
            conversation (Optional[Conversation]): Shared conversation context.
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.conversation = conversation

    def run(self, task: str, img: str = None, *args: Any, **kwargs: Any) -> Any:
        """
        Execute a specific task assigned to the agent.

        Args:
            task (str): The task description.
            img (str): The image input for processing.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the task execution.
        """
        # Add context from shared conversation if available
        context = ""
        if self.conversation:
            context = f"Previous context: {self.conversation.get_last_message_as_string()}\n\n"
        
        # Process the task (implement your custom logic here)
        result = f"Agent {self.agent_name} processed: {context}{task}"
        
        logger.info(f"🤖 Agent '{self.agent_name}' completed task")
        return result
```

### Using Swarms Framework Agents

You can also use the built-in Agent class from the Swarms framework:

```python
from swarms.structs.agent import Agent

def create_financial_agent() -> Agent:
    """Create a financial analysis agent."""
    return Agent(
        agent_name="FinancialAnalyst",
        system_prompt="You are a financial analyst specializing in market analysis and risk assessment.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

def create_marketing_agent() -> Agent:
    """Create a marketing analysis agent."""
    return Agent(
        agent_name="MarketingSpecialist", 
        system_prompt="You are a marketing specialist focused on campaign analysis and customer insights.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
```

---

## Complete Implementation Example

### Setting Up Your Swarm

```python
import time
from typing import List

def create_multi_domain_swarm() -> CustomSwarm:
    """
    Create a comprehensive multi-domain analysis swarm.
    
    Returns:
        CustomSwarm: A configured swarm with multiple specialized agents.
    """
    # Create agents
    agents = [
        create_financial_agent(),
        create_marketing_agent(),
        Agent(
            agent_name="OperationsAnalyst",
            system_prompt="You are an operations analyst specializing in process optimization and efficiency.",
            model_name="gpt-4o-mini",
            max_loops=1,
        ),
    ]
    
    # Configure conversation settings
    conversation_config = {
        "backend": "sqlite",  # Use SQLite for persistent storage
        "db_path": f"conversations/swarm_conversations.db",
        "time_enabled": True,
        "token_count": True,
    }
    
    # Create the swarm
    swarm = CustomSwarm(
        name="MultiDomainAnalysisSwarm",
        description="A comprehensive swarm for financial, marketing, and operations analysis",
        agents=agents,
        max_workers=3,
        autosave_conversation=True,
        conversation_config=conversation_config,
    )
    
    return swarm

# Usage example
if __name__ == "__main__":
    # Create and initialize the swarm
    swarm = create_multi_domain_swarm()
    
    # Execute a complex analysis task
    task = """
    Analyze the Q3 2024 performance data for our company:
    - Revenue: $2.5M (up 15% from Q2)
    
    - Customer acquisition: 1,200 new customers
    
    - Marketing spend: $150K
    
    - Operational costs: $800K
    
    
    Provide insights from financial, marketing, and operations perspectives.
    """
    
    # Run the analysis
    results = swarm.run(task)
    
    # Display results
    print("\n" + "="*50)
    print("SWARM ANALYSIS RESULTS")
    print("="*50)
    
    for i, result in enumerate(results):
        agent_name = swarm.agents[i].agent_name
        print(f"\n🤖 {agent_name}:")
        print(f"📊 {result}")
    
    # Get conversation summary
    summary = swarm.get_conversation_summary()
    print(f"\n📈 Conversation Summary:")
    print(f"   Total messages: {summary['total_messages']}")
    print(f"   Total tokens: {summary['token_summary']['total_tokens']}")
    
    # Export conversation for later analysis
    export_path = swarm.export_conversation()
    print(f"💾 Conversation saved to: {export_path}")
```

### Advanced Usage with Concurrent Execution

```python
def run_batch_analysis():
    """Example of running multiple tasks concurrently."""
    swarm = create_multi_domain_swarm()
    
    tasks = [
        "Analyze Q1 financial performance",
        "Evaluate marketing campaign effectiveness", 
        "Review operational efficiency metrics",
        "Assess customer satisfaction trends",
    ]
    
    # Process all tasks concurrently
    all_results = []
    for task in tasks:
        results = swarm.run_concurrent(task)
        all_results.append({"task": task, "results": results})
    
    return all_results
```

---

## Conversation Management Integration

The swarm uses the Swarms framework's [Conversation structure](../conversation/) for comprehensive message storage and management. This provides:

### Key Features

- **Persistent Storage**: Multiple backend options (SQLite, Redis, Supabase, etc.)

- **Message Categorization**: Organize messages by type (input, output, error, etc.)

- **Token Tracking**: Monitor token usage across conversations

- **Export/Import**: Save and load conversation histories

- **Search Capabilities**: Find specific messages or content


### Conversation Configuration Options

```python
conversation_config = {
    # Backend storage options
    "backend": "sqlite",  # or "redis", "supabase", "duckdb", "in-memory"
    
    # File-based storage
    "db_path": "conversations/swarm_data.db",
    
    # Redis configuration (if using Redis backend)
    "redis_host": "localhost",
    "redis_port": 6379,
    
    # Features
    "time_enabled": True,     # Add timestamps to messages
    "token_count": True,      # Track token usage
    "autosave": True,         # Automatically save conversations
    "save_enabled": True,     # Enable saving functionality
}
```

### Accessing Conversation Data

```python
# Get conversation history
history = swarm.conversation.return_history_as_string()

# Search for specific content
financial_messages = swarm.conversation.search("financial")

# Export conversation data
swarm.conversation.export_conversation("analysis_session.json")

# Get conversation statistics
stats = swarm.conversation.count_messages_by_role()
token_usage = swarm.conversation.export_and_count_categories()
```

For complete documentation on conversation management, see the [Conversation Structure Documentation](../conversation/).


---

## Conclusion

Building custom swarms with proper conversation management enables you to create powerful, scalable, and maintainable multi-agent systems. The integration with the Swarms framework's conversation structure provides:

- **Complete audit trail** of all agent interactions

- **Persistent storage** options for different deployment scenarios  

- **Performance monitoring** through token and message tracking

- **Easy debugging** with searchable conversation history

- **Scalable architecture** that grows with your needs


By following the patterns and best practices outlined in this guide, you can create robust swarms that handle complex tasks efficiently while maintaining full visibility into their operations.

### Key Takeaways

1. **Always implement conversation management** for tracking and auditing
2. **Use proper error handling and retries** for production resilience  
3. **Implement monitoring and logging** for observability
4. **Design for scalability** with concurrent execution patterns
5. **Test thoroughly** with unit tests and integration tests
6. **Configure appropriately** for your deployment environment

For more advanced patterns and examples, explore the [Swarms Examples](../../examples/) and consider contributing your custom swarms back to the community by submitting a pull request to the [Swarms repository](https://github.com/kyegomez/swarms).

---

## Additional Resources

- [Conversation Structure Documentation](../conversation/) - Complete guide to conversation management

- [Agent Documentation](../../agents/) - Learn about creating and configuring agents

- [Multi-Agent Architectures](../overview/) - Explore other swarm patterns and architectures

- [Examples Repository](../../examples/) - Real-world swarm implementations

- [Swarms Framework GitHub](https://github.com/kyegomez/swarms) - Source code and contributions
