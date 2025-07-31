"""
Stagehand MCP Server Integration with Swarms
============================================

This example demonstrates how to use the Stagehand MCP (Model Context Protocol)
server with Swarms agents. The MCP server provides browser automation capabilities
as standardized tools that can be discovered and used by agents.

Prerequisites:
1. Install and run the Stagehand MCP server:
   cd stagehand-mcp-server
   npm install
   npm run build
   npm start

2. The server will start on http://localhost:3000/sse

Features:
- Automatic tool discovery from MCP server
- Multi-session browser management
- Built-in screenshot resources
- Prompt templates for common tasks
"""

import asyncio
import os
from typing import List, Optional

from dotenv import load_dotenv
from loguru import logger

from swarms import Agent

load_dotenv()


class StagehandMCPAgent:
    """
    A Swarms agent that connects to the Stagehand MCP server
    for browser automation capabilities.
    """

    def __init__(
        self,
        agent_name: str = "StagehandMCPAgent",
        mcp_server_url: str = "http://localhost:3000/sse",
        model_name: str = "gpt-4o-mini",
        max_loops: int = 1,
    ):
        """
        Initialize the Stagehand MCP Agent.

        Args:
            agent_name: Name of the agent
            mcp_server_url: URL of the Stagehand MCP server
            model_name: LLM model to use
            max_loops: Maximum number of reasoning loops
        """
        self.agent = Agent(
            agent_name=agent_name,
            model_name=model_name,
            max_loops=max_loops,
            # Connect to the Stagehand MCP server
            mcp_url=mcp_server_url,
            system_prompt="""You are a web browser automation specialist with access to Stagehand MCP tools.

Available tools from the MCP server:
- navigate: Navigate to a URL
- act: Perform actions on web pages (click, type, etc.)
- extract: Extract data from web pages
- observe: Find and observe elements on pages
- screenshot: Take screenshots
- createSession: Create new browser sessions for parallel tasks
- listSessions: List active browser sessions
- closeSession: Close browser sessions

For multi-page workflows, you can create multiple sessions.
Always be specific in your actions and extractions.
Remember to close sessions when done with them.""",
            verbose=True,
        )

    def run(self, task: str) -> str:
        """Run a browser automation task."""
        return self.agent.run(task)


class MultiSessionBrowserSwarm:
    """
    A multi-agent swarm that uses multiple browser sessions
    for parallel web automation tasks.
    """

    def __init__(
        self,
        mcp_server_url: str = "http://localhost:3000/sse",
        num_agents: int = 3,
    ):
        """
        Initialize a swarm of browser automation agents.

        Args:
            mcp_server_url: URL of the Stagehand MCP server
            num_agents: Number of agents to create
        """
        self.agents = []
        
        # Create specialized agents for different tasks
        agent_roles = [
            ("DataExtractor", "You specialize in extracting structured data from websites."),
            ("FormFiller", "You specialize in filling out forms and interacting with web applications."),
            ("WebMonitor", "You specialize in monitoring websites for changes and capturing screenshots."),
        ]
        
        for i in range(min(num_agents, len(agent_roles))):
            name, specialization = agent_roles[i]
            agent = Agent(
                agent_name=f"{name}_{i}",
                model_name="gpt-4o-mini",
                max_loops=1,
                mcp_url=mcp_server_url,
                system_prompt=f"""You are a web browser automation specialist. {specialization}

You have access to Stagehand MCP tools including:
- createSession: Create a new browser session
- navigate_session: Navigate to URLs in a specific session
- act_session: Perform actions in a specific session
- extract_session: Extract data from a specific session
- observe_session: Observe elements in a specific session
- closeSession: Close a session when done

Always create your own session for tasks to work independently from other agents.""",
                verbose=True,
            )
            self.agents.append(agent)

    def distribute_tasks(self, tasks: List[str]) -> List[str]:
        """Distribute tasks among agents."""
        results = []
        
        # Distribute tasks round-robin among agents
        for i, task in enumerate(tasks):
            agent_idx = i % len(self.agents)
            agent = self.agents[agent_idx]
            
            logger.info(f"Assigning task to {agent.agent_name}: {task}")
            result = agent.run(task)
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Stagehand MCP Server Integration Examples")
    print("=" * 70)
    print("\nMake sure the Stagehand MCP server is running on http://localhost:3000/sse")
    print("Run: cd stagehand-mcp-server && npm start\n")
    
    # Example 1: Single agent with MCP tools
    print("\nExample 1: Single Agent with MCP Tools")
    print("-" * 40)
    
    mcp_agent = StagehandMCPAgent(
        agent_name="WebResearchAgent",
        mcp_server_url="http://localhost:3000/sse",
    )
    
    # Research task using MCP tools
    result1 = mcp_agent.run(
        """Navigate to news.ycombinator.com and extract the following:
        1. The titles of the top 5 stories
        2. Their points/scores
        3. Number of comments for each
        Then take a screenshot of the page."""
    )
    print(f"Result: {result1}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 2: Multi-session parallel browsing
    print("Example 2: Multi-Session Parallel Browsing")
    print("-" * 40)
    
    parallel_agent = StagehandMCPAgent(
        agent_name="ParallelBrowserAgent",
        mcp_server_url="http://localhost:3000/sse",
    )
    
    result2 = parallel_agent.run(
        """Create 3 browser sessions and perform these tasks in parallel:
        1. Session 1: Go to github.com/trending and extract the top 3 trending repositories
        2. Session 2: Go to reddit.com/r/programming and extract the top 3 posts
        3. Session 3: Go to stackoverflow.com and extract the featured questions
        
        After extracting data from all sessions, close them."""
    )
    print(f"Result: {result2}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 3: Multi-agent browser swarm
    print("Example 3: Multi-Agent Browser Swarm")
    print("-" * 40)
    
    # Create a swarm of specialized browser agents
    browser_swarm = MultiSessionBrowserSwarm(
        mcp_server_url="http://localhost:3000/sse",
        num_agents=3,
    )
    
    # Define tasks for the swarm
    swarm_tasks = [
        "Create a session, navigate to python.org, and extract information about the latest Python version and its key features",
        "Create a session, go to npmjs.com, search for 'stagehand', and extract information about the package including version and description",
        "Create a session, visit playwright.dev, and extract the main features and benefits listed on the homepage",
    ]
    
    print("Distributing tasks to browser swarm...")
    swarm_results = browser_swarm.distribute_tasks(swarm_tasks)
    
    for i, result in enumerate(swarm_results):
        print(f"\nTask {i+1} Result: {result}")
    
    print("\n" + "=" * 70 + "\n")
    
    # Example 4: Complex workflow with session management
    print("Example 4: Complex Multi-Page Workflow")
    print("-" * 40)
    
    workflow_agent = StagehandMCPAgent(
        agent_name="WorkflowAgent",
        mcp_server_url="http://localhost:3000/sse",
        max_loops=2,  # Allow more complex reasoning
    )
    
    result4 = workflow_agent.run(
        """Perform a comprehensive analysis of AI frameworks:
        1. Create a new session
        2. Navigate to github.com/huggingface/transformers and extract the star count and latest release info
        3. In the same session, navigate to github.com/openai/gpt-3 and extract similar information
        4. Navigate to github.com/anthropics/anthropic-sdk-python and extract repository statistics
        5. Take screenshots of each repository page
        6. Compile a comparison report of all three repositories
        7. Close the session when done"""
    )
    print(f"Result: {result4}")
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)