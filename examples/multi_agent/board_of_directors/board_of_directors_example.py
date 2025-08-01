"""
Board of Directors Example

This example demonstrates how to use the Board of Directors swarm feature
in the Swarms Framework. It shows how to create a board, configure it,
and use it to orchestrate tasks across multiple agents.

To run this example:
1. Make sure you're in the root directory of the swarms project
2. Run: python examples/multi_agent/board_of_directors/board_of_directors_example.py
"""

import os
import sys
from typing import List

# Add the root directory to the Python path if running from examples directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if 'examples' in current_dir:
    root_dir = current_dir
    while os.path.basename(root_dir) != 'examples' and root_dir != os.path.dirname(root_dir):
        root_dir = os.path.dirname(root_dir)
    if os.path.basename(root_dir) == 'examples':
        root_dir = os.path.dirname(root_dir)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole,
)
from swarms.structs.agent import Agent


def create_board_members() -> List[BoardMember]:
    """Create board members with specific roles."""
    
    chairman = Agent(
        agent_name="Chairman",
        agent_description="Executive Chairman with strategic vision",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are the Executive Chairman. Provide strategic leadership and facilitate decision-making.",
    )
    
    cto = Agent(
        agent_name="CTO",
        agent_description="Chief Technology Officer with technical expertise",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are the CTO. Provide technical leadership and evaluate technology solutions.",
    )
    
    cfo = Agent(
        agent_name="CFO",
        agent_description="Chief Financial Officer with financial expertise",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are the CFO. Provide financial analysis and ensure fiscal responsibility.",
    )
    
    return [
        BoardMember(
            agent=chairman,
            role=BoardMemberRole.CHAIRMAN,
            voting_weight=2.0,
            expertise_areas=["leadership", "strategy"]
        ),
        BoardMember(
            agent=cto,
            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
            voting_weight=1.5,
            expertise_areas=["technology", "innovation"]
        ),
        BoardMember(
            agent=cfo,
            role=BoardMemberRole.EXECUTIVE_DIRECTOR,
            voting_weight=1.5,
            expertise_areas=["finance", "risk_management"]
        ),
    ]


def create_worker_agents() -> List[Agent]:
    """Create worker agents for the swarm."""
    
    researcher = Agent(
        agent_name="Researcher",
        agent_description="Research analyst for data analysis",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a Research Analyst. Conduct thorough research and provide data-driven insights.",
    )
    
    developer = Agent(
        agent_name="Developer",
        agent_description="Software developer for implementation",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a Software Developer. Design and implement software solutions.",
    )
    
    marketer = Agent(
        agent_name="Marketer",
        agent_description="Marketing specialist for strategy",
        model_name="gpt-4o-mini",
        max_loops=1,
        system_prompt="You are a Marketing Specialist. Develop marketing strategies and campaigns.",
    )
    
    return [researcher, developer, marketer]


def run_board_example() -> None:
    """Run a Board of Directors example."""
    
    # Create board members and worker agents
    board_members = create_board_members()
    worker_agents = create_worker_agents()
    
    # Create the Board of Directors swarm
    board_swarm = BoardOfDirectorsSwarm(
        name="Executive_Board",
        board_members=board_members,
        agents=worker_agents,
        max_loops=2,
        verbose=True,
        decision_threshold=0.6,
    )
    
    # Define task
    task = """
    Develop a strategy for launching a new AI-powered product in the market.
    Include market research, technical planning, marketing strategy, and financial projections.
    """
    
    # Execute the task
    result = board_swarm.run(task=task)
    
    print("Task completed successfully!")
    print(f"Result: {result}")


def run_simple_example() -> None:
    """Run a simple Board of Directors example."""
    
    # Create simple agents
    analyst = Agent(
        agent_name="Analyst",
        agent_description="Data analyst",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    
    writer = Agent(
        agent_name="Writer",
        agent_description="Content writer",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    
    # Create swarm with default settings
    board_swarm = BoardOfDirectorsSwarm(
        name="Simple_Board",
        agents=[analyst, writer],
        verbose=True,
    )
    
    # Execute simple task
    task = "Analyze current market trends and create a summary report."
    result = board_swarm.run(task=task)
    
    print("Simple example completed!")
    print(f"Result: {result}")


def main() -> None:
    """Main function to run the examples."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Example may not work.")
        return
    
    try:
        print("Running simple Board of Directors example...")
        run_simple_example()
        
        print("\nRunning comprehensive Board of Directors example...")
        run_board_example()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
