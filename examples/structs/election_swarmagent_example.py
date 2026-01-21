"""
ElectionSwarm Example - Demonstrating multi-agent voting consensus mechanisms

This example shows how to:
1. Create multiple agents with different characteristics
2. Set up an ElectionSwarm with various voting methods
3. Run tasks through the ElectionSwarm
4. Compare results from different voting methods
5. Handle edge cases like ties and disagreements
"""

import os
import time
from swarms import Agent
from swarms.utils.formatter import formatter
from swarms.structs.election_swarm_agent import ElectionSwarm
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
# Set your API key if needed
# os.environ["OPENAI_API_KEY"] = "your-api-key"

def main():
    # Create header for the example
    formatter.print_panel(
        "ElectionSwarm Example - Multi-agent Voting Consensus",
        title="🗳️ Election Swarm Demo"
    )
    
    # Create agents with different personalities and models
    agents = create_diverse_agents(5)
    
    # Example 1: Basic Majority Voting
    formatter.print_panel("Example 1: Basic Majority Voting", style="blue")
    run_basic_majority_vote(agents)
    
    # Example 2: Compare Different Voting Methods
    formatter.print_panel("Example 2: Compare Different Voting Methods", style="green")
    compare_voting_methods(agents)
    
    # Example 3: Handling Ties with Judge
    formatter.print_panel("Example 3: Handling Ties with Judge", style="yellow")
    handle_ties_with_judge(agents)
    
    # Example 4: Performance with Complex Task
    formatter.print_panel("Example 4: Performance with Complex Task", style="purple")
    run_complex_task(agents)

def create_diverse_agents(num_agents=5):
    """Create a set of diverse agents with different models and system prompts."""
    
    # Define different personalities/perspectives for the agents
    personalities = [
        "You are an optimistic problem solver who focuses on innovative solutions.",
        "You are a careful analyst who prioritizes accuracy and details.",
        "You are a critical thinker who identifies potential issues and risks.",
        "You are a practical implementer who focuses on feasible actions.",
        "You are a strategic advisor who considers long-term implications.",
    ]
    
    # Use different temperature settings to create diversity
    temperatures = [0.5, 0.7, 0.3, 0.8, 0.4]
    
    # Create the agents
    agents = []
    
    for i in range(min(num_agents, len(personalities))):
        agent = Agent(
            agent_name=f"Agent-{i+1}",
            system_prompt=personalities[i],
            model_name="claude-3-7-sonnet-20250219",
            temperature=temperatures[i],
            verbose=False
        )
        agents.append(agent)
    
    formatter.print_panel(
        f"Created {len(agents)} diverse agents with different personalities and parameters",
        title="Agent Creation"
    )
    return agents

def run_basic_majority_vote(agents):
    """Run a simple task with majority voting."""
    # Create ElectionSwarm with majority voting
    election = ElectionSwarm(
        agents=agents,
        voting_method="majority",
        verbose=True,
        print_on=True
    )
    
    # Define a task that might generate diverse opinions
    task = "What's the best programming language for beginners and why? Keep your answer under 100 words."
    
    # Run the election
    print(f"Running task: {task}")
    result = election.run(task)
    
    # Show the result
    formatter.print_panel(
        f"Final result:\n{result}",
        title="Majority Vote Result"
    )
    
    # Show statistics
    stats = election.get_election_statistics()
    print("Election Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def compare_voting_methods(agents):
    """Compare different voting methods on the same task."""
    # Task that might have polarizing opinions
    task = "List three advantages and three disadvantages of remote work. Be concise."
    
    voting_methods = ["majority", "unanimous", "quorum", "ranked"]
    results = {}
    
    for method in voting_methods:
        # Create ElectionSwarm with the current voting method
        election = ElectionSwarm(
            agents=agents,
            voting_method=method,
            quorum_percentage=0.6 if method == "quorum" else 0.51,
            verbose=False,
            print_on=True
        )
        
        print(f"\nRunning with {method} voting method...")
        start_time = time.time()
        results[method] = election.run(task)
        duration = time.time() - start_time
        
        formatter.print_panel(
            f"{method.capitalize()} Voting Result (took {duration:.2f}s):\n\n{results[method]}",
            title=f"{method.capitalize()} Vote"
        )
    
    # Compare agreement between methods
    print("\nAgreement Analysis:")
    for i, method1 in enumerate(voting_methods):
        for method2 in voting_methods[i+1:]:
            similarity = "Same" if results[method1] == results[method2] else "Different"
            print(f"  {method1} vs {method2}: {similarity}")

def handle_ties_with_judge(agents):
    """Demonstrate how to handle ties with a judge agent."""
    # Create a special judge agent
    judge = Agent(
        agent_name="Judge",
        system_prompt="You are a fair and impartial judge. Your role is to evaluate responses and select the best one based on accuracy, clarity, and helpfulness.",
        model_name="claude-3-7-sonnet-20250219", 
        temperature=0.2,
        verbose=False
    )
    
    # Create ElectionSwarm with judge for tie-breaking
    election = ElectionSwarm(
        agents=agents[:4],  # Use even number of agents to increase tie probability
        voting_method="majority",
        tie_breaking_method="judge",
        judge_agent=judge,
        verbose=True,
        print_on=True
    )
    
    # Task designed to potentially create a split vote
    task = "Should companies adopt a 4-day work week? Give exactly two reasons for your position."
    
    print(f"Running task with potential for ties: {task}")
    result = election.run(task)
    
    # Show the result
    formatter.print_panel(
        f"Final result after potential tie-breaking:\n{result}",
        title="Tie Breaking Result"
    )
    
    # Get details of the election
    details = election.get_last_election_details()
    print("\nElection Details:")
    for agent_response in details.get("agent_responses", []):
        print(f"  {agent_response['agent_name']}: {agent_response['response'][:50]}...")

def run_complex_task(agents):
    """Test the system with a more complex task."""
    # Create ElectionSwarm with ranked voting
    election = ElectionSwarm(
        agents=agents,
        voting_method="ranked",
        max_voting_rounds=5,
        verbose=True,
        print_on=True
    )
    
    # Complex task requiring detailed analysis
    task = """
    Analyze the following scenario and provide recommendations:
    
    A mid-sized company (250 employees) is transitioning to a hybrid work model after being fully remote during the pandemic. 
    They want to maintain productivity while supporting employee wellbeing. 
    
    What are the three most important policies they should implement for this transition? Explain briefly why each policy matters.
    """
    
    print(f"Running complex task...")
    result = election.run(task)
    
    # Show the result
    formatter.print_panel(
        f"Final result for complex task:\n{result}",
        title="Complex Task Result"
    )

if __name__ == "__main__":
    main()