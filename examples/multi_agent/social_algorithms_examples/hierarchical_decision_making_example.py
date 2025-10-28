from typing import List, Dict, Any
from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms


def hierarchical_decision_making_algorithm(
    agents: List[Agent], task: str, **kwargs
) -> Dict[str, Any]:
    """
    A hierarchical decision-making algorithm with a leader and specialized workers.

    Args:
        agents: List of agents participating in the algorithm
        task: The task to be processed
        **kwargs: Additional keyword arguments

    Returns:
        Dict containing the decision-making results
    """
    if len(agents) < 2:
        raise ValueError("This algorithm requires at least 2 agents")

    # First agent is the leader/coordinator
    leader = agents[0]
    workers = agents[1:]

    # Leader analyzes the task and creates a plan
    planning_prompt = f"As a leader, analyze this task and create a detailed plan: {task}"
    plan = leader.run(planning_prompt)

    # Leader assigns subtasks to workers
    assignment_prompt = f"Based on this plan, assign specific subtasks to {len(workers)} workers:\n\nPlan: {plan}"
    assignments = leader.run(assignment_prompt)

    # Workers execute their assigned tasks
    worker_results = {}
    for i, worker in enumerate(workers):
        worker_prompt = f"Execute this assigned task (Worker {i+1}): {assignments}"
        result = worker.run(worker_prompt)
        worker_results[f"worker_{i+1}_{worker.agent_name}"] = result

    # Leader synthesizes all results
    synthesis_prompt = f"Review all worker results and make a final decision:\n\nPlan: {plan}\n\nWorker Results: {worker_results}"
    final_decision = leader.run(synthesis_prompt)

    return {
        "leader_plan": plan,
        "task_assignments": assignments,
        "worker_results": worker_results,
        "final_decision": final_decision,
        "task": task,
    }


# Create agents
leader = Agent(
    agent_name="Leader",
    system_prompt="You are a strategic leader focused on planning, coordination, and decision-making.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

worker1 = Agent(
    agent_name="Worker1",
    system_prompt="You are a specialized worker focused on executing assigned tasks efficiently.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

worker2 = Agent(
    agent_name="Worker2",
    system_prompt="You are a specialized worker focused on executing assigned tasks efficiently.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create and run the social algorithm
social_alg = SocialAlgorithms(
    name="Hierarchical-Decision-Making",
    description="Hierarchical decision-making with leader and workers",
    agents=[leader, worker1, worker2],
    social_algorithm=hierarchical_decision_making_algorithm,
    verbose=True,
)

result = social_alg.run(
    "Develop a comprehensive marketing strategy for a new product launch"
)
