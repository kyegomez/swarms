from swarms import Agent
from swarms.structs.social_algorithms import SocialAlgorithms
import random


# Create agents that represent different "individuals" in the swarm
explorer1 = Agent(
    agent_name="Explorer_1",
    system_prompt="You are an explorer in a swarm. You seek new information, discover opportunities, and share findings with nearby agents. You are curious and adaptive.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

explorer2 = Agent(
    agent_name="Explorer_2",
    system_prompt="You are an explorer in a swarm. You seek new information, discover opportunities, and share findings with nearby agents. You are curious and adaptive.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

explorer3 = Agent(
    agent_name="Explorer_3",
    system_prompt="You are an explorer in a swarm. You seek new information, discover opportunities, and share findings with nearby agents. You are curious and adaptive.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

exploiter1 = Agent(
    agent_name="Exploiter_1",
    system_prompt="You are an exploiter in a swarm. You focus on developing and refining promising solutions found by explorers. You are methodical and detail-oriented.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

exploiter2 = Agent(
    agent_name="Exploiter_2",
    system_prompt="You are an exploiter in a swarm. You focus on developing and refining promising solutions found by explorers. You are methodical and detail-oriented.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

coordinator = Agent(
    agent_name="Swarm_Coordinator",
    system_prompt="You are a swarm coordinator that observes emergent patterns and provides minimal guidance. You don't control the swarm but help identify collective insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)


def swarm_intelligence_algorithm(agents, task, **kwargs):
    """
    A swarm intelligence algorithm where agents exhibit emergent behavior
    through local interactions and simple rules.
    """
    explorers = [
        agent for agent in agents if "Explorer" in agent.agent_name
    ]
    exploiters = [
        agent for agent in agents if "Exploiter" in agent.agent_name
    ]
    coordinator_agent = next(
        agent for agent in agents if "Coordinator" in agent.agent_name
    )

    max_iterations = kwargs.get("max_iterations", 8)
    exploration_ratio = kwargs.get(
        "exploration_ratio", 0.6
    )  # 60% exploration, 40% exploitation

    # Initialize swarm state
    swarm_knowledge = []
    pheromone_trails = (
        {}
    )  # Simulate pheromone trails for solution attractiveness
    agent_positions = {}  # Track agent "positions" in solution space

    # Phase 1: Initial Exploration
    coordinator_agent.run(
        f"Swarm intelligence process starting for: {task}"
    )

    # Initial random exploration by all agents
    initial_discoveries = []
    for i, agent in enumerate(explorers + exploiters):
        # Random exploration direction
        exploration_focus = random.choice(
            [
                "technical approach",
                "user experience",
                "business model",
                "implementation strategy",
                "risk mitigation",
                "innovation",
            ]
        )

        discovery_prompt = f"""
        As a swarm agent, explore this area: {exploration_focus}
        Related to task: {task}
        
        Make a random discovery or observation. Think like an individual in a swarm
        - you don't have the full picture, just your local perspective.
        Share what you find in your immediate environment.
        """

        discovery = agent.run(discovery_prompt)
        initial_discoveries.append(
            {
                "agent": agent.agent_name,
                "discovery": discovery,
                "focus": exploration_focus,
                "attractiveness": random.uniform(
                    0.1, 1.0
                ),  # Random initial attractiveness
            }
        )

        agent_positions[agent.agent_name] = exploration_focus

    swarm_knowledge.extend(initial_discoveries)

    # Phase 2: Swarm Dynamics - Multiple Iterations
    for iteration in range(1, max_iterations + 1):
        coordinator_agent.run(
            f"Swarm iteration {iteration} - observing emergent patterns"
        )

        # Calculate pheromone trails based on solution attractiveness
        for discovery in swarm_knowledge:
            solution_key = discovery["focus"]
            if solution_key not in pheromone_trails:
                pheromone_trails[solution_key] = 0
            pheromone_trails[solution_key] += discovery[
                "attractiveness"
            ]

        # Agents decide whether to explore or exploit based on local information
        agent_actions = []

        for agent in explorers + exploiters:
            # Simple decision rule: explore if pheromone trails are weak, exploit if strong
            current_position = agent_positions[agent.agent_name]
            local_pheromone = pheromone_trails.get(
                current_position, 0.1
            )

            # Random factor for swarm behavior
            random_factor = random.random()

            if (
                random_factor < exploration_ratio
                or local_pheromone < 0.5
            ):
                # Exploration behavior
                action_type = "explore"
                # Move to a new area or deepen current exploration
                if (
                    random.random() < 0.3
                ):  # 30% chance to move to new area
                    new_focus = random.choice(
                        [
                            "technical approach",
                            "user experience",
                            "business model",
                            "implementation strategy",
                            "risk mitigation",
                            "innovation",
                        ]
                    )
                    agent_positions[agent.agent_name] = new_focus
                    focus = new_focus
                else:
                    focus = current_position

                action_prompt = f"""
                As a swarm agent, continue exploring in: {focus}
                Task: {task}
                
                You've observed some pheromone trails (attractiveness: {local_pheromone:.2f}).
                Make a new discovery or observation in this area.
                Consider what other agents might have found.
                """
            else:
                # Exploitation behavior
                action_type = "exploit"
                # Focus on most attractive solutions
                best_solution = max(
                    pheromone_trails.items(), key=lambda x: x[1]
                )
                focus = best_solution[0]

                action_prompt = f"""
                As a swarm agent, exploit the promising area: {focus}
                Task: {task}
                
                This area has high pheromone concentration ({best_solution[1]:.2f}).
                Develop and refine solutions in this area.
                Build upon what the swarm has discovered.
                """

            action_result = agent.run(action_prompt)

            # Calculate new attractiveness based on result quality
            attractiveness = random.uniform(
                0.3, 1.0
            )  # Simplified attractiveness calculation

            agent_actions.append(
                {
                    "agent": agent.agent_name,
                    "action_type": action_type,
                    "focus": focus,
                    "result": action_result,
                    "attractiveness": attractiveness,
                    "iteration": iteration,
                }
            )

        # Update swarm knowledge
        swarm_knowledge.extend(agent_actions)

        # Swarm communication - agents share information locally
        if iteration % 2 == 0:  # Every other iteration
            for agent in explorers + exploiters:
                # Find nearby agents (simplified: random selection)
                nearby_agents = random.sample(
                    [
                        a
                        for a in explorers + exploiters
                        if a.agent_name != agent.agent_name
                    ],
                    min(2, len(explorers + exploiters) - 1),
                )

                # Share information with nearby agents
                shared_info = [
                    action
                    for action in agent_actions
                    if action["agent"]
                    in [a.agent_name for a in nearby_agents]
                ]

                if shared_info:
                    communication_prompt = f"""
                    As a swarm agent, you've received information from nearby agents:
                    {shared_info}
                    
                    How does this influence your next actions?
                    What patterns do you observe in the swarm's behavior?
                    """

                    communication_result = agent.run(
                        communication_prompt
                    )
                    agent_actions.append(
                        {
                            "agent": agent.agent_name,
                            "action_type": "communication",
                            "result": communication_result,
                            "iteration": iteration,
                        }
                    )

        # Coordinator observes emergent patterns
        if iteration % 3 == 0:  # Every third iteration
            pattern_prompt = f"""
            As swarm coordinator, observe the emergent patterns in iteration {iteration}:
            
            Recent agent actions: {agent_actions[-6:] if len(agent_actions) >= 6 else agent_actions}
            Pheromone trails: {pheromone_trails}
            
            What collective behaviors are emerging?
            What solutions are the swarm converging on?
            What guidance should you provide to the swarm?
            """

            pattern_observation = coordinator_agent.run(
                pattern_prompt
            )
            agent_actions.append(
                {
                    "agent": coordinator_agent.agent_name,
                    "action_type": "pattern_observation",
                    "result": pattern_observation,
                    "iteration": iteration,
                }
            )

    # Phase 3: Emergent Solution Synthesis
    # Find the most attractive solutions (highest pheromone trails)
    top_solutions = sorted(
        pheromone_trails.items(), key=lambda x: x[1], reverse=True
    )[:3]

    # Collect all discoveries related to top solutions
    emergent_solutions = []
    for solution_focus, attractiveness in top_solutions:
        related_discoveries = [
            action
            for action in swarm_knowledge
            if action.get("focus") == solution_focus
        ]
        emergent_solutions.append(
            {
                "focus": solution_focus,
                "attractiveness": attractiveness,
                "discoveries": related_discoveries,
            }
        )

    # Final swarm synthesis
    synthesis_prompt = f"""
    As swarm coordinator, synthesize the emergent intelligence from the swarm:
    
    Task: {task}
    Top emergent solutions: {emergent_solutions}
    Total swarm knowledge: {len(swarm_knowledge)} discoveries
    
    Create a final synthesis that represents the collective intelligence of the swarm.
    Identify the most promising solutions and explain how they emerged from local interactions.
    """

    final_synthesis = coordinator_agent.run(synthesis_prompt)

    return {
        "task": task,
        "swarm_knowledge": swarm_knowledge,
        "pheromone_trails": pheromone_trails,
        "emergent_solutions": emergent_solutions,
        "final_synthesis": final_synthesis,
        "total_iterations": max_iterations,
        "total_discoveries": len(swarm_knowledge),
        "algorithm_type": "swarm_intelligence",
    }


# Swarm Intelligence Algorithm
social_alg = SocialAlgorithms(
    name="Swarm-Intelligence-Algorithm",
    description="Swarm intelligence algorithm with emergent behavior and local interactions",
    agents=[
        explorer1,
        explorer2,
        explorer3,
        exploiter1,
        exploiter2,
        coordinator,
    ],
    social_algorithm=swarm_intelligence_algorithm,
    verbose=True,
    max_execution_time=1000,  # 16+ minutes for multiple swarm iterations
)

if __name__ == "__main__":
    result = social_alg.run(
        "Find innovative solutions for sustainable urban transportation",
        algorithm_args={
            "max_iterations": 8,
            "exploration_ratio": 0.6,
        },
    )

    print("=== SWARM INTELLIGENCE ALGORITHM RESULTS ===")
    print(f"Task: {result.final_outputs['task']}")
    print(
        f"Total Iterations: {result.final_outputs['total_iterations']}"
    )
    print(
        f"Total Discoveries: {result.final_outputs['total_discoveries']}"
    )
    print(f"Execution Time: {result.execution_time:.2f} seconds")
    print(f"Total Communication Steps: {result.total_steps}")

    print("\n=== PHEROMONE TRAILS (Solution Attractiveness) ===")
    for solution, attractiveness in sorted(
        result.final_outputs["pheromone_trails"].items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        print(f"{solution}: {attractiveness:.2f}")

    print("\n=== EMERGENT SOLUTIONS ===")
    for i, solution in enumerate(
        result.final_outputs["emergent_solutions"]
    ):
        print(
            f"Solution {i+1}: {solution['focus']} (attractiveness: {solution['attractiveness']:.2f})"
        )
        print(f"  Discoveries: {len(solution['discoveries'])}")
        print()

    print("\n=== FINAL SWARM SYNTHESIS ===")
    print(result.final_outputs["final_synthesis"][:500] + "...")

    print("\n=== SWARM BEHAVIOR PATTERNS ===")
    action_types = {}
    for action in result.final_outputs["swarm_knowledge"]:
        action_type = action.get("action_type", "unknown")
        action_types[action_type] = (
            action_types.get(action_type, 0) + 1
        )

    for action_type, count in action_types.items():
        print(f"{action_type}: {count} actions")
