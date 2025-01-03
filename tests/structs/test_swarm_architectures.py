import asyncio
import time
from typing import List

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    exponential_swarm,
    geometric_swarm,
    grid_swarm,
    harmonic_swarm,
    linear_swarm,
    log_swarm,
    mesh_swarm,
    one_to_one,
    one_to_three,
    power_swarm,
    pyramid_swarm,
    sigmoid_swarm,
    sinusoidal_swarm,
    staircase_swarm,
    star_swarm,
)


def create_test_agent(name: str) -> Agent:
    """Create a test agent with specified name"""
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Respond with your name and the task you received.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )


def create_test_agents(num_agents: int) -> List[Agent]:
    """Create specified number of test agents"""
    return [
        create_test_agent(f"Agent{i+1}") for i in range(num_agents)
    ]


def print_separator():
    print("\n" + "=" * 50 + "\n")


def test_circular_swarm():
    """Test and display circular swarm outputs"""
    print_separator()
    print("CIRCULAR SWARM TEST")
    try:
        agents = create_test_agents(3)
        tasks = [
            "Analyze data",
            "Generate report",
            "Summarize findings",
        ]

        print("Running circular swarm with:")
        print(f"Tasks: {tasks}\n")

        result = circular_swarm(agents, tasks)
        print("Circular Swarm Outputs:")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_grid_swarm():
    """Test and display grid swarm outputs"""
    print_separator()
    print("GRID SWARM TEST")
    try:
        agents = create_test_agents(4)  # 2x2 grid
        tasks = ["Task A", "Task B", "Task C", "Task D"]

        print("Running grid swarm with 2x2 grid")
        print(f"Tasks: {tasks}\n")

        print(grid_swarm(agents, tasks))
        print(
            "Grid Swarm completed - each agent processed tasks in its grid position"
        )
    except Exception as e:
        print(f"Error: {str(e)}")


def test_linear_swarm():
    """Test and display linear swarm outputs"""
    print_separator()
    print("LINEAR SWARM TEST")
    try:
        agents = create_test_agents(3)
        tasks = ["Research task", "Write content", "Review output"]

        print("Running linear swarm with:")
        print(f"Tasks: {tasks}\n")

        result = linear_swarm(agents, tasks)
        print("Linear Swarm Outputs:")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_star_swarm():
    """Test and display star swarm outputs"""
    print_separator()
    print("STAR SWARM TEST")
    try:
        agents = create_test_agents(4)  # 1 center + 3 peripheral
        tasks = ["Coordinate workflow", "Process data"]

        print("Running star swarm with:")
        print(f"Center agent: {agents[0].agent_name}")
        print(
            f"Peripheral agents: {[agent.agent_name for agent in agents[1:]]}"
        )
        print(f"Tasks: {tasks}\n")

        result = star_swarm(agents, tasks)
        print("Star Swarm Outputs:")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_mesh_swarm():
    """Test and display mesh swarm outputs"""
    print_separator()
    print("MESH SWARM TEST")
    try:
        agents = create_test_agents(3)
        tasks = [
            "Analyze data",
            "Process information",
            "Generate insights",
        ]

        print("Running mesh swarm with:")
        print(f"Tasks: {tasks}\n")

        result = mesh_swarm(agents, tasks)
        print(f"Mesh Swarm Outputs: {result}")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")


def test_pyramid_swarm():
    """Test and display pyramid swarm outputs"""
    print_separator()
    print("PYRAMID SWARM TEST")
    try:
        agents = create_test_agents(6)  # 1-2-3 pyramid
        tasks = [
            "Top task",
            "Middle task 1",
            "Middle task 2",
            "Bottom task 1",
            "Bottom task 2",
            "Bottom task 3",
        ]

        print("Running pyramid swarm with:")
        print(f"Tasks: {tasks}\n")

        result = pyramid_swarm(agents, tasks)
        print(f"Pyramid Swarm Outputs: {result}")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")


async def test_communication_patterns():
    """Test and display agent communication patterns"""
    print_separator()
    print("COMMUNICATION PATTERNS TEST")
    try:
        sender = create_test_agent("Sender")
        receiver = create_test_agent("Receiver")
        task = "Process and relay this message"

        print("Testing One-to-One Communication:")
        result = one_to_one(sender, receiver, task)
        print(f"\nOne-to-One Communication Outputs: {result}")
        for log in result["history"]:
            print(f"\nAgent: {log['agent_name']}")
            print(f"Task: {log['task']}")
            print(f"Response: {log['response']}")

        print("\nTesting One-to-Three Communication:")
        receivers = create_test_agents(3)
        await one_to_three(sender, receivers, task)

        print("\nTesting Broadcast Communication:")
        broadcast_receivers = create_test_agents(5)
        await broadcast(sender, broadcast_receivers, task)

    except Exception as e:
        print(f"Error: {str(e)}")


def test_mathematical_swarms():
    """Test and display mathematical swarm patterns"""
    print_separator()
    print("MATHEMATICAL SWARMS TEST")
    try:
        agents = create_test_agents(8)
        base_tasks = ["Calculate", "Process", "Analyze"]

        # Test each mathematical swarm
        for swarm_type, swarm_func in [
            ("Power Swarm", power_swarm),
            ("Log Swarm", log_swarm),
            ("Exponential Swarm", exponential_swarm),
            ("Geometric Swarm", geometric_swarm),
            ("Harmonic Swarm", harmonic_swarm),
        ]:
            print(f"\nTesting {swarm_type}:")
            tasks = [f"{task} in {swarm_type}" for task in base_tasks]
            print(f"Tasks: {tasks}")
            swarm_func(agents, tasks.copy())

    except Exception as e:
        print(f"Error: {str(e)}")


def test_pattern_swarms():
    """Test and display pattern-based swarms"""
    print_separator()
    print("PATTERN-BASED SWARMS TEST")
    try:
        agents = create_test_agents(10)
        task = "Process according to pattern"

        for swarm_type, swarm_func in [
            ("Staircase Swarm", staircase_swarm),
            ("Sigmoid Swarm", sigmoid_swarm),
            ("Sinusoidal Swarm", sinusoidal_swarm),
        ]:
            print(f"\nTesting {swarm_type}:")
            print(f"Task: {task}")
            swarm_func(agents, task)

    except Exception as e:
        print(f"Error: {str(e)}")


def run_all_tests():
    """Run all swarm architecture tests"""
    print(
        "\n=== Starting Swarm Architectures Test Suite with Outputs ==="
    )
    start_time = time.time()

    try:
        # Test basic swarm patterns
        test_circular_swarm()
        test_grid_swarm()
        test_linear_swarm()
        test_star_swarm()
        test_mesh_swarm()
        test_pyramid_swarm()

        # Test mathematical and pattern swarms
        test_mathematical_swarms()
        test_pattern_swarms()

        # Test communication patterns
        asyncio.run(test_communication_patterns())

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print("\n=== Test Suite Completed Successfully ===")
        print(f"Time taken: {duration} seconds")

    except Exception as e:
        print("\n=== Test Suite Failed ===")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()
