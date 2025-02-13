import asyncio
from typing import List

from swarms.structs.agent import Agent
from swarms.structs.swarming_architectures import (
    broadcast,
    circular_swarm,
    exponential_swarm,
    fibonacci_swarm,
    grid_swarm,
    linear_swarm,
    mesh_swarm,
    one_to_three,
    prime_swarm,
    sigmoid_swarm,
    sinusoidal_swarm,
    staircase_swarm,
    star_swarm,
)


def create_finance_agents() -> List[Agent]:
    """Create specialized finance agents"""
    return [
        Agent(
            agent_name="MarketAnalyst",
            system_prompt="You are a market analysis expert. Analyze market trends and provide insights.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="RiskManager",
            system_prompt="You are a risk management specialist. Evaluate risks and provide mitigation strategies.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="PortfolioManager",
            system_prompt="You are a portfolio management expert. Optimize investment portfolios and asset allocation.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="ComplianceOfficer",
            system_prompt="You are a financial compliance expert. Ensure regulatory compliance and identify issues.",
            model_name="gpt-4o-mini",
        ),
    ]


def create_healthcare_agents() -> List[Agent]:
    """Create specialized healthcare agents"""
    return [
        Agent(
            agent_name="Diagnostician",
            system_prompt="You are a medical diagnostician. Analyze symptoms and suggest potential diagnoses.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="Treatment_Planner",
            system_prompt="You are a treatment planning specialist. Develop comprehensive treatment plans.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="MedicalResearcher",
            system_prompt="You are a medical researcher. Analyze latest research and provide evidence-based recommendations.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="PatientCareCoordinator",
            system_prompt="You are a patient care coordinator. Manage patient care workflow and coordination.",
            model_name="gpt-4o-mini",
        ),
    ]


def print_separator():
    print("\n" + "=" * 50 + "\n")


def run_finance_circular_swarm():
    """Investment analysis workflow using circular swarm"""
    print_separator()
    print("FINANCE - INVESTMENT ANALYSIS (Circular Swarm)")

    agents = create_finance_agents()
    tasks = [
        "Analyze Tesla stock performance for Q4 2024",
        "Assess market risks and potential hedging strategies",
        "Recommend portfolio adjustments based on analysis",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    result = circular_swarm(agents, tasks)
    print("\nResults:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Task: {log['task']}")
        print(f"Response: {log['response']}")


def run_healthcare_grid_swarm():
    """Patient diagnosis and treatment planning using grid swarm"""
    print_separator()
    print("HEALTHCARE - PATIENT DIAGNOSIS (Grid Swarm)")

    agents = create_healthcare_agents()
    tasks = [
        "Review patient symptoms: fever, fatigue, joint pain",
        "Research latest treatment protocols",
        "Develop preliminary treatment plan",
        "Coordinate with specialists",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    result = grid_swarm(agents, tasks)
    print("\nGrid swarm processing completed")
    print(result)


def run_finance_linear_swarm():
    """Loan approval process using linear swarm"""
    print_separator()
    print("FINANCE - LOAN APPROVAL PROCESS (Linear Swarm)")

    agents = create_finance_agents()[:3]
    tasks = [
        "Review loan application and credit history",
        "Assess risk factors and compliance requirements",
        "Generate final loan recommendation",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    result = linear_swarm(agents, tasks)
    print("\nResults:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Task: {log['task']}")
        print(f"Response: {log['response']}")


def run_healthcare_star_swarm():
    """Complex medical case management using star swarm"""
    print_separator()
    print("HEALTHCARE - COMPLEX CASE MANAGEMENT (Star Swarm)")

    agents = create_healthcare_agents()
    tasks = [
        "Complex case: Patient with multiple chronic conditions",
        "Develop integrated care plan",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    result = star_swarm(agents, tasks)
    print("\nResults:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Task: {log['task']}")
        print(f"Response: {log['response']}")


def run_finance_mesh_swarm():
    """Market risk assessment using mesh swarm"""
    print_separator()
    print("FINANCE - MARKET RISK ASSESSMENT (Mesh Swarm)")

    agents = create_finance_agents()
    tasks = [
        "Analyze global market conditions",
        "Assess currency exchange risks",
        "Evaluate sector-specific risks",
        "Review portfolio exposure",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    result = mesh_swarm(agents, tasks)
    print("\nResults:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Task: {log['task']}")
        print(f"Response: {log['response']}")


def run_mathematical_finance_swarms():
    """Complex financial analysis using mathematical swarms"""
    print_separator()
    print("FINANCE - MARKET PATTERN ANALYSIS")

    agents = create_finance_agents()
    tasks = [
        "Analyze historical market patterns",
        "Predict market trends using technical analysis",
        "Identify potential arbitrage opportunities",
    ]

    print("\nTasks:")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task}")

    print("\nFibonacci Swarm Results:")
    result = fibonacci_swarm(agents, tasks.copy())
    print(result)

    print("\nPrime Swarm Results:")
    result = prime_swarm(agents, tasks.copy())
    print(result)

    print("\nExponential Swarm Results:")
    result = exponential_swarm(agents, tasks.copy())
    print(result)


def run_healthcare_pattern_swarms():
    """Patient monitoring using pattern swarms"""
    print_separator()
    print("HEALTHCARE - PATIENT MONITORING PATTERNS")

    agents = create_healthcare_agents()
    task = "Monitor and analyze patient vital signs: BP, heart rate, temperature, O2 saturation"

    print(f"\nTask: {task}")

    print("\nStaircase Pattern Analysis:")
    result = staircase_swarm(agents, task)
    print(result)

    print("\nSigmoid Pattern Analysis:")
    result = sigmoid_swarm(agents, task)
    print(result)

    print("\nSinusoidal Pattern Analysis:")
    result = sinusoidal_swarm(agents, task)
    print(result)


async def run_communication_examples():
    """Communication patterns for emergency scenarios"""
    print_separator()
    print("EMERGENCY COMMUNICATION PATTERNS")

    # Finance market alert
    finance_sender = create_finance_agents()[0]
    finance_receivers = create_finance_agents()[1:]
    market_alert = "URGENT: Major market volatility detected - immediate risk assessment required"

    print("\nFinance Market Alert:")
    print(f"Alert: {market_alert}")
    result = await broadcast(
        finance_sender, finance_receivers, market_alert
    )
    print("\nBroadcast Results:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Response: {log['response']}")

    # Healthcare emergency
    health_sender = create_healthcare_agents()[0]
    health_receivers = create_healthcare_agents()[1:4]
    emergency_case = "EMERGENCY: Trauma patient with multiple injuries - immediate consultation required"

    print("\nHealthcare Emergency:")
    print(f"Case: {emergency_case}")
    result = await one_to_three(
        health_sender, health_receivers, emergency_case
    )
    print("\nConsultation Results:")
    for log in result["history"]:
        print(f"\n{log['agent_name']}:")
        print(f"Response: {log['response']}")


async def run_all_examples():
    """Execute all swarm examples"""
    print("\n=== SWARM ARCHITECTURE EXAMPLES ===\n")

    # Finance examples
    run_finance_circular_swarm()
    run_finance_linear_swarm()
    run_finance_mesh_swarm()
    run_mathematical_finance_swarms()

    # Healthcare examples
    run_healthcare_grid_swarm()
    run_healthcare_star_swarm()
    run_healthcare_pattern_swarms()

    # Communication examples
    await run_communication_examples()

    print("\n=== ALL EXAMPLES COMPLETED ===")


if __name__ == "__main__":
    asyncio.run(run_all_examples())
