from swarms.structs.agent import Agent
from te import run_concurrently_greenlets, with_retries
from typing import Callable, List, Tuple


# Define some example agent tasks
@with_retries(max_retries=2)
def financial_analysis_task(query: str) -> str:
    agent = Agent(
        agent_name="Financial-Analysis-Agent",
        agent_description="Personal finance advisor agent",
        system_prompt="You are a personal finance advisor agent",
        max_loops=2,
        model_name="gpt-4o-mini",
        dynamic_temperature_enabled=True,
        interactive=False,
        output_type="final",
        safety_prompt_on=True,
    )
    return agent.run(query)


@with_retries(max_retries=2)
def investment_advice_task(query: str) -> str:
    agent = Agent(
        agent_name="Investment-Advisor-Agent",
        agent_description="Investment strategy advisor agent",
        system_prompt="You are an investment strategy advisor agent",
        max_loops=2,
        model_name="gpt-4o-mini",
        dynamic_temperature_enabled=True,
        interactive=False,
        output_type="final",
        safety_prompt_on=True,
    )
    return agent.run(query)


async def market_analysis_task(query: str) -> str:
    agent = Agent(
        agent_name="Market-Analysis-Agent",
        agent_description="Market analysis agent",
        system_prompt="You are a market analysis agent",
        max_loops=2,
        model_name="gpt-4o-mini",
        dynamic_temperature_enabled=True,
        interactive=False,
        output_type="final",
        safety_prompt_on=True,
    )
    return agent.run(query)


def main():
    # Define the tasks to run concurrently
    tasks: List[Tuple[Callable, tuple, dict]] = [
        (
            financial_analysis_task,
            ("What are the best practices for saving money?",),
            {},
        ),
        (
            investment_advice_task,
            ("What are the current market trends?",),
            {},
        ),
        (
            market_analysis_task,
            ("Analyze the current market conditions",),
            {},
        ),
    ]

    # Run the tasks concurrently
    results = run_concurrently_greenlets(
        tasks,
        timeout=30,  # 30 seconds global timeout
        max_concurrency=3,  # Run 3 tasks concurrently
        max_retries=2,
        task_timeout=10,  # 10 seconds per task timeout
    )

    # Process and display results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed with error: {result}")
        else:
            print(f"Task {i} succeeded with result: {result}")


if __name__ == "__main__":
    main()
