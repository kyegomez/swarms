import pytest
import asyncio
from swarms import Agent, AsyncWorkflow
from swarm_models import OpenAIChat  # or any other model you prefer

@pytest.mark.asyncio
async def test_async_workflow():
    # Create test agents
    model = OpenAIChat()  # Initialize with appropriate parameters
    agents = [
        Agent(
            agent_name=f"Test-Agent-{i}",
            llm=model,
            max_loops=1,
            dashboard=False,
            verbose=True,
        )
        for i in range(3)
    ]

    # Initialize workflow
    workflow = AsyncWorkflow(
        name="Test-Async-Workflow",
        agents=agents,
        max_workers=3,
        verbose=True
    )

    # Run test task
    test_task = "What is 2+2?"
    results = await workflow.run(test_task)

    # Assertions
    assert len(results) == len(agents)
    assert all(isinstance(result, str) for result in results)
