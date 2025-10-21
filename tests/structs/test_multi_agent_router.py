import pytest
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


# Agents fixture
@pytest.fixture
def agents():
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics and providing detailed, factual information",
            system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
            max_loops=1,
        ),
        Agent(
            agent_name="CodeExpertAgent",
            agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
            system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
            max_loops=1,
        ),
        Agent(
            agent_name="WritingAgent",
            agent_description="Skilled in creative and technical writing, content creation, and editing",
            system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
            max_loops=1,
        ),
    ]


@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4.1",
        "gpt-4o",
        "gpt-5-mini",
        "o4-mini",
        "o3",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
    ],
)
def test_multiagentrouter_models(agents, model_name):
    """
    Run MultiAgentRouter on a variety of models to ensure no errors are raised.
    """
    task = "Use all the agents available to you to remake the Fibonacci function in Python, providing both an explanation and code."
    router_execute = MultiAgentRouter(
        agents=agents,
        temperature=0.5,
        model=model_name,
    )
    try:
        result = router_execute.run(task)
        assert result is not None
    except Exception as e:
        pytest.fail(f"Model {model_name} raised exception: {e}")


if __name__ == "__main__":
    pytest.main(args=[__file__])
