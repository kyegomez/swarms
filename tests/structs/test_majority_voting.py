from unittest.mock import MagicMock

import pytest

from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def test_majority_voting_run_concurrent(mocker):
    # Create mock agents
    agent1 = MagicMock(spec=Agent)
    agent2 = MagicMock(spec=Agent)
    agent3 = MagicMock(spec=Agent)

    # Create mock majority voting
    mv = MajorityVoting(
        agents=[agent1, agent2, agent3],
        concurrent=True,
        multithreaded=False,
    )

    # Create mock conversation
    conversation = MagicMock()
    mv.conversation = conversation

    # Create mock results
    results = ["Paris", "Paris", "Lyon"]

    # Mock agent.run method
    agent1.run.return_value = results[0]
    agent2.run.return_value = results[1]
    agent3.run.return_value = results[2]

    # Run majority voting
    majority_vote = mv.run("What is the capital of France?")

    # Assert agent.run method was called with the correct task
    agent1.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent2.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent3.run.assert_called_once_with(
        "What is the capital of France?"
    )

    # Assert conversation.add method was called with the correct responses
    conversation.add.assert_any_call(agent1.agent_name, results[0])
    conversation.add.assert_any_call(agent2.agent_name, results[1])
    conversation.add.assert_any_call(agent3.agent_name, results[2])

    # Assert majority vote is correct
    assert majority_vote is not None


def test_majority_voting_run_multithreaded(mocker):
    # Create mock agents
    agent1 = MagicMock(spec=Agent)
    agent2 = MagicMock(spec=Agent)
    agent3 = MagicMock(spec=Agent)

    # Create mock majority voting
    mv = MajorityVoting(
        agents=[agent1, agent2, agent3],
        concurrent=False,
        multithreaded=True,
    )

    # Create mock conversation
    conversation = MagicMock()
    mv.conversation = conversation

    # Create mock results
    results = ["Paris", "Paris", "Lyon"]

    # Mock agent.run method
    agent1.run.return_value = results[0]
    agent2.run.return_value = results[1]
    agent3.run.return_value = results[2]

    # Run majority voting
    majority_vote = mv.run("What is the capital of France?")

    # Assert agent.run method was called with the correct task
    agent1.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent2.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent3.run.assert_called_once_with(
        "What is the capital of France?"
    )

    # Assert conversation.add method was called with the correct responses
    conversation.add.assert_any_call(agent1.agent_name, results[0])
    conversation.add.assert_any_call(agent2.agent_name, results[1])
    conversation.add.assert_any_call(agent3.agent_name, results[2])

    # Assert majority vote is correct
    assert majority_vote is not None


@pytest.mark.asyncio
async def test_majority_voting_run_asynchronous(mocker):
    # Create mock agents
    agent1 = MagicMock(spec=Agent)
    agent2 = MagicMock(spec=Agent)
    agent3 = MagicMock(spec=Agent)

    # Create mock majority voting
    mv = MajorityVoting(
        agents=[agent1, agent2, agent3],
        concurrent=False,
        multithreaded=False,
        asynchronous=True,
    )

    # Create mock conversation
    conversation = MagicMock()
    mv.conversation = conversation

    # Create mock results
    results = ["Paris", "Paris", "Lyon"]

    # Mock agent.run method
    agent1.run.return_value = results[0]
    agent2.run.return_value = results[1]
    agent3.run.return_value = results[2]

    # Run majority voting
    majority_vote = await mv.run("What is the capital of France?")

    # Assert agent.run method was called with the correct task
    agent1.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent2.run.assert_called_once_with(
        "What is the capital of France?"
    )
    agent3.run.assert_called_once_with(
        "What is the capital of France?"
    )

    # Assert conversation.add method was called with the correct responses
    conversation.add.assert_any_call(agent1.agent_name, results[0])
    conversation.add.assert_any_call(agent2.agent_name, results[1])
    conversation.add.assert_any_call(agent3.agent_name, results[2])

    # Assert majority vote is correct
    assert majority_vote is not None

def test_streaming_majority_voting():
    """
    Test the streaming_majority_voting with logging/try-except and assertion.
    """
    logs = []
    def streaming_callback(agent_name: str, chunk: str, is_final: bool):
        # Chunk buffer static per call (reset each session)
        if not hasattr(streaming_callback, "_buffer"):
            streaming_callback._buffer = ""
            streaming_callback._buffer_size = 0

        min_chunk_size = 512  # or any large chunk size you want

        if chunk:
            streaming_callback._buffer += chunk
            streaming_callback._buffer_size += len(chunk)
        if streaming_callback._buffer_size >= min_chunk_size or is_final:
            if streaming_callback._buffer:
                print(streaming_callback._buffer, end="", flush=True)
                logs.append(streaming_callback._buffer)
                streaming_callback._buffer = ""
                streaming_callback._buffer_size = 0
        if is_final:
            print()
     
    try:
        # Initialize the agent
        agent = Agent(
            agent_name="Financial-Analysis-Agent",
            agent_description="Personal finance advisor agent",
            system_prompt="You are a financial analysis agent.",  # replaced missing const
            max_loops=1,
            model_name="gpt-4.1",
            dynamic_temperature_enabled=True,
            user_name="swarms_corp",
            retry_attempts=3,
            context_length=8192,
            return_step_meta=False,
            output_type="str",  # "json", "dict", "csv" OR "string" "yaml" and
            auto_generate_prompt=False,  # Auto generate prompt for the agent based on name, description, and system prompt, task
            max_tokens=4000,  # max output tokens
            saved_state_path="agent_00.json",
            interactive=False,
            streaming_on=True,  #if concurrent agents want to be streamed
        )
        
        swarm = MajorityVoting(agents=[agent, agent, agent])
        
        result = swarm.run(
            "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown.",
            streaming_callback=streaming_callback,
        )
        assert result is not None
    except Exception as e:
        print("Error in test_streaming_majority_voting:", e)
        print("Logs so far:", logs)
        raise
