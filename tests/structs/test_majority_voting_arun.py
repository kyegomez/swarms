import pytest
from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def _record(agent, calls):
    """Replace an agent's run() and arun() with stubs recording the context."""
    name = agent.agent_name
    turns = [0]

    def _run(task=None, *args, messages=None, **kwargs):
        turns[0] += 1
        answer = f"{name}-answer-{turns[0]}"
        calls.append(
            {
                "agent": name,
                "task": task,
                "messages": list(messages or []),
                "answer": answer,
                "async": False,
            }
        )
        agent.short_memory.add(role=name, content=answer)
        return answer

    async def _arun(task=None, messages=None, **kwargs):
        turns[0] += 1
        answer = f"{name}-async-answer-{turns[0]}"
        calls.append(
            {
                "agent": name,
                "task": task,
                "messages": list(messages or []),
                "answer": answer,
                "async": True,
            }
        )
        agent.short_memory.add(role=name, content=answer)
        return answer

    agent.run = _run
    agent.arun = _arun
    return agent


def _recording_agents(names):
    calls = []
    agents = []
    for name in names:
        agent = Agent(
            agent_name=name,
            agent_description=f"Recording voter {name}",
            model_name="gpt-4o",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
            autosave=False,
        )
        agents.append(_record(agent, calls))
    return agents, calls


def _voting_system(names, max_loops):
    agents, calls = _recording_agents(names)
    mv = MajorityVoting(agents=agents, max_loops=max_loops)
    _record(mv.consensus_agent, calls)
    return mv, calls


@pytest.mark.asyncio
async def test_majority_voting_arun_basic():
    mv, calls = _voting_system(["Voter-A", "Voter-B"], max_loops=1)
    result = await mv.arun("Calculate 2+2")
    assert result is not None
    voter_calls = [c for c in calls if c["agent"] in ("Voter-A", "Voter-B")]
    assert len(voter_calls) == 2
    assert all(c["async"] is True for c in voter_calls)
    consensus_calls = [
        c for c in calls if c["agent"] == mv.consensus_agent.agent_name
    ]
    assert len(consensus_calls) == 1


@pytest.mark.asyncio
async def test_majority_voting_arun_multiple_loops():
    mv, calls = _voting_system(["Voter-1", "Voter-2"], max_loops=2)
    result = await mv.arun("Discuss pros and cons")
    assert result is not None
    voter_calls = [c for c in calls if c["agent"] in ("Voter-1", "Voter-2")]
    assert len(voter_calls) == 4
    consensus_calls = [
        c for c in calls if c["agent"] == mv.consensus_agent.agent_name
    ]
    assert len(consensus_calls) == 2


@pytest.mark.asyncio
async def test_majority_voting_arun_voter_failure_isolation():
    mv, calls = _voting_system(["Voter-A", "Voter-B"], max_loops=1)

    async def _failing_arun(task=None, messages=None, **kwargs):
        raise RuntimeError("Voter A connection dropped")

    mv.agents[0].arun = _failing_arun

    result = await mv.arun("Resilience test")
    assert result is not None

    voter_b_calls = [c for c in calls if c["agent"] == "Voter-B"]
    assert len(voter_b_calls) == 1
    consensus_calls = [
        c for c in calls if c["agent"] == mv.consensus_agent.agent_name
    ]
    assert len(consensus_calls) == 1


@pytest.mark.asyncio
async def test_majority_voting_arun_streaming_callback():
    mv, calls = _voting_system(["Voter-1", "Voter-2"], max_loops=1)
    streamed_chunks = []

    def streaming_callback(agent_name, chunk, completed):
        streamed_chunks.append((agent_name, chunk, completed))

    consensus = mv.consensus_agent

    async def _streaming_arun(
        task=None, messages=None, streaming_callback=None, **kwargs
    ):
        if streaming_callback:
            streaming_callback("chunk-1")
            streaming_callback("chunk-2")
        answer = "Final async consensus answer"
        consensus.short_memory.add(role=consensus.agent_name, content=answer)
        return answer

    consensus.arun = _streaming_arun

    result = await mv.arun("Test stream", streaming_callback=streaming_callback)
    assert result is not None
    assert len(streamed_chunks) == 2
    assert streamed_chunks[0][1] == "chunk-1"
    assert streamed_chunks[1][1] == "chunk-2"


@pytest.mark.asyncio
async def test_majority_voting_arun_sync_agents_fallback():
    mv, calls = _voting_system(["Sync-A", "Sync-B"], max_loops=1)
    for agent in mv.agents:
        delattr(agent, "arun")
    delattr(mv.consensus_agent, "arun")

    result = await mv.arun("Sync agent test")
    assert result is not None
    voter_calls = [c for c in calls if c["agent"] in ("Sync-A", "Sync-B")]
    assert len(voter_calls) == 2
    assert all(c["async"] is False for c in voter_calls)
    consensus_calls = [
        c for c in calls if c["agent"] == mv.consensus_agent.agent_name
    ]
    assert len(consensus_calls) == 1
    assert consensus_calls[0]["async"] is False


@pytest.mark.asyncio
async def test_majority_voting_batch_arun():
    mv, calls = _voting_system(["Voter-1", "Voter-2"], max_loops=1)
    tasks = ["Task 1", "Task 2"]
    results = await mv.batch_arun(tasks)
    assert len(results) == 2
    assert all(r is not None for r in results)


@pytest.mark.asyncio
async def test_majority_voting_arun_max_concurrency():
    mv, calls = _voting_system(["Voter-1", "Voter-2", "Voter-3"], max_loops=1)
    result = await mv.arun("Calculate with concurrency cap", max_concurrency=1)
    assert result is not None
    voter_calls = [c for c in calls if c["agent"] in ("Voter-1", "Voter-2", "Voter-3")]
    assert len(voter_calls) == 3


@pytest.mark.asyncio
async def test_majority_voting_arun_sync_agent_fallback():
    mv, calls = _voting_system(["Voter-Sync"], max_loops=1)
    # Remove arun so it must fall back to sync run via asyncio.to_thread
    del mv.agents[0].arun
    result = await mv.arun("Test sync agent fallback")
    assert result is not None
    voter_calls = [c for c in calls if c["agent"] == "Voter-Sync"]
    assert len(voter_calls) == 1
    assert voter_calls[0]["async"] is False
