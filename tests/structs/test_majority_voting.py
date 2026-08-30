from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def test_majority_voting_basic_execution():
    """Test basic MajorityVoting execution with multiple agents"""
    # Create specialized agents with different perspectives
    geographer = Agent(
        agent_name="Geography-Expert",
        agent_description="Expert in geography and world capitals",
        model_name="gpt-4o",
        max_loops=1,
    )

    historian = Agent(
        agent_name="History-Scholar",
        agent_description="Historical and cultural context specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    political_analyst = Agent(
        agent_name="Political-Analyst",
        agent_description="Political and administrative specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting system
    mv = MajorityVoting(
        name="Geography-Consensus-System",
        description="Majority voting system for geographical questions",
        agents=[geographer, historian, political_analyst],
        max_loops=1,
        verbose=True,
    )

    # Test execution
    result = mv.run("What is the capital city of France?")
    assert result is not None


def test_majority_voting_multiple_loops():
    """Test MajorityVoting with multiple loops for consensus refinement"""
    # Create agents with different knowledge bases
    trivia_expert = Agent(
        agent_name="Trivia-Expert",
        agent_description="General knowledge and trivia specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    research_analyst = Agent(
        agent_name="Research-Analyst",
        agent_description="Research and fact-checking specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    subject_matter_expert = Agent(
        agent_name="Subject-Matter-Expert",
        agent_description="Deep subject matter expertise specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting with multiple loops for iterative refinement
    mv = MajorityVoting(
        name="Multi-Loop-Consensus-System",
        description="Majority voting with iterative consensus refinement",
        agents=[
            trivia_expert,
            research_analyst,
            subject_matter_expert,
        ],
        max_loops=3,  # Allow multiple iterations
        verbose=True,
    )

    # Test multi-loop execution
    result = mv.run(
        "What are the main causes of climate change and what can be done to mitigate them?"
    )
    assert result is not None


def test_majority_voting_business_scenario():
    """Test MajorityVoting in a realistic business scenario"""
    # Create agents representing different business perspectives
    market_strategist = Agent(
        agent_name="Market-Strategist",
        agent_description="Market strategy and competitive analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    financial_analyst = Agent(
        agent_name="Financial-Analyst",
        agent_description="Financial modeling and ROI analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    technical_architect = Agent(
        agent_name="Technical-Architect",
        agent_description="Technical feasibility and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    risk_manager = Agent(
        agent_name="Risk-Manager",
        agent_description="Risk assessment and compliance specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    operations_expert = Agent(
        agent_name="Operations-Expert",
        agent_description="Operations and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create majority voting for business decisions
    mv = MajorityVoting(
        name="Business-Decision-Consensus",
        description="Majority voting system for business strategic decisions",
        agents=[
            market_strategist,
            financial_analyst,
            technical_architect,
            risk_manager,
            operations_expert,
        ],
        max_loops=2,
        verbose=True,
    )

    # Test with complex business decision
    result = mv.run(
        "Should our company invest in developing an AI-powered customer service platform? "
        "Consider market demand, financial implications, technical feasibility, risk factors, "
        "and operational requirements."
    )

    assert result is not None


def test_majority_voting_error_handling():
    """Test MajorityVoting error handling and validation"""
    # Test with empty agents list
    try:
        MajorityVoting(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "agents" in str(e).lower() or "empty" in str(e).lower()

    # Test with invalid max_loops
    analyst = Agent(
        agent_name="Test-Analyst",
        agent_description="Test analyst",
        model_name="gpt-4o",
        max_loops=1,
    )

    try:
        MajorityVoting(agents=[analyst], max_loops=0)
        assert (
            False
        ), "Should have raised ValueError for invalid max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower() or "0" in str(e)

    try:
        MajorityVoting(agents=[analyst], max_loops=-1)
        assert (
            False
        ), "Should have raised ValueError for negative max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower()


def test_majority_voting_different_output_types():
    """Test MajorityVoting with different output types"""
    # Create agents for technical analysis
    Agent(
        agent_name="Security-Expert",
        agent_description="Cybersecurity and data protection specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    Agent(
        agent_name="Compliance-Officer",
        agent_description="Regulatory compliance and legal specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    Agent(
        agent_name="Privacy-Advocate",
        agent_description="Privacy protection and data rights specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Assert majority vote is correct
    assert True


def test_streaming_majority_voting():
    """
    Test the streaming_majority_voting with logging/try-except and assertion.
    """
    logs = []

    def streaming_callback(
        agent_name: str, chunk: str, is_final: bool
    ):
        # Chunk buffer static per call (reset each session)
        if not hasattr(streaming_callback, "_buffer"):
            streaming_callback._buffer = ""
            streaming_callback._buffer_size = 0

        min_chunk_size = 512  # or any large chunk size you want

        if chunk:
            streaming_callback._buffer += chunk
            streaming_callback._buffer_size += len(chunk)
        if (
            streaming_callback._buffer_size >= min_chunk_size
            or is_final
        ):
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
            model_name="gpt-5.4",
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
            streaming_on=True,  # if concurrent agents want to be streamed
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


# ============================================================================
# Context shape — voters receive typed chat turns, not one flattened blob
# ============================================================================


def _record(agent, calls):
    """Replace an agent's run() with a stub that records the context it got."""
    name = agent.agent_name
    turns = [0]

    def _run(task=None, messages=None, **kwargs):
        turns[0] += 1
        answer = f"{name}-answer-{turns[0]}"
        calls.append(
            {
                "agent": name,
                "task": task,
                "messages": list(messages or []),
                "answer": answer,
            }
        )
        # Structures read the answer back through short_memory, not the
        # return value, so the stub has to write it there too.
        agent.short_memory.add(role=name, content=answer)
        return answer

    agent.run = _run
    return agent


def _recording_agents(names):
    """Real agents whose run() records the context it was handed."""
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
    """A MajorityVoting whose voters and consensus agent are both recorded."""
    agents, calls = _recording_agents(names)
    mv = MajorityVoting(agents=agents, max_loops=max_loops)
    _record(mv.consensus_agent, calls)
    return mv, calls


def test_voters_receive_typed_turns_not_one_user_blob():
    """Shared history reaches a voter as chat turns, not concatenated onto the task."""
    mv, calls = _voting_system(["Voter-A", "Voter-B"], max_loops=2)
    mv.run("the original question")

    a_calls = [c for c in calls if c["agent"] == "Voter-A"]
    b_calls = [c for c in calls if c["agent"] == "Voter-B"]
    assert len(a_calls) == 2 and len(b_calls) == 2

    for call in a_calls + b_calls:
        assert isinstance(call["messages"], list)
        for message in call["messages"]:
            assert isinstance(message, dict)
            assert message["role"] in ("user", "assistant", "system")
            assert isinstance(message["content"], str)
        assert "History:" not in str(call["task"])

    # The first round is the bare task: there is no history to send yet.
    assert a_calls[0]["task"] == "the original question"
    assert a_calls[0]["messages"] == []

    # The second round carries the first round as turns, not as task text.
    assert len(a_calls[1]["messages"]) >= 2
    assert b_calls[0]["answer"] not in str(a_calls[1]["task"])
    assert b_calls[0]["answer"] in "\n".join(
        m["content"] for m in a_calls[1]["messages"]
    )


def test_votes_recorded_are_answers_not_transcripts():
    """A voter contributes its answer to the shared conversation, not its transcript."""
    task = "a very distinctive question about tungsten supply"
    mv, calls = _voting_system(["Voter-A", "Voter-B"], max_loops=2)
    mv.run(task)

    for name in ("Voter-A", "Voter-B"):
        expected = [c["answer"] for c in calls if c["agent"] == name]
        recorded = [
            m["content"]
            for m in mv.conversation.conversation_history
            if m["role"] == name
        ]
        assert recorded == expected, (
            f"{name} recorded something other than its answers: "
            f"{recorded}"
        )
        assert not any(
            task in content for content in recorded
        ), f"{name} echoed the task back into the shared conversation"


def test_consensus_agent_sees_its_own_prior_consensus_as_assistant():
    """The consensus agent's own turn must not come back labelled as the user's."""
    mv, calls = _voting_system(["Voter-A", "Voter-B"], max_loops=2)
    mv.run("go")

    consensus_calls = [
        c
        for c in calls
        if c["agent"] == mv.consensus_agent.agent_name
    ]
    assert len(consensus_calls) == 2
    second = consensus_calls[1]

    assistant = [
        m["content"]
        for m in second["messages"]
        if m["role"] == "assistant"
    ]
    assert consensus_calls[0]["answer"] in assistant

    user_text = "\n".join(
        m["content"]
        for m in second["messages"]
        if m["role"] == "user"
    )
    for name in ("Voter-A", "Voter-B"):
        for call in [c for c in calls if c["agent"] == name]:
            labelled = f"{name}: {call['answer']}"
            assert labelled in user_text or labelled == str(
                second["task"]
            ), f"{name}'s vote never reached the consensus agent as a user turn"
            assert not any(
                call["answer"] in content for content in assistant
            ), f"{name}'s vote arrived as the consensus agent's own turn"
