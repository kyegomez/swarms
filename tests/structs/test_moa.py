import re
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.agent import Agent


def test_mixture_of_agents_basic_initialization():
    """Test basic MixtureOfAgents initialization with multiple agents"""
    # Create multiple specialized agents
    research_agent = Agent(
        agent_name="Research-Specialist",
        agent_description="Specialist in research and data collection",
        model_name="gpt-4o",
        max_loops=1,
    )

    analysis_agent = Agent(
        agent_name="Analysis-Expert",
        agent_description="Expert in data analysis and insights",
        model_name="gpt-4o",
        max_loops=1,
    )

    strategy_agent = Agent(
        agent_name="Strategy-Consultant",
        agent_description="Strategy and planning consultant",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator agent
    aggregator = Agent(
        agent_name="Aggregator-Agent",
        agent_description="Agent that aggregates responses from other agents",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture of agents
    moa = MixtureOfAgents(
        name="Business-Analysis-Mixture",
        description="Mixture of agents for comprehensive business analysis",
        agents=[research_agent, analysis_agent, strategy_agent],
        aggregator_agent=aggregator,
        layers=3,
        max_loops=1,
    )

    # Verify initialization
    assert moa.name == "Business-Analysis-Mixture"
    assert (
        moa.description
        == "Mixture of agents for comprehensive business analysis"
    )
    assert len(moa.agents) == 3
    assert moa.aggregator_agent == aggregator
    assert moa.layers == 3
    assert moa.max_loops == 1


def test_mixture_of_agents_execution():
    """Test MixtureOfAgents execution with multiple agents"""
    # Create diverse agents for different perspectives
    market_analyst = Agent(
        agent_name="Market-Analyst",
        agent_description="Market analysis and trend specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    technical_expert = Agent(
        agent_name="Technical-Expert",
        agent_description="Technical feasibility and implementation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    financial_analyst = Agent(
        agent_name="Financial-Analyst",
        agent_description="Financial modeling and ROI specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    risk_assessor = Agent(
        agent_name="Risk-Assessor",
        agent_description="Risk assessment and mitigation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator for synthesis
    aggregator = Agent(
        agent_name="Executive-Summary-Agent",
        agent_description="Executive summary and recommendation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture of agents
    moa = MixtureOfAgents(
        name="Comprehensive-Evaluation-Mixture",
        description="Mixture of agents for comprehensive business evaluation",
        agents=[
            market_analyst,
            technical_expert,
            financial_analyst,
            risk_assessor,
        ],
        aggregator_agent=aggregator,
        layers=2,
        max_loops=1,
    )

    # Test execution
    result = moa.run(
        "Evaluate the feasibility of launching an AI-powered healthcare platform"
    )
    assert result is not None


def test_mixture_of_agents_multiple_layers():
    """Test MixtureOfAgents with multiple layers"""
    # Create agents for layered analysis
    data_collector = Agent(
        agent_name="Data-Collector",
        agent_description="Data collection and research specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    pattern_analyzer = Agent(
        agent_name="Pattern-Analyzer",
        agent_description="Pattern recognition and analysis specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    insight_generator = Agent(
        agent_name="Insight-Generator",
        agent_description="Insight generation and interpretation specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator
    final_aggregator = Agent(
        agent_name="Final-Aggregator",
        agent_description="Final aggregation and conclusion specialist",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create mixture with multiple layers for deeper analysis
    moa = MixtureOfAgents(
        name="Multi-Layer-Analysis-Mixture",
        description="Mixture of agents with multiple analysis layers",
        agents=[data_collector, pattern_analyzer, insight_generator],
        aggregator_agent=final_aggregator,
        layers=4,
        max_loops=1,
    )

    # Test multi-layer execution
    result = moa.run(
        "Analyze customer behavior patterns and provide strategic insights"
    )
    assert result is not None


def test_mixture_of_agents_error_handling():
    """Test MixtureOfAgents error handling and validation"""
    # Test with empty agents list
    try:
        MixtureOfAgents(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "No agents provided" in str(e)

    # Test with invalid aggregator system prompt
    analyst = Agent(
        agent_name="Test-Analyst",
        agent_description="Test analyst",
        model_name="gpt-4o",
        max_loops=1,
    )

    try:
        MixtureOfAgents(agents=[analyst], aggregator_system_prompt="")
        assert (
            False
        ), "Should have raised ValueError for empty system prompt"
    except ValueError as e:
        assert "No aggregator system prompt" in str(e)


def test_mixture_of_agents_real_world_scenario():
    """Test MixtureOfAgents in a realistic business scenario"""
    # Create agents representing different business functions
    marketing_director = Agent(
        agent_name="Marketing-Director",
        agent_description="Senior marketing director with market expertise",
        model_name="gpt-4o",
        max_loops=1,
    )

    product_manager = Agent(
        agent_name="Product-Manager",
        agent_description="Product strategy and development manager",
        model_name="gpt-4o",
        max_loops=1,
    )

    engineering_lead = Agent(
        agent_name="Engineering-Lead",
        agent_description="Senior engineering and technical architecture lead",
        model_name="gpt-4o",
        max_loops=1,
    )

    sales_executive = Agent(
        agent_name="Sales-Executive",
        agent_description="Enterprise sales and customer relationship executive",
        model_name="gpt-4o",
        max_loops=1,
    )

    legal_counsel = Agent(
        agent_name="Legal-Counsel",
        agent_description="Legal compliance and regulatory counsel",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create aggregator for executive decision making
    executive_aggregator = Agent(
        agent_name="Executive-Decision-Maker",
        agent_description="Executive decision maker and strategic aggregator",
        model_name="gpt-4o",
        max_loops=1,
    )

    # Create comprehensive mixture of agents
    moa = MixtureOfAgents(
        name="Executive-Board-Mixture",
        description="Mixture of agents representing executive board for strategic decisions",
        agents=[
            marketing_director,
            product_manager,
            engineering_lead,
            sales_executive,
            legal_counsel,
        ],
        aggregator_agent=executive_aggregator,
        layers=3,
        max_loops=1,
    )

    # Test with complex business scenario
    result = moa.run(
        "Develop a comprehensive go-to-market strategy for our new AI-powered enterprise platform. "
        "Consider market positioning, technical requirements, competitive landscape, sales channels, "
        "and legal compliance requirements."
    )

    assert result is not None


# ============================================================================
# Context management: what each worker contributes to the shared conversation
# ============================================================================


def _scripted_agent(name):
    """A real Agent whose LLM call is stubbed, so short_memory is real."""
    from swarms import Agent

    agent = Agent(
        agent_name=name,
        model_name="gpt-4o-mini",
        max_loops=1,
        persistent_memory=False,
        print_on=False,
        autosave=False,
    )
    agent.call_llm = lambda task=None, *a, **k: f"[{name}-out]"
    return agent


class TestWorkerContributions:
    """
    Workers must contribute their answer, not their whole transcript.

    ``agent.run`` honours each agent's output_type, which defaults to
    "str-all-except-first" - the agent's entire conversation. Recording that
    re-injected the task and the previous layer's synthesis into the shared
    history, so later layers and the aggregator read the same text repeatedly.
    """

    def test_workers_record_only_their_answer(self):
        from swarms import MixtureOfAgents

        moa = MixtureOfAgents(
            agents=[_scripted_agent("W1"), _scripted_agent("W2")],
            aggregator_agent=_scripted_agent("Agg"),
            layers=2,
        )
        moa.run("Question?")

        worker_messages = [
            m["content"]
            for m in moa.conversation.conversation_history
            if m["role"].startswith(("W1", "W2"))
        ]
        assert worker_messages == [
            "[W1-out]",
            "[W2-out]",
            "[W1-out]",
            "[W2-out]",
        ], f"workers recorded more than their answers: {worker_messages}"

    def test_the_task_is_not_re_injected_by_workers(self):
        from swarms import MixtureOfAgents

        moa = MixtureOfAgents(
            agents=[_scripted_agent("W1")],
            aggregator_agent=_scripted_agent("Agg"),
            layers=2,
        )
        moa.run("a very distinctive question")

        worker_messages = [
            str(m["content"])
            for m in moa.conversation.conversation_history
            if m["role"] == "W1"
        ]
        assert not any(
            "a very distinctive question" in c
            for c in worker_messages
        ), "a worker echoed the task back into the shared conversation"

    def test_the_conversation_starts_clean(self):
        """
        Conversation used to auto-load a default-named file from disk, so
        every swarm began with messages left by an unrelated run.
        """
        from swarms import MixtureOfAgents

        moa = MixtureOfAgents(
            agents=[_scripted_agent("W1")],
            aggregator_agent=_scripted_agent("Agg"),
            layers=1,
        )
        roles = [
            m["role"] for m in moa.conversation.conversation_history
        ]
        assert (
            "user" not in roles
        ), f"stale messages loaded from disk: {roles}"


# ============================================================================
# Context shape — the aggregator receives typed chat turns, not one blob
# ============================================================================


def _recording_agents(names):
    """Real agents whose run() records the context it was handed."""
    calls = []
    agents = []
    for name in names:
        agent = Agent(
            agent_name=name,
            model_name="gpt-4o-mini",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
            autosave=False,
        )

        def _make(agent_obj, agent_name):
            turns = [0]

            def _run(task=None, messages=None, **kwargs):
                turns[0] += 1
                answer = f"{agent_name}-answer-{turns[0]}"
                calls.append(
                    {
                        "agent": agent_name,
                        "task": task,
                        "messages": list(messages or []),
                        "answer": answer,
                    }
                )
                # The mixture reads the answer back through short_memory,
                # not the return value, so the stub writes it there too.
                agent_obj.short_memory.add(
                    role=agent_name, content=answer
                )
                return answer

            return _run

        agent.run = _make(agent, name)
        agents.append(agent)
    return agents, calls


def _aggregator_call(calls, name="Aggregator"):
    """The single call handed to the aggregator, plus its turns as one list."""
    aggregator_calls = [c for c in calls if c["agent"] == name]
    assert (
        len(aggregator_calls) == 1
    ), f"the aggregator ran {len(aggregator_calls)} times"
    call = aggregator_calls[0]
    # split_last_turn hands the newest turn over as the task instead of
    # repeating it at the end of messages.
    turns = call["messages"] + [
        {"role": "user", "content": str(call["task"])}
    ]
    return call, turns


def test_aggregator_receives_typed_turns_per_worker():
    """Each worker contribution reaches the aggregator as its own labelled turn.

    The flattened form collapsed every worker into one user string, so the
    aggregator could not tell one contributor from another.
    """
    agents, calls = _recording_agents(["W1", "W2"])
    aggregator, agg_calls = _recording_agents(["Agg"])

    MixtureOfAgents(
        agents=agents,
        aggregator_agent=aggregator[0],
        layers=2,
    ).run("Question?")

    assert len(agg_calls) == 1
    turns = agg_calls[0]["messages"] + [
        {"role": "user", "content": agg_calls[0]["task"]}
    ]

    for message in turns:
        assert isinstance(message, dict)
        assert message["role"] in ("user", "assistant", "system")
        assert isinstance(message["content"], str)

    contents = [m["content"] for m in turns]
    for worker in ("W1", "W2"):
        assert any(
            re.match(rf"^{worker}(?: \(layer \d+/\d+\))?: ", text)
            for text in contents
        ), f"no turn attributed to {worker}: {contents}"

    # Four worker outputs across two layers, plus the task - not one blob.
    assert len(turns) == 5


def test_team_roster_does_not_reach_aggregator_as_user_prose():
    """The roster is structure bookkeeping and belongs in a system prompt."""
    agents, calls = _recording_agents(["W1", "W2", "Aggregator"])
    moa = MixtureOfAgents(
        name="Team Name",
        agents=agents[:2],
        aggregator_agent=agents[2],
        layers=1,
    )
    moa.run("Question?")

    roster = [
        m
        for m in moa.conversation.conversation_history
        if str(m["role"]).lower() == "system"
    ]
    assert roster, "expected list_all_agents to add a System row"

    _, turns = _aggregator_call(calls)
    for message in turns:
        if message["role"] != "user":
            continue
        assert not message["content"].startswith("System: Team Name")
        assert "System: Team Name" not in message["content"]
        assert (
            "These are the agents in your team"
            not in message["content"]
        )


def test_aggregator_can_tell_which_layer_produced_each_answer():
    agents, _ = _recording_agents(["W1", "W2"])
    aggregator, agg_calls = _recording_agents(["Agg"])

    MixtureOfAgents(
        agents=agents,
        aggregator_agent=aggregator[0],
        layers=3,
    ).run("Question?")

    assert agg_calls, "aggregator was never invoked"
    turns = agg_calls[0]["messages"] + [
        {"role": "user", "content": agg_calls[0]["task"]}
    ]
    contents = " ".join(m["content"] for m in turns)

    for layer in (1, 2, 3):
        assert (
            f"layer {layer}/3" in contents
        ), f"layer {layer} not attributed: {contents}"
