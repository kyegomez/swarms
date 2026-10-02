from swarms import Agent


def dummy_tool(x: int) -> int:
    """Double x."""
    return x * 2


def test_tool_schemas_not_added_as_assistant_turn():
    """Verify Issue #1989: tool schemas must not be duplicated into short_memory."""
    agent = Agent(
        agent_name="ToolTester",
        model_name="gpt-4o-mini",
        tools=[dummy_tool],
        dynamic_tools=False,
        max_loops=1,
    )

    # Tool should be registered in tools_list_dictionary
    assert agent.tools_list_dictionary is not None
    assert len(agent.tools_list_dictionary) > 0
    assert agent.tools_list_dictionary[0]["function"]["name"] == "dummy_tool"

    # short_memory must NOT contain a turn with role=agent_name reciting the schema
    roles = [msg.get("role") for msg in agent.short_memory.conversation_history]
    assert "ToolTester" not in roles
    for msg in agent.short_memory.conversation_history:
        content = msg.get("content")
        assert not isinstance(content, list), "Tool schema list should not be a message content"
