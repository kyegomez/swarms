import json
from unittest.mock import MagicMock

from swarms.structs.agent import Agent



def _make_agent() -> Agent:
    """Create an offline-friendly agent for autonomous-loop unit tests."""
    agent = Agent(
        llm=MagicMock(),
        max_loops="auto",
        print_on=False,
        verbose=False,
        reasoning_prompt_on=False,
        selected_tools=[
            "create_plan",
            "think",
            "subtask_done",
            "complete_task",
        ],
    )
    # Keep autonomous-loop tests offline by preventing LLM reinitialization.
    agent.llm_handling = MagicMock(return_value=agent.llm)
    return agent



def test_generate_final_summary_forwards_streaming_callback_to_call_llm():
    agent = _make_agent()
    callback = lambda _token: None

    agent.call_llm = MagicMock(return_value="summary output")
    agent.parse_llm_output = MagicMock(side_effect=lambda x: x)

    result = agent._generate_final_summary(streaming_callback=callback)

    assert result == "summary output"
    assert agent.call_llm.call_count == 1
    assert (
        agent.call_llm.call_args.kwargs["streaming_callback"]
        is callback
    )



def test_generate_final_summary_streams_tokens_via_callback():
    agent = _make_agent()
    received_tokens = []

    def callback(token: str):
        received_tokens.append(token)

    def fake_call_llm(**kwargs):
        kwargs["streaming_callback"]("tok-1")
        kwargs["streaming_callback"]("tok-2")
        return "final summary"

    agent.call_llm = fake_call_llm
    agent.parse_llm_output = MagicMock(side_effect=lambda x: x)

    result = agent._generate_final_summary(streaming_callback=callback)

    assert result == "final summary"
    assert received_tokens == ["tok-1", "tok-2"]



def test_autonomous_loop_passes_callback_to_summary_in_both_paths():
    callback = lambda _token: None
    planning_tool_call = [
        {
            "function": {
                "name": "create_plan",
                "arguments": json.dumps(
                    {
                        "task_description": "task",
                        "steps": [
                            {
                                "step_id": "s1",
                                "description": "step",
                                "priority": "high",
                                "dependencies": [],
                            }
                        ],
                    }
                ),
            }
        }
    ]

    # Path A: normal phase-3 summary path
    normal_agent = _make_agent()
    normal_agent.call_llm = MagicMock(return_value=planning_tool_call)
    normal_agent.parse_llm_output = MagicMock(side_effect=lambda x: x)
    normal_agent._all_subtasks_complete = MagicMock(return_value=True)
    normal_agent._generate_final_summary = MagicMock(
        return_value="normal-summary"
    )

    normal_result = normal_agent._run_autonomous_loop(
        task="task",
        streaming_callback=callback,
    )

    assert normal_result == "normal-summary"
    normal_agent._generate_final_summary.assert_called_once_with(
        streaming_callback=callback
    )

    # Path B: early return when complete_task is emitted in execution phase
    early_agent = _make_agent()
    early_agent.call_llm = MagicMock(
        side_effect=[
            planning_tool_call,
            [
                {
                    "function": {
                        "name": "complete_task",
                        "arguments": json.dumps(
                            {
                                "task_id": "main_task",
                                "summary": "done",
                                "success": True,
                            }
                        ),
                    }
                }
            ],
        ]
    )
    early_agent.parse_llm_output = MagicMock(side_effect=lambda x: x)
    early_agent._generate_final_summary = MagicMock(
        return_value="early-summary"
    )

    early_result = early_agent._run_autonomous_loop(
        task="task",
        streaming_callback=callback,
    )

    assert early_result == "early-summary"
    early_agent._generate_final_summary.assert_called_once_with(
        streaming_callback=callback
    )
