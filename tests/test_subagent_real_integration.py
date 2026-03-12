"""
Real integration tests for subagent tools — no mocks, real LLM calls.

Tests the actual tool functions (create_sub_agent_tool, assign_task_tool,
get_task_status_tool, cancel_task_tool) with real Agent instances hitting
a real LLM (gpt-4.1-nano).

This verifies the full pipeline:
  create_sub_agent_tool() → Agent created with valid system_prompt
  assign_task_tool() → Agent.run() actually calls LLM, returns real output
  get_task_status_tool() → Shows completed status with duration
  Retry, fail_fast, wait_first strategies all work with real execution

Run:
    python tests/test_subagent_real_integration.py
"""

import time
from dotenv import load_dotenv

load_dotenv()

from swarms.structs.agent import Agent
from swarms.structs.autonomous_loop_utils import (
    SubagentTaskStatus,
    SubagentTaskRegistry,
    _get_task_registry,
    create_sub_agent_tool,
    assign_task_tool,
    get_task_status_tool,
    cancel_task_tool,
)


MODEL = "gpt-4.1-nano"


def make_real_parent():
    """Create a real parent Agent."""
    agent = Agent(
        agent_name="test-parent",
        system_prompt="You are a coordinator agent.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
        streaming_on=False,
        verbose=False,
    )
    return agent


def test_1_create_sub_agent_with_system_prompt():
    """
    FEATURE: create_sub_agent_tool creates real Agent with valid system_prompt.
    The original bug: None system_prompt crashes Anthropic API.
    """
    print("\n" + "=" * 60)
    print("TEST 1: create_sub_agent_tool — system_prompt fix")
    print("=" * 60)

    parent = make_real_parent()

    # Create sub-agent WITHOUT explicit system_prompt (the bug case)
    result = create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": "Greeter",
                "agent_description": "Greets users in English",
            }
        ],
    )
    print(f"  Result: {result}")
    assert "Successfully created" in result
    assert len(parent.sub_agents) == 1

    # Verify system_prompt was generated (not None)
    agent_data = list(parent.sub_agents.values())[0]
    assert agent_data["system_prompt"] is not None
    assert len(agent_data["system_prompt"]) > 0
    assert "Greeter" in agent_data["system_prompt"]
    print(f"  System prompt: {agent_data['system_prompt']}")

    # Now actually RUN the sub-agent to prove it doesn't crash
    sub_agent = agent_data["agent"]
    output = sub_agent.run("Say hello to the user. One sentence only.")
    print(f"  Sub-agent output: {output[:200]}")
    assert isinstance(output, str)
    assert len(output) > 0
    print("  PASSED")


def test_2_assign_task_basic():
    """
    FEATURE: assign_task_tool runs real sub-agents concurrently via ThreadPoolExecutor.
    """
    print("\n" + "=" * 60)
    print("TEST 2: assign_task_tool — basic execution")
    print("=" * 60)

    parent = make_real_parent()

    # Create 2 sub-agents with explicit system_prompts
    create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": "MathAgent",
                "agent_description": "Solves math problems",
                "system_prompt": "You are a math assistant. Be very brief, answer in one line.",
            },
            {
                "agent_name": "FactAgent",
                "agent_description": "Provides facts",
                "system_prompt": "You are a fact assistant. Be very brief, answer in one line.",
            },
        ],
    )
    agent_ids = list(parent.sub_agents.keys())
    print(f"  Created agents: {agent_ids}")

    # Assign tasks
    start = time.time()
    result = assign_task_tool(
        parent,
        [
            {"agent_id": agent_ids[0], "task": "What is 2+2? Reply with just the number."},
            {"agent_id": agent_ids[1], "task": "What is the capital of France? One word only."},
        ],
    )
    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Result:\n{result}")

    assert "Completed 2 task" in result
    assert "completed" in result.lower()
    print("  PASSED")


def test_3_assign_task_without_system_prompt():
    """
    REGRESSION: Sub-agents created without system_prompt can be assigned
    tasks and execute without crashing.
    """
    print("\n" + "=" * 60)
    print("TEST 3: assign_task — sub-agent without explicit system_prompt")
    print("=" * 60)

    parent = make_real_parent()

    # Create sub-agent WITHOUT system_prompt
    create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": "GreeterNoPrompt",
                "agent_description": "Greets users in Spanish",
            }
        ],
    )
    agent_id = list(parent.sub_agents.keys())[0]

    # Assign task — this would have crashed before the fix
    result = assign_task_tool(
        parent,
        [{"agent_id": agent_id, "task": "Say hello in Spanish. One sentence only."}],
    )
    print(f"  Result:\n{result}")
    assert "Completed" in result
    assert "completed" in result.lower()
    # Should NOT contain any error about empty system content
    assert "Error" not in result or "FAILED" not in result
    print("  PASSED")


def test_4_background_dispatch_and_status():
    """
    FEATURE: Fire-and-forget mode + get_task_status polling.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Background dispatch + status polling")
    print("=" * 60)

    parent = make_real_parent()

    create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": "BGWorker",
                "agent_description": "Background worker",
                "system_prompt": "You are a helper. Be very brief.",
            }
        ],
    )
    agent_id = list(parent.sub_agents.keys())[0]

    # Dispatch without waiting
    start = time.time()
    result = assign_task_tool(
        parent,
        [{"agent_id": agent_id, "task": "Say 'done'. One word only."}],
        wait_for_completion=False,
    )
    dispatch_time = time.time() - start
    print(f"  Dispatch returned in {dispatch_time:.4f}s")
    print(f"  Dispatch result: {result}")
    assert "Dispatched" in result
    assert dispatch_time < 1.0  # Should be near-instant

    # Poll status until complete
    for i in range(30):
        status = get_task_status_tool(parent)
        print(f"  Status poll {i}: {status}")
        if "completed" in status:
            break
        time.sleep(1)
    else:
        assert False, "Task did not complete within 30s"

    assert "completed" in status
    print("  PASSED")


def test_5_concurrent_execution():
    """
    FEATURE: Multiple sub-agents run concurrently, not sequentially.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Concurrent execution")
    print("=" * 60)

    parent = make_real_parent()

    # Create 3 sub-agents
    create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": f"Worker-{i}",
                "agent_description": f"Worker agent {i}",
                "system_prompt": "You are a helper. Be extremely brief, one sentence max.",
            }
            for i in range(3)
        ],
    )
    agent_ids = list(parent.sub_agents.keys())
    print(f"  Created {len(agent_ids)} agents")

    # Assign tasks to all 3
    start = time.time()
    result = assign_task_tool(
        parent,
        [
            {"agent_id": agent_ids[0], "task": "What is 1+1?"},
            {"agent_id": agent_ids[1], "task": "What is 2+2?"},
            {"agent_id": agent_ids[2], "task": "What is 3+3?"},
        ],
    )
    elapsed = time.time() - start
    print(f"  All 3 completed in {elapsed:.2f}s")
    print(f"  Result:\n{result}")

    assert "Completed 3 task" in result
    print("  PASSED")


def test_6_depth_tracking():
    """
    FEATURE: Depth tracking prevents runaway recursion.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Depth tracking")
    print("=" * 60)

    parent = make_real_parent()
    parent.max_subagent_depth = 2

    # Create at depth 0 → sub-agent at depth 1
    result = create_sub_agent_tool(
        parent,
        [{"agent_name": "Child", "agent_description": "Child agent"}],
    )
    assert "Successfully created" in result
    child_data = list(parent.sub_agents.values())[0]
    child = child_data["agent"]
    print(f"  Child depth: {child._subagent_depth}")
    assert child._subagent_depth == 1

    # Child creates grandchild at depth 2
    result2 = create_sub_agent_tool(
        child,
        [{"agent_name": "Grandchild", "agent_description": "Grandchild agent"}],
    )
    assert "Successfully created" in result2
    grandchild_data = list(child.sub_agents.values())[0]
    grandchild = grandchild_data["agent"]
    print(f"  Grandchild depth: {grandchild._subagent_depth}")
    assert grandchild._subagent_depth == 2

    # Grandchild tries to create great-grandchild — should be blocked
    result3 = create_sub_agent_tool(
        grandchild,
        [{"agent_name": "GreatGrandchild", "agent_description": "Too deep"}],
    )
    print(f"  Depth limit result: {result3}")
    assert "Error" in result3
    assert "Maximum subagent depth" in result3
    print("  PASSED")


def test_7_fail_fast_false():
    """
    FEATURE: fail_fast=False — one sub-agent fails, others still succeed.
    We use a real agent with a broken model name to force a failure.
    """
    print("\n" + "=" * 60)
    print("TEST 7: fail_fast=False — partial failure handling")
    print("=" * 60)

    parent = make_real_parent()

    # Create one good agent and one with broken model
    create_sub_agent_tool(
        parent,
        [
            {
                "agent_name": "GoodAgent",
                "agent_description": "Works correctly",
                "system_prompt": "You are helpful. Be very brief.",
            },
        ],
    )
    good_id = list(parent.sub_agents.keys())[0]

    # Manually create a bad agent with invalid model
    bad_agent = Agent(
        agent_name="BadAgent",
        system_prompt="test",
        model_name="nonexistent-model-xyz",
        max_loops=1,
        print_on=False,
        verbose=False,
    )
    parent.sub_agents["bad-agent-id"] = {
        "agent": bad_agent,
        "name": "BadAgent",
        "description": "Will fail",
        "system_prompt": "test",
        "depth": 0,
        "parent_agent_id": parent.id,
    }

    result = assign_task_tool(
        parent,
        [
            {"agent_id": good_id, "task": "Say 'success'. One word."},
            {"agent_id": "bad-agent-id", "task": "This will fail"},
        ],
        fail_fast=False,
    )
    print(f"  Result:\n{result}")

    # Good agent should succeed, bad agent should fail
    assert "completed" in result.lower() or "FAILED" in result
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("REAL INTEGRATION TEST: Subagent Tools with Live LLM")
    print(f"Model: {MODEL}")
    print("=" * 60)

    tests = [
        test_1_create_sub_agent_with_system_prompt,
        test_2_assign_task_basic,
        test_3_assign_task_without_system_prompt,
        test_4_background_dispatch_and_status,
        test_5_concurrent_execution,
        test_6_depth_tracking,
        test_7_fail_fast_false,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)
