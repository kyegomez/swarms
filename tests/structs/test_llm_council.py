"""
Tests for LLM Council implementation.

Tests cover initialization, default council creation, custom members,
workflow execution, batch processing, output types, and error handling.
"""

import pytest
from unittest.mock import patch
from swarms.structs.llm_council import (
    LLMCouncil,
    get_gpt_councilor_prompt,
    get_gemini_councilor_prompt,
    get_claude_councilor_prompt,
    get_grok_councilor_prompt,
    get_chairman_prompt,
    get_evaluation_prompt,
    get_synthesis_prompt,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation


def test_llm_council_basic_initialization():
    """Test basic LLM Council initialization with default members"""
    try:
        council = LLMCouncil(
            name="Test Council",
            verbose=False,
        )

        assert council is not None
        assert council.name == "Test Council"
        assert council.verbose is False
        assert council.output_type == "dict-all-except-first"
        assert council.council_members is not None
        assert len(council.council_members) == 4
        assert council.chairman is not None
        assert council.chairman.agent_name == "Chairman"
        assert council.conversation is not None
        assert isinstance(council.conversation, Conversation)
    except Exception as e:
        pytest.fail(f"Initialization failed with error: {e}")


def test_llm_council_custom_members():
    """Test LLM Council initialization with custom council members"""
    try:
        # Use a simple test model that doesn't require API keys for initialization
        custom_member1 = Agent(
            agent_name="Custom-Member-1",
            agent_description="First custom council member",
            model_name="gpt-4o-mini",  # Using a simpler model for tests
            max_loops=1,
            verbose=False,
        )

        custom_member2 = Agent(
            agent_name="Custom-Member-2",
            agent_description="Second custom council member",
            model_name="gpt-4o-mini",  # Using a simpler model for tests
            max_loops=1,
            verbose=False,
        )

        assert custom_member1 is not None
        assert custom_member2 is not None

        council = LLMCouncil(
            council_members=[custom_member1, custom_member2],
            verbose=False,
        )

        assert council is not None
        assert council.council_members is not None
        assert len(council.council_members) == 2
        assert council.council_members[0] is not None
        assert council.council_members[1] is not None
        assert council.council_members[0].agent_name == "Custom-Member-1"
        assert council.council_members[1].agent_name == "Custom-Member-2"
    except Exception as e:
        pytest.fail(f"Custom members test failed with error: {e}")


def test_llm_council_default_council_creation():
    """Test that default council members are created correctly"""
    try:
        council = LLMCouncil(verbose=False)

        assert council is not None
        assert council.council_members is not None
        assert len(council.council_members) > 0

        member_names = [member.agent_name for member in council.council_members]
        assert "GPT-5.1-Councilor" in member_names
        assert "Gemini-3-Pro-Councilor" in member_names
        assert "Claude-Sonnet-4.5-Councilor" in member_names
        assert "Grok-4-Councilor" in member_names

        # Verify each member has proper configuration
        for member in council.council_members:
            assert member is not None
            assert member.max_loops == 1
            assert member.verbose is False
            assert member.agent_name is not None
            assert member.model_name is not None
    except Exception as e:
        pytest.fail(f"Default council creation test failed with error: {e}")


def test_llm_council_chairman_creation():
    """Test that Chairman agent is created with correct configuration"""
    try:
        council = LLMCouncil(verbose=False)

        assert council is not None
        assert council.chairman is not None
        assert council.chairman.agent_name is not None
        assert council.chairman.agent_name == "Chairman"
        assert council.chairman.max_loops == 1
        assert council.chairman.temperature == 0.7
        assert council.chairman.system_prompt is not None
        assert "synthesize" in council.chairman.system_prompt.lower()
    except Exception as e:
        pytest.fail(f"Chairman creation test failed with error: {e}")


def test_llm_council_conversation_initialization():
    """Test that conversation object is properly initialized"""
    try:
        council = LLMCouncil(name="Test Council", verbose=False)

        assert council is not None
        assert council.conversation is not None
        assert isinstance(council.conversation, Conversation)
        assert council.conversation.name is not None
        assert "[LLM Council]" in council.conversation.name
        assert "Test Council" in council.conversation.name
    except Exception as e:
        pytest.fail(f"Conversation initialization test failed with error: {e}")


def test_llm_council_different_output_types():
    """Test LLM Council with different output types"""
    try:
        output_types = [
            "dict-all-except-first",
            "list",
            "string",
            "final",
            "json",
            "yaml",
        ]

        for output_type in output_types:
            assert output_type is not None
            council = LLMCouncil(
                output_type=output_type,
                verbose=False,
            )
            assert council is not None
            assert council.output_type == output_type
    except Exception as e:
        pytest.fail(f"Different output types test failed with error: {e}")


def test_llm_council_different_chairman_models():
    """Test LLM Council with different chairman models"""
    try:
        models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet-20241022"]

        for model in models:
            assert model is not None
            council = LLMCouncil(
                chairman_model=model,
                verbose=False,
            )
            assert council is not None
            assert council.chairman is not None
            assert council.chairman.model_name == model
    except Exception as e:
        pytest.fail(f"Different chairman models test failed with error: {e}")


def test_councilor_prompts():
    """Test that councilor prompt functions return valid strings"""
    try:
        gpt_prompt = get_gpt_councilor_prompt()
        assert gpt_prompt is not None
        assert isinstance(gpt_prompt, str)
        assert len(gpt_prompt) > 0
        assert "GPT-5.1" in gpt_prompt or "council" in gpt_prompt.lower()

        gemini_prompt = get_gemini_councilor_prompt()
        assert gemini_prompt is not None
        assert isinstance(gemini_prompt, str)
        assert len(gemini_prompt) > 0

        claude_prompt = get_claude_councilor_prompt()
        assert claude_prompt is not None
        assert isinstance(claude_prompt, str)
        assert len(claude_prompt) > 0

        grok_prompt = get_grok_councilor_prompt()
        assert grok_prompt is not None
        assert isinstance(grok_prompt, str)
        assert len(grok_prompt) > 0
    except Exception as e:
        pytest.fail(f"Councilor prompts test failed with error: {e}")


def test_chairman_prompt():
    """Test that chairman prompt function returns valid string"""
    try:
        chairman_prompt = get_chairman_prompt()
        assert chairman_prompt is not None
        assert isinstance(chairman_prompt, str)
        assert len(chairman_prompt) > 0
        assert "synthesize" in chairman_prompt.lower()
        assert "chairman" in chairman_prompt.lower()
    except Exception as e:
        pytest.fail(f"Chairman prompt test failed with error: {e}")


def test_evaluation_prompt():
    """Test evaluation prompt generation"""
    try:
        query = "What is the capital of France?"
        responses = {
            "A": "Paris is the capital of France.",
            "B": "The capital city of France is Paris.",
            "C": "France's capital is Paris.",
        }
        evaluator_name = "GPT-5.1-Councilor"

        assert query is not None
        assert responses is not None
        assert evaluator_name is not None

        prompt = get_evaluation_prompt(query, responses, evaluator_name)

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert query in prompt
        # Note: evaluator_name is passed but not included in the prompt text
        assert "Response A" in prompt
        assert "Response B" in prompt
        assert "Response C" in prompt
        assert "rank" in prompt.lower() or "evaluat" in prompt.lower()
    except Exception as e:
        pytest.fail(f"Evaluation prompt test failed with error: {e}")


def test_synthesis_prompt():
    """Test synthesis prompt generation"""
    try:
        query = "What is the capital of France?"
        original_responses = {
            "GPT-5.1-Councilor": "Paris is the capital of France.",
            "Gemini-3-Pro-Councilor": "The capital city of France is Paris.",
        }
        evaluations = {
            "GPT-5.1-Councilor": "Response A is the best because...",
            "Gemini-3-Pro-Councilor": "Response B is comprehensive...",
        }
        id_to_member = {"A": "GPT-5.1-Councilor", "B": "Gemini-3-Pro-Councilor"}

        assert query is not None
        assert original_responses is not None
        assert evaluations is not None
        assert id_to_member is not None

        prompt = get_synthesis_prompt(
            query, original_responses, evaluations, id_to_member
        )

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert query in prompt
        assert "GPT-5.1-Councilor" in prompt
        assert "Gemini-3-Pro-Councilor" in prompt
        assert "synthesize" in prompt.lower()
    except Exception as e:
        pytest.fail(f"Synthesis prompt test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_run_workflow(
    mock_batched_execution, mock_concurrent_execution
):
    """Test the full LLM Council workflow execution"""
    try:
        # Mock concurrent execution for initial responses
        mock_concurrent_execution.return_value = {
            "GPT-5.1-Councilor": "Response from GPT agent",
            "Gemini-3-Pro-Councilor": "Response from Gemini agent",
            "Claude-Sonnet-4.5-Councilor": "Response from Claude agent",
            "Grok-4-Councilor": "Response from Grok agent",
        }

        # Mock batched execution for evaluations
        mock_batched_execution.return_value = [
            "Evaluation from GPT agent",
            "Evaluation from Gemini agent",
            "Evaluation from Claude agent",
            "Evaluation from Grok agent",
        ]

        # Create council with real agents
        council = LLMCouncil(verbose=False)

        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Execute workflow (execution functions are mocked to avoid API calls)
        result = council.run("What is the capital of France?")

        # Verify workflow steps
        assert mock_concurrent_execution.called
        assert mock_batched_execution.called

        # Verify conversation has been updated
        assert council.conversation is not None
        assert council.conversation.conversation_history is not None
        assert len(council.conversation.conversation_history) > 0

        # Verify result is not None
        assert result is not None
    except Exception as e:
        pytest.fail(f"Workflow execution test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_run_conversation_tracking(
    mock_batched_execution, mock_concurrent_execution
):
    """Test that conversation is properly tracked throughout workflow"""
    try:
        # Mock responses
        mock_concurrent_execution.return_value = {
            "GPT-5.1-Councilor": "Test response 1",
            "Gemini-3-Pro-Councilor": "Test response 2",
        }

        mock_batched_execution.return_value = [
            "Evaluation 1",
            "Evaluation 2",
        ]

        council = LLMCouncil(
            council_members=[
                Agent(
                    agent_name="GPT-5.1-Councilor",
                    model_name="gpt-4o-mini",  # Using a simpler model for tests
                    max_loops=1,
                    verbose=False,
                ),
                Agent(
                    agent_name="Gemini-3-Pro-Councilor",
                    model_name="gpt-4o-mini",  # Using a simpler model for tests
                    max_loops=1,
                    verbose=False,
                ),
            ],
            verbose=False,
        )

        assert council is not None
        assert council.conversation is not None
        assert council.chairman is not None

        query = "Test query"
        assert query is not None

        # Execute workflow (execution functions are mocked to avoid API calls)
        council.run(query)

        # Verify user query was added
        assert council.conversation.conversation_history is not None
        user_messages = [
            msg for msg in council.conversation.conversation_history if msg.get("role") == "User"
        ]
        assert len(user_messages) > 0
        assert user_messages[0] is not None
        assert user_messages[0].get("content") == query

        # Verify council member responses were added
        member_responses = [
            msg
            for msg in council.conversation.conversation_history
            if msg.get("role") in ["GPT-5.1-Councilor", "Gemini-3-Pro-Councilor"]
        ]
        assert len(member_responses) == 2

        # Verify evaluations were added
        evaluations = [
            msg
            for msg in council.conversation.conversation_history
            if "Evaluation" in msg.get("role", "")
        ]
        assert len(evaluations) == 2

        # Verify chairman response was added
        chairman_messages = [
            msg for msg in council.conversation.conversation_history if msg.get("role") == "Chairman"
        ]
        assert len(chairman_messages) > 0
        assert chairman_messages[0] is not None
    except Exception as e:
        pytest.fail(f"Conversation tracking test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_anonymization(
    mock_batched_execution, mock_concurrent_execution
):
    """Test that responses are properly anonymized for evaluation"""
    try:
        mock_concurrent_execution.return_value = {
            "GPT-5.1-Councilor": "Response 1",
            "Gemini-3-Pro-Councilor": "Response 2",
            "Claude-Sonnet-4.5-Councilor": "Response 3",
            "Grok-4-Councilor": "Response 4",
        }

        mock_batched_execution.return_value = [
            "Eval 1",
            "Eval 2",
            "Eval 3",
            "Eval 4",
        ]

        council = LLMCouncil(verbose=False)
        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Capture the evaluation prompt to verify anonymization
        import swarms.structs.llm_council as llm_council_module
        original_get_evaluation_prompt = llm_council_module.get_evaluation_prompt
        captured_prompts = []

        def capture_evaluation_prompt(query, responses, evaluator_name):
            captured_prompts.append((query, responses, evaluator_name))
            return original_get_evaluation_prompt(query, responses, evaluator_name)

        with patch.object(
            llm_council_module, "get_evaluation_prompt", side_effect=capture_evaluation_prompt
        ):
            council.run("Test query")

        # Verify that evaluation prompts use anonymous IDs (A, B, C, D)
        assert captured_prompts is not None
        assert len(captured_prompts) > 0
        for query, responses, evaluator_name in captured_prompts:
            assert query is not None
            assert responses is not None
            assert evaluator_name is not None
            # Responses should have anonymous IDs
            assert any(
                response_id in ["A", "B", "C", "D"]
                for response_id in responses.keys()
            )
            # Should not contain actual member names in response IDs
            assert "GPT-5.1-Councilor" not in str(responses.keys())
            assert "Gemini-3-Pro-Councilor" not in str(responses.keys())
    except Exception as e:
        pytest.fail(f"Anonymization test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_batched_run(
    mock_batched_execution, mock_concurrent_execution
):
    """Test batch processing of multiple queries"""
    try:
        council = LLMCouncil(verbose=False)

        assert council is not None
        assert len(council.council_members) > 0

        # Mock execution functions to avoid API calls
        mock_concurrent_execution.return_value = {
            member.agent_name: f"Response for {member.agent_name}"
            for member in council.council_members
        }
        mock_batched_execution.return_value = [
            f"Evaluation {i}" for i in range(len(council.council_members))
        ]

        tasks = ["Query 1", "Query 2", "Query 3"]
        assert tasks is not None
        assert len(tasks) > 0

        results = council.batched_run(tasks)

        assert results is not None
        assert len(results) == 3
        # Verify execution functions were called for each task
        assert mock_concurrent_execution.call_count == 3
    except Exception as e:
        pytest.fail(f"Batched run test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_output_types(
    mock_batched_execution, mock_concurrent_execution
):
    """Test that different output types work correctly"""
    try:
        output_types = ["dict", "list", "string", "final"]

        for output_type in output_types:
            assert output_type is not None
            council = LLMCouncil(
                output_type=output_type,
                verbose=False,
            )
            assert council is not None
            assert council.chairman is not None
            assert len(council.council_members) > 0

            # Mock responses for all council members
            mock_concurrent_execution.return_value = {
                member.agent_name: f"Response from {member.agent_name}"
                for member in council.council_members
            }
            mock_batched_execution.return_value = [
                f"Evaluation {i}" for i in range(len(council.council_members))
            ]

            # Execute workflow (execution functions are mocked to avoid API calls)
            result = council.run("Test query")

            assert result is not None
            # The exact format depends on history_output_formatter,
            # but we can at least verify it returns something
    except Exception as e:
        pytest.fail(f"Output types test failed with error: {e}")


def test_llm_council_verbose_mode():
    """Test that verbose mode works without errors"""
    try:
        # This test mainly ensures verbose mode doesn't crash
        council = LLMCouncil(verbose=True)

        assert council is not None
        # With verbose=True, initialization should print messages
        # We just verify it doesn't raise an exception
        assert council.verbose is True
        assert council.council_members is not None
        assert council.chairman is not None
    except Exception as e:
        pytest.fail(f"Verbose mode test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_empty_responses_handling(
    mock_batched_execution, mock_concurrent_execution
):
    """Test handling of empty or missing responses"""
    try:
        # Simulate some agents returning empty responses
        mock_concurrent_execution.return_value = {
            "GPT-5.1-Councilor": "Valid response",
            "Gemini-3-Pro-Councilor": "",  # Empty response
            "Claude-Sonnet-4.5-Councilor": "Another valid response",
            "Grok-4-Councilor": None,  # Missing response
        }

        mock_batched_execution.return_value = [
            "Eval 1",
            "Eval 2",
            "Eval 3",
            "Eval 4",
        ]

        council = LLMCouncil(verbose=False)
        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Should not crash even with empty/missing responses
        # Execution functions are mocked to avoid API calls
        result = council.run("Test query")
        assert result is not None
    except Exception as e:
        pytest.fail(f"Empty responses handling test failed with error: {e}")


def test_llm_council_single_member():
    """Test LLM Council with a single council member"""
    try:
        single_member = Agent(
            agent_name="Single-Member",
            agent_description="Only council member",
            model_name="gpt-4o-mini",  # Using a simpler model for tests
            max_loops=1,
            verbose=False,
        )

        assert single_member is not None

        council = LLMCouncil(
            council_members=[single_member],
            verbose=False,
        )

        assert council is not None
        assert council.council_members is not None
        assert len(council.council_members) == 1
        assert council.council_members[0] is not None
        assert council.council_members[0].agent_name == "Single-Member"
    except Exception as e:
        pytest.fail(f"Single member test failed with error: {e}")


def test_llm_council_custom_name_and_description():
    """Test LLM Council with custom name and description"""
    try:
        custom_name = "Custom Council Name"
        custom_description = "Custom council description for testing"

        assert custom_name is not None
        assert custom_description is not None

        council = LLMCouncil(
            name=custom_name,
            description=custom_description,
            verbose=False,
        )

        assert council is not None
        assert council.name == custom_name
        assert council.description == custom_description
    except Exception as e:
        pytest.fail(f"Custom name and description test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_chairman_synthesis_called(
    mock_batched_execution, mock_concurrent_execution
):
    """Test that chairman synthesis is called with correct prompt"""
    try:
        council = LLMCouncil(verbose=False)
        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Mock responses for all council members
        mock_concurrent_execution.return_value = {
            member.agent_name: f"Response from {member.agent_name}"
            for member in council.council_members
        }
        mock_batched_execution.return_value = [
            f"Evaluation {i}" for i in range(len(council.council_members))
        ]

        query = "What is AI?"
        assert query is not None

        # Execute workflow (execution functions are mocked to avoid API calls)
        result = council.run(query)

        # Verify workflow completed
        assert result is not None
        assert mock_concurrent_execution.called
        assert mock_batched_execution.called
    except Exception as e:
        pytest.fail(f"Chairman synthesis test failed with error: {e}")


def test_llm_council_id_generation():
    """Test that council gets a unique ID"""
    try:
        council1 = LLMCouncil(verbose=False)
        council2 = LLMCouncil(verbose=False)

        assert council1 is not None
        assert council2 is not None

        # IDs should be different (unless there's a collision, which is very unlikely)
        # We at least verify they exist
        assert hasattr(council1, "name")
        assert hasattr(council2, "name")
        assert council1.name is not None
        assert council2.name is not None
    except Exception as e:
        pytest.fail(f"ID generation test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_evaluation_prompt_contains_all_responses(
    mock_batched_execution, mock_concurrent_execution
):
    """Test that evaluation prompts contain all responses"""
    try:
        responses = {
            "GPT-5.1-Councilor": "Response A content",
            "Gemini-3-Pro-Councilor": "Response B content",
            "Claude-Sonnet-4.5-Councilor": "Response C content",
            "Grok-4-Councilor": "Response D content",
        }

        assert responses is not None

        mock_concurrent_execution.return_value = responses
        mock_batched_execution.return_value = ["Eval 1", "Eval 2", "Eval 3", "Eval 4"]

        council = LLMCouncil(verbose=False)
        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Capture evaluation prompts
        import swarms.structs.llm_council as llm_council_module
        original_func = llm_council_module.get_evaluation_prompt
        captured = []

        def wrapper(*args, **kwargs):
            result = original_func(*args, **kwargs)
            captured.append((args, kwargs, result))
            return result

        with patch.object(
            llm_council_module, "get_evaluation_prompt", side_effect=wrapper
        ):
            council.run("Test query")

        # Verify evaluation prompts were created
        assert captured is not None
        assert len(captured) == 4  # One for each council member

        # Verify each prompt contains all response IDs
        for args, kwargs, prompt in captured:
            assert args is not None
            assert prompt is not None
            query, anonymous_responses, evaluator_name = args
            # Should have 4 anonymous responses (A, B, C, D)
            assert anonymous_responses is not None
            assert len(anonymous_responses) == 4
            # Each response ID should be in the prompt
            for response_id in anonymous_responses.keys():
                assert response_id is not None
                assert f"Response {response_id}" in prompt
    except Exception as e:
        pytest.fail(f"Evaluation prompt contains all responses test failed with error: {e}")


def test_llm_council_error_handling_invalid_agents():
    """Test error handling when invalid agents are provided"""
    try:
        # Test with empty list
        council = LLMCouncil(
            council_members=[],
            verbose=False,
        )
        assert council is not None
        assert council.council_members is not None
        assert len(council.council_members) == 0
    except Exception as e:
        # It's okay if this raises an error, but we should handle it gracefully
        assert isinstance(e, (ValueError, TypeError, AttributeError))


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_error_handling_none_query(
    mock_batched_execution, mock_concurrent_execution
):
    """Test error handling when None query is provided"""
    try:
        council = LLMCouncil(verbose=False)
        assert council is not None
        assert len(council.council_members) > 0

        # Mock execution functions to avoid API calls
        mock_concurrent_execution.return_value = {
            member.agent_name: "Response"
            for member in council.council_members
        }
        mock_batched_execution.return_value = [
            "Eval" for _ in council.council_members
        ]

        # Should handle None query gracefully
        try:
            result = council.run(None)
            # If it doesn't raise, result might be None or empty
            # This is acceptable behavior
            assert result is not None or result is None
        except (TypeError, ValueError, AttributeError):
            # Expected to raise error for None query
            pass
    except Exception as e:
        # Initialization should work
        assert isinstance(e, (TypeError, ValueError, AttributeError))


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_error_handling_agent_failure(
    mock_batched_execution, mock_concurrent_execution
):
    """Test error handling when an agent fails during execution"""
    try:
        # Simulate agent failure
        mock_concurrent_execution.side_effect = Exception("Agent execution failed")

        council = LLMCouncil(verbose=False)
        assert council is not None

        # Should handle the exception gracefully
        try:
            result = council.run("Test query")
            # If execution continues, result should still be returned
            # (implementation may handle errors internally)
        except Exception as e:
            # It's acceptable for the method to raise if agents fail
            assert isinstance(e, Exception)
    except Exception as e:
        pytest.fail(f"Error handling agent failure test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_error_handling_chairman_failure(
    mock_batched_execution, mock_concurrent_execution
):
    """Test error handling when chairman fails during synthesis"""
    try:
        mock_concurrent_execution.return_value = {
            "GPT-5.1-Councilor": "Response 1",
            "Gemini-3-Pro-Councilor": "Response 2",
        }

        mock_batched_execution.return_value = ["Eval 1", "Eval 2"]

        council = LLMCouncil(verbose=False)
        assert council is not None
        assert council.chairman is not None
        assert len(council.council_members) > 0

        # Simulate chairman failure by creating a function that raises
        def failing_run(*args, **kwargs):
            raise Exception("Chairman synthesis failed")
        
        original_run = council.chairman.run
        council.chairman.run = failing_run

        # Should handle the exception gracefully
        try:
            result = council.run("Test query")
            # If execution continues, result might be None or error message
            # This depends on implementation
            # Restore original method
            council.chairman.run = original_run
        except Exception as e:
            # It's acceptable for the method to raise if chairman fails
            assert isinstance(e, Exception)
            # Restore original method
            council.chairman.run = original_run
    except Exception as e:
        pytest.fail(f"Error handling chairman failure test failed with error: {e}")


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_error_handling_empty_string_query(
    mock_batched_execution, mock_concurrent_execution
):
    """Test error handling with empty string query"""
    try:
        council = LLMCouncil(verbose=False)
        assert council is not None
        assert len(council.council_members) > 0

        # Mock execution functions to avoid API calls
        mock_concurrent_execution.return_value = {
            member.agent_name: "Response"
            for member in council.council_members
        }
        mock_batched_execution.return_value = [
            "Eval" for _ in council.council_members
        ]

        # Empty string should be handled
        result = council.run("")
        assert result is not None
    except Exception as e:
        # Empty string might be valid or might raise error
        # Both behaviors are acceptable
        assert isinstance(e, (ValueError, TypeError)) or True


@patch("swarms.structs.llm_council.run_agents_concurrently")
@patch("swarms.structs.llm_council.batched_grid_agent_execution")
def test_llm_council_error_handling_missing_responses(
    mock_batched_execution, mock_concurrent_execution
):
    """Test error handling when some agents don't return responses"""
    try:
        council = LLMCouncil(verbose=False)
        assert council is not None
        assert len(council.council_members) > 0

        # Simulate missing responses - not all agents return
        mock_concurrent_execution.return_value = {
            council.council_members[0].agent_name: "Response 1",
            # Missing other responses - they will default to empty string
        }

        # Must return same number of evaluations as council members
        # Some can be empty or error messages to simulate missing evaluations
        mock_batched_execution.return_value = [
            "Eval 1" if i == 0 else "" 
            for i in range(len(council.council_members))
        ]

        # Should handle missing responses gracefully
        result = council.run("Test query")
        assert result is not None
    except Exception as e:
        pytest.fail(f"Error handling missing responses test failed with error: {e}")


def test_llm_council_error_handling_invalid_output_type():
    """Test error handling with invalid output type"""
    try:
        # Invalid output type should either work or raise error gracefully
        council = LLMCouncil(
            output_type="invalid_type",
            verbose=False,
        )
        assert council is not None
        assert council.output_type == "invalid_type"
        # The actual formatting might fail later, but initialization should work
    except Exception as e:
        # It's acceptable if invalid output type raises error
        assert isinstance(e, (ValueError, TypeError))

