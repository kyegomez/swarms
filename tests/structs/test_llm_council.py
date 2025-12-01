"""
Test file for LLM Council functionality.

Tests core functionalities of the LLM Council including:
- Initialization (default and custom)
- Running queries
- Batch processing
- Output formatting
"""

import pytest
from loguru import logger
from dotenv import load_dotenv
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.agent import Agent

load_dotenv()


def test_llm_council_default_initialization():
    """Test LLM Council initialization with default council members."""
    try:
        logger.info("Testing LLM Council default initialization...")
        
        council = LLMCouncil(
            verbose=False,
            output_type="dict-all-except-first"
        )
        
        assert council is not None, "Council should be initialized"
        assert council.name == "LLM Council", "Default name should be 'LLM Council'"
        assert len(council.council_members) > 0, "Should have council members"
        assert council.chairman is not None, "Chairman should be initialized"
        assert council.conversation is not None, "Conversation should be initialized"
        
        logger.info(f"✓ Council initialized with {len(council.council_members)} members")
        logger.info("✓ Default initialization test passed")
        
    except Exception as e:
        logger.error(f"✗ Default initialization test failed: {e}")
        raise


def test_llm_council_custom_initialization():
    """Test LLM Council initialization with custom council members."""
    try:
        logger.info("Testing LLM Council custom initialization...")
        
        # Create custom council members with simpler models
        custom_members = [
            Agent(
                agent_name="TestAgent1",
                agent_description="First test agent",
                system_prompt="You are a helpful test agent.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
            Agent(
                agent_name="TestAgent2",
                agent_description="Second test agent",
                system_prompt="You are a helpful test agent.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
        ]
        
        council = LLMCouncil(
            name="Custom Council",
            council_members=custom_members,
            chairman_model="gpt-4o-mini",
            verbose=False,
            output_type="string"
        )
        
        assert council is not None, "Council should be initialized"
        assert council.name == "Custom Council", "Name should match custom value"
        assert len(council.council_members) == 2, "Should have 2 custom members"
        assert council.council_members[0].agent_name == "TestAgent1", "First member should match"
        assert council.council_members[1].agent_name == "TestAgent2", "Second member should match"
        assert council.output_type == "string", "Output type should be 'string'"
        
        logger.info("✓ Custom initialization test passed")
        
    except Exception as e:
        logger.error(f"✗ Custom initialization test failed: {e}")
        raise


def test_llm_council_run():
    """Test LLM Council run method with a simple query."""
    try:
        logger.info("Testing LLM Council run method...")
        
        # Use simpler models for testing
        custom_members = [
            Agent(
                agent_name="TestAgent1",
                agent_description="First test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
            Agent(
                agent_name="TestAgent2",
                agent_description="Second test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
        ]
        
        council = LLMCouncil(
            council_members=custom_members,
            chairman_model="gpt-4o-mini",
            verbose=False,
            output_type="dict-all-except-first"
        )
        
        query = "What is 2 + 2? Provide a brief answer."
        result = council.run(query)
        
        # Basic assertions
        assert result is not None, "Result should not be None"
        assert council.conversation is not None, "Conversation should exist"
        assert len(council.conversation.conversation_history) > 0, "Conversation should have messages"
        
        # Enhanced assertions to verify workflow steps
        messages = council.conversation.conversation_history
        
        # Step 1: Verify User query was added
        user_messages = [msg for msg in messages if msg.get("role") == "User"]
        assert len(user_messages) > 0, "User query should be in conversation"
        
        # Step 2: Verify all council members responded
        member_responses = [msg for msg in messages if msg.get("role") in ["TestAgent1", "TestAgent2"]]
        assert len(member_responses) == len(custom_members), f"All {len(custom_members)} council members should have responded"
        
        # Step 3: Verify evaluations were performed
        evaluation_messages = [msg for msg in messages if "-Evaluation" in msg.get("role", "")]
        assert len(evaluation_messages) == len(custom_members), f"All {len(custom_members)} members should have evaluated"
        
        # Step 4: Verify Chairman synthesis occurred
        chairman_messages = [msg for msg in messages if msg.get("role") == "Chairman"]
        assert len(chairman_messages) > 0, "Chairman should have synthesized final response"
        
        logger.info("✓ Run method test passed")
        logger.info(f"✓ Verified {len(member_responses)} member responses, {len(evaluation_messages)} evaluations, and {len(chairman_messages)} chairman synthesis")
        
    except Exception as e:
        logger.error(f"✗ Run method test failed: {e}")
        raise


def test_llm_council_batched_run():
    """Test LLM Council batched_run method with multiple tasks."""
    try:
        logger.info("Testing LLM Council batched_run method...")
        
        # Use simpler models for testing
        custom_members = [
            Agent(
                agent_name="TestAgent1",
                agent_description="First test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
            Agent(
                agent_name="TestAgent2",
                agent_description="Second test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
        ]
        
        council = LLMCouncil(
            council_members=custom_members,
            chairman_model="gpt-4o-mini",
            verbose=False,
            output_type="dict-all-except-first"
        )
        
        tasks = [
            "What is 1 + 1?",
            "What is 3 + 3?",
        ]
        
        results = council.batched_run(tasks)
        
        assert results is not None, "Results should not be None"
        assert len(results) == len(tasks), f"Should have {len(tasks)} results"
        assert all(result is not None for result in results), "All results should not be None"
        
        logger.info(f"✓ Batched run test passed with {len(results)} results")
        
    except Exception as e:
        logger.error(f"✗ Batched run test failed: {e}")
        raise


def test_llm_council_output_types():
    """Test LLM Council with different output types."""
    try:
        logger.info("Testing LLM Council with different output types...")
        
        # Use simpler models for testing
        custom_members = [
            Agent(
                agent_name="TestAgent1",
                agent_description="First test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
            Agent(
                agent_name="TestAgent2",
                agent_description="Second test agent",
                system_prompt="You are a helpful test agent. Provide concise answers.",
                model_name="gpt-4o-mini",
                max_loops=1,
                verbose=False,
            ),
        ]
        
        output_types = ["string", "dict-all-except-first", "final"]
        
        for output_type in output_types:
            logger.info(f"Testing output type: {output_type}")
            
            council = LLMCouncil(
                council_members=custom_members,
                chairman_model="gpt-4o-mini",
                verbose=False,
                output_type=output_type
            )
            
            query = "What is 5 + 5? Provide a brief answer."
            result = council.run(query)
            
            assert result is not None, f"Result should not be None for output type {output_type}"
            assert council.output_type == output_type, f"Output type should be {output_type}"
            
            logger.info(f"✓ Output type '{output_type}' test passed")
        
        logger.info("✓ All output types test passed")
        
    except Exception as e:
        logger.error(f"✗ Output types test failed: {e}")
        raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

