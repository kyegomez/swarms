
import pytest
import asyncio
from mock_multi_agent import MultiAgentMathSystem
import logging

logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_multi_agent_math():
    """Test the multi-agent math system with various operations"""
    system = MultiAgentMathSystem()
    
    test_cases = [
        "Add 5 and 3",
        "Multiply 4 by 6",
        "Divide 10 by 2"
    ]
    
    for task in test_cases:
        print(f"\nTesting: {task}")
        results = await system.process_task(task)
        
        for result in results:
            assert "error" not in result, f"Agent {result['agent']} encountered error"
            assert result["response"] is not None
            print(f"{result['agent']} response: {result['response']}")

def test_interactive_system():
    """Test the interactive system manually"""
    try:
        system = MultiAgentMathSystem()
        system.run_interactive()
    except Exception as e:
        pytest.fail(f"Interactive test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
