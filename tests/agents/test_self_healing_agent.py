import pytest
from unittest.mock import Mock, patch
from swarms.agents import SelfHealingAgent

# Test data
SAMPLE_ERROR_RESPONSE = {
    "error_type": "ZeroDivisionError",
    "analysis": "Division by zero error occurred",
    "context": "Error in calculate_ratio() function",
    "solution": "Add zero check before division",
    "confidence": 0.95,
    "metadata": {
        "file": "test.py",
        "line": 10,
        "function": "calculate_ratio"
    }
}

@pytest.fixture
def agent():
    """Create a SelfHealingAgent instance for testing"""
    return SelfHealingAgent(
        model_name="gpt-4o-mini",
        max_retries=3,
        verbose=False
    )

def test_agent_initialization():
    """Test agent initialization with different parameters"""
    agent = SelfHealingAgent(
        model_name="gpt-4o-mini",
        max_retries=5,
        verbose=True
    )
    
    assert agent.model_name == "gpt-4o-mini"
    assert agent.max_retries == 5
    assert agent.verbose is True

def test_analyze_and_fix_zero_division():
    """Test error analysis for zero division error"""
    agent = SelfHealingAgent()
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        
        assert isinstance(fix, dict)
        assert "error_type" in fix
        assert "analysis" in fix
        assert "solution" in fix
        assert "confidence" in fix
        assert fix["error_type"] == "ZeroDivisionError"
        assert isinstance(fix["confidence"], float)
        assert 0 <= fix["confidence"] <= 1

def test_analyze_and_fix_key_error():
    """Test error analysis for key error"""
    agent = SelfHealingAgent()
    
    try:
        empty_dict = {}
        _ = empty_dict["nonexistent_key"]
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        
        assert isinstance(fix, dict)
        assert fix["error_type"] == "KeyError"
        assert "solution" in fix
        assert isinstance(fix["confidence"], float)

def test_analyze_and_fix_type_error():
    """Test error analysis for type error"""
    agent = SelfHealingAgent()
    
    try:
        _ = "string" + 123
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        
        assert isinstance(fix, dict)
        assert fix["error_type"] == "TypeError"
        assert "solution" in fix
        assert isinstance(fix["confidence"], float)

@patch('swarms.agents.self_healing_agent.SelfHealingAgent.analyze_and_fix')
def test_apply_fix_success(mock_analyze):
    """Test successful fix application"""
    agent = SelfHealingAgent()
    mock_analyze.return_value = SAMPLE_ERROR_RESPONSE
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        result = agent.apply_fix(fix)
        
        assert result is True
        assert fix["confidence"] > 0.8
        assert "solution" in fix

@patch('swarms.agents.self_healing_agent.SelfHealingAgent.analyze_and_fix')
def test_apply_fix_low_confidence(mock_analyze):
    """Test fix application with low confidence"""
    agent = SelfHealingAgent()
    low_confidence_response = SAMPLE_ERROR_RESPONSE.copy()
    low_confidence_response["confidence"] = 0.3
    mock_analyze.return_value = low_confidence_response
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        result = agent.apply_fix(fix)
        
        assert result is False
        assert fix["confidence"] < 0.8

def test_max_retries_limit():
    """Test that agent respects max_retries limit"""
    agent = SelfHealingAgent(max_retries=2)
    retry_count = 0
    
    try:
        while True:
            1/0
            retry_count += 1
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        
        assert retry_count <= agent.max_retries
        assert isinstance(fix, dict)

def test_context_handling():
    """Test error analysis with additional context"""
    agent = SelfHealingAgent()
    context = {
        "function": "test_function",
        "input_data": "test_input",
        "expected_output": "test_output"
    }
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e, context=context)
        
        assert isinstance(fix, dict)
        assert "context" in fix
        assert fix["context"] is not None

def test_batch_error_handling():
    """Test handling multiple errors in batch"""
    agent = SelfHealingAgent()
    errors = []
    
    test_operations = [
        lambda: 1/0,
        lambda: {}["nonexistent"],
        lambda: "string" + 123
    ]
    
    for op in test_operations:
        try:
            op()
        except Exception as e:
            fix = agent.analyze_and_fix(e)
            errors.append(fix)
    
    assert len(errors) == 3
    assert all(isinstance(fix, dict) for fix in errors)
    assert all("error_type" in fix for fix in errors)
    assert all("solution" in fix for fix in errors)

def test_error_metadata():
    """Test error metadata collection"""
    agent = SelfHealingAgent()
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        
        assert "metadata" in fix
        assert isinstance(fix["metadata"], dict)
        assert "file" in fix["metadata"]
        assert "line" in fix["metadata"]

def test_verbose_mode():
    """Test agent's verbose mode"""
    agent = SelfHealingAgent(verbose=True)
    
    try:
        1/0
    except Exception as e:
        with patch('builtins.print') as mock_print:
            fix = agent.analyze_and_fix(e)
            assert mock_print.called

def test_custom_system_prompt():
    """Test agent with custom system prompt"""
    custom_prompt = "You are an expert Python developer specializing in error fixing."
    agent = SelfHealingAgent(system_prompt=custom_prompt)
    
    try:
        1/0
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        assert isinstance(fix, dict)
        assert "solution" in fix

if __name__ == "__main__":
    pytest.main([__file__]) 