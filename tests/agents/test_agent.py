import asyncio
import pytest
from swarms.agent import Agent

@pytest.mark.asyncio
async def test_streaming_callback_during_summary():
    """Test that streaming callback is invoked during summary phase"""
    tokens = []
    
    def on_token(token):
        tokens.append(token)
    
    agent = Agent(
        name="test_agent",
        max_loops="auto",
        streaming_on=True,
        on_token=on_token
    )
    
    await agent.autonomous_loop()
    
    # Verify that tokens were emitted during summary generation
    assert len(tokens) > 0, "No tokens were emitted during summary phase"
    assert "Summary:" in "".join(tokens), "Summary content not found in tokens"
    assert agent.get_summary() in "".join(tokens), "Full summary not found in tokens"

@pytest.mark.asyncio
async def test_streaming_callback_not_called_when_disabled():
    """Test that streaming callback is not invoked when streaming is disabled"""
    tokens = []
    
    def on_token(token):
        tokens.append(token)
    
    agent = Agent(
        name="test_agent",
        max_loops=1,
        streaming_on=False,
        on_token=on_token
    )
    
    await agent.autonomous_loop()
    
    # Verify that no tokens were emitted
    assert len(tokens) == 0, "Tokens were emitted when streaming was disabled"
```