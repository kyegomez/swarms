"""
Tests for Stagehand Integration with Swarms
==========================================

This module contains tests for the Stagehand browser automation
integration with the Swarms framework.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from swarms import Agent
from swarms.tools.base_tool import BaseTool


# Mock Stagehand classes
class MockObserveResult:
    def __init__(self, description, selector, method="click"):
        self.description = description
        self.selector = selector
        self.method = method


class MockStagehandPage:
    async def goto(self, url):
        return None
    
    async def act(self, action):
        return f"Performed action: {action}"
    
    async def extract(self, query):
        return {"extracted": query, "data": ["item1", "item2"]}
    
    async def observe(self, query):
        return [
            MockObserveResult("Search box", "#search-input"),
            MockObserveResult("Submit button", "#submit-btn"),
        ]


class MockStagehand:
    def __init__(self, config):
        self.config = config
        self.page = MockStagehandPage()
    
    async def init(self):
        pass
    
    async def close(self):
        pass


# Test StagehandAgent wrapper
class TestStagehandAgent:
    """Test the StagehandAgent wrapper class."""
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_agent_initialization(self):
        """Test that StagehandAgent initializes correctly."""
        from examples.stagehand.stagehand_wrapper_agent import StagehandAgent
        
        agent = StagehandAgent(
            agent_name="TestAgent",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        
        assert agent.agent_name == "TestAgent"
        assert agent.stagehand_config.env == "LOCAL"
        assert agent.stagehand_config.model_name == "gpt-4o-mini"
        assert not agent._initialized
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_navigation_task(self):
        """Test navigation and extraction task."""
        from examples.stagehand.stagehand_wrapper_agent import StagehandAgent
        
        agent = StagehandAgent(
            agent_name="TestAgent",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        
        result = agent.run("Navigate to example.com and extract the main content")
        
        # Parse result
        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert "navigated_to" in result_data["data"]
        assert result_data["data"]["navigated_to"] == "https://example.com"
        assert "extracted" in result_data["data"]
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_search_task(self):
        """Test search functionality."""
        from examples.stagehand.stagehand_wrapper_agent import StagehandAgent
        
        agent = StagehandAgent(
            agent_name="TestAgent",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        
        result = agent.run("Go to google.com and search for 'test query'")
        
        result_data = json.loads(result)
        assert result_data["status"] == "completed"
        assert result_data["data"]["search_query"] == "test query"
        assert result_data["action"] == "search"
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_cleanup(self):
        """Test that cleanup properly closes browser."""
        from examples.stagehand.stagehand_wrapper_agent import StagehandAgent
        
        agent = StagehandAgent(
            agent_name="TestAgent",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        
        # Initialize the agent
        agent.run("Navigate to example.com")
        assert agent._initialized
        
        # Cleanup
        agent.cleanup()
        
        # After cleanup, should be able to run again
        result = agent.run("Navigate to example.com")
        assert result is not None


# Test Stagehand Tools
class TestStagehandTools:
    """Test individual Stagehand tools."""
    
    @patch('examples.stagehand.stagehand_tools_agent.browser_state')
    async def test_navigate_tool(self, mock_browser_state):
        """Test NavigateTool functionality."""
        from examples.stagehand.stagehand_tools_agent import NavigateTool
        
        # Setup mock
        mock_page = AsyncMock()
        mock_browser_state.get_page = AsyncMock(return_value=mock_page)
        mock_browser_state.init_browser = AsyncMock()
        
        tool = NavigateTool()
        result = await tool._async_run("https://example.com")
        
        assert "Successfully navigated to https://example.com" in result
        mock_page.goto.assert_called_once_with("https://example.com")
    
    @patch('examples.stagehand.stagehand_tools_agent.browser_state')
    async def test_act_tool(self, mock_browser_state):
        """Test ActTool functionality."""
        from examples.stagehand.stagehand_tools_agent import ActTool
        
        # Setup mock
        mock_page = AsyncMock()
        mock_page.act = AsyncMock(return_value="Action completed")
        mock_browser_state.get_page = AsyncMock(return_value=mock_page)
        mock_browser_state.init_browser = AsyncMock()
        
        tool = ActTool()
        result = await tool._async_run("click the button")
        
        assert "Action performed" in result
        assert "click the button" in result
        mock_page.act.assert_called_once_with("click the button")
    
    @patch('examples.stagehand.stagehand_tools_agent.browser_state')
    async def test_extract_tool(self, mock_browser_state):
        """Test ExtractTool functionality."""
        from examples.stagehand.stagehand_tools_agent import ExtractTool
        
        # Setup mock
        mock_page = AsyncMock()
        mock_page.extract = AsyncMock(return_value={"title": "Test Page", "content": "Test content"})
        mock_browser_state.get_page = AsyncMock(return_value=mock_page)
        mock_browser_state.init_browser = AsyncMock()
        
        tool = ExtractTool()
        result = await tool._async_run("extract the page title")
        
        # Result should be JSON string
        parsed_result = json.loads(result)
        assert parsed_result["title"] == "Test Page"
        assert parsed_result["content"] == "Test content"
    
    @patch('examples.stagehand.stagehand_tools_agent.browser_state')
    async def test_observe_tool(self, mock_browser_state):
        """Test ObserveTool functionality."""
        from examples.stagehand.stagehand_tools_agent import ObserveTool
        
        # Setup mock
        mock_page = AsyncMock()
        mock_observations = [
            MockObserveResult("Search input", "#search"),
            MockObserveResult("Submit button", "#submit"),
        ]
        mock_page.observe = AsyncMock(return_value=mock_observations)
        mock_browser_state.get_page = AsyncMock(return_value=mock_page)
        mock_browser_state.init_browser = AsyncMock()
        
        tool = ObserveTool()
        result = await tool._async_run("find the search box")
        
        # Result should be JSON string
        parsed_result = json.loads(result)
        assert len(parsed_result) == 2
        assert parsed_result[0]["description"] == "Search input"
        assert parsed_result[0]["selector"] == "#search"


# Test MCP integration
class TestStagehandMCP:
    """Test Stagehand MCP server integration."""
    
    def test_mcp_agent_initialization(self):
        """Test that MCP agent initializes with correct parameters."""
        from examples.stagehand.stagehand_mcp_agent import StagehandMCPAgent
        
        mcp_agent = StagehandMCPAgent(
            agent_name="TestMCPAgent",
            mcp_server_url="http://localhost:3000/sse",
            model_name="gpt-4o-mini",
        )
        
        assert mcp_agent.agent.agent_name == "TestMCPAgent"
        assert mcp_agent.agent.mcp_url == "http://localhost:3000/sse"
        assert mcp_agent.agent.model_name == "gpt-4o-mini"
    
    def test_multi_session_swarm_creation(self):
        """Test multi-session browser swarm creation."""
        from examples.stagehand.stagehand_mcp_agent import MultiSessionBrowserSwarm
        
        swarm = MultiSessionBrowserSwarm(
            mcp_server_url="http://localhost:3000/sse",
            num_agents=3,
        )
        
        assert len(swarm.agents) == 3
        assert swarm.agents[0].agent_name == "DataExtractor_0"
        assert swarm.agents[1].agent_name == "FormFiller_1"
        assert swarm.agents[2].agent_name == "WebMonitor_2"
    
    @patch('swarms.Agent.run')
    def test_task_distribution(self, mock_run):
        """Test task distribution among swarm agents."""
        from examples.stagehand.stagehand_mcp_agent import MultiSessionBrowserSwarm
        
        mock_run.return_value = "Task completed"
        
        swarm = MultiSessionBrowserSwarm(num_agents=2)
        tasks = ["Task 1", "Task 2", "Task 3"]
        
        results = swarm.distribute_tasks(tasks)
        
        assert len(results) == 3
        assert all(result == "Task completed" for result in results)
        assert mock_run.call_count == 3


# Test multi-agent workflows
class TestMultiAgentWorkflows:
    """Test multi-agent workflow configurations."""
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_price_comparison_workflow_creation(self):
        """Test creation of price comparison workflow."""
        from examples.stagehand.stagehand_multi_agent_workflow import create_price_comparison_workflow
        
        workflow = create_price_comparison_workflow()
        
        # Should be a SequentialWorkflow with 2 agents
        assert len(workflow.agents) == 2
        # First agent should be a ConcurrentWorkflow
        assert hasattr(workflow.agents[0], 'agents')
        # Second agent should be the analysis agent
        assert workflow.agents[1].agent_name == "PriceAnalysisAgent"
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_competitive_analysis_workflow_creation(self):
        """Test creation of competitive analysis workflow."""
        from examples.stagehand.stagehand_multi_agent_workflow import create_competitive_analysis_workflow
        
        workflow = create_competitive_analysis_workflow()
        
        # Should have 3 agents in the rearrange pattern
        assert len(workflow.agents) == 3
        assert workflow.flow == "company_researcher -> social_media_agent -> report_compiler"
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_automated_testing_workflow_creation(self):
        """Test creation of automated testing workflow."""
        from examples.stagehand.stagehand_multi_agent_workflow import create_automated_testing_workflow
        
        workflow = create_automated_testing_workflow()
        
        # Should be a SequentialWorkflow
        assert len(workflow.agents) == 2
        # First should be concurrent testing
        assert hasattr(workflow.agents[0], 'agents')
        assert len(workflow.agents[0].agents) == 3  # UI, Form, Accessibility testers
    
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    def test_news_aggregation_workflow_creation(self):
        """Test creation of news aggregation workflow."""
        from examples.stagehand.stagehand_multi_agent_workflow import create_news_aggregation_workflow
        
        workflow = create_news_aggregation_workflow()
        
        # Should be a SequentialWorkflow with 3 stages
        assert len(workflow.agents) == 3
        # First stage should be concurrent scrapers
        assert hasattr(workflow.agents[0], 'agents')
        assert len(workflow.agents[0].agents) == 3  # 3 news sources


# Integration tests
class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    @patch('examples.stagehand.stagehand_wrapper_agent.Stagehand', MockStagehand)
    async def test_full_browser_automation_flow(self):
        """Test a complete browser automation flow."""
        from examples.stagehand.stagehand_wrapper_agent import StagehandAgent
        
        agent = StagehandAgent(
            agent_name="IntegrationTestAgent",
            model_name="gpt-4o-mini",
            env="LOCAL",
        )
        
        # Test navigation
        nav_result = agent.run("Navigate to example.com")
        assert "navigated_to" in nav_result
        
        # Test extraction
        extract_result = agent.run("Extract all text from the page")
        assert "extracted" in extract_result
        
        # Test observation
        observe_result = agent.run("Find all buttons on the page")
        assert "observation" in observe_result
        
        # Cleanup
        agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])