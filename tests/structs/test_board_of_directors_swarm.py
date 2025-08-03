"""
Comprehensive test suite for Board of Directors Swarm.

This module contains extensive tests for the Board of Directors swarm implementation,
covering all aspects including initialization, board operations, task execution,
error handling, and performance characteristics.

The test suite follows the Swarms testing philosophy:
- Comprehensive coverage of all functionality
- Proper mocking and isolation
- Performance and integration testing
- Error handling validation
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import List, Dict, Any, Optional

from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
    BoardMember,
    BoardMemberRole,
    BoardDecisionType,
    BoardOrder,
    BoardDecision,
    BoardSpec,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation


# Test fixtures
@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.agent_name = "TestAgent"
    agent.agent_description = "A test agent for unit testing"
    agent.run = Mock(return_value="Test agent response")
    agent.arun = AsyncMock(return_value="Async test agent response")
    return agent


@pytest.fixture
def mock_board_member(mock_agent):
    """Create a mock board member for testing."""
    return BoardMember(
        agent=mock_agent,
        role=BoardMemberRole.CHAIRMAN,
        voting_weight=1.5,
        expertise_areas=["leadership", "strategy"]
    )


@pytest.fixture
def sample_agents():
    """Create sample agents for testing."""
    agents = []
    for i in range(3):
        agent = Mock(spec=Agent)
        agent.agent_name = f"Agent{i+1}"
        agent.agent_description = f"Test agent {i+1}"
        agent.run = Mock(return_value=f"Response from Agent{i+1}")
        agents.append(agent)
    return agents


@pytest.fixture
def sample_board_members(sample_agents):
    """Create sample board members for testing."""
    roles = [BoardMemberRole.CHAIRMAN, BoardMemberRole.VICE_CHAIRMAN, BoardMemberRole.SECRETARY]
    board_members = []
    
    for i, (agent, role) in enumerate(zip(sample_agents, roles)):
        board_member = BoardMember(
            agent=agent,
            role=role,
            voting_weight=1.0 + (i * 0.2),
            expertise_areas=[f"expertise_{i+1}"]
        )
        board_members.append(board_member)
    
    return board_members


@pytest.fixture
def basic_board_swarm(sample_agents):
    """Create a basic Board of Directors swarm for testing."""
    return BoardOfDirectorsSwarm(
        name="TestBoard",
        agents=sample_agents,
        verbose=False,
        max_loops=1
    )


@pytest.fixture
def configured_board_swarm(sample_agents, sample_board_members):
    """Create a configured Board of Directors swarm for testing."""
    return BoardOfDirectorsSwarm(
        name="ConfiguredBoard",
        description="A configured board for testing",
        board_members=sample_board_members,
        agents=sample_agents,
        max_loops=2,
        verbose=True,
        decision_threshold=0.7,
        enable_voting=True,
        enable_consensus=True,
        max_workers=4
    )


# Unit tests for enums and data models
class TestBoardMemberRole:
    """Test BoardMemberRole enum."""
    
    def test_enum_values(self):
        """Test that all enum values are correctly defined."""
        assert BoardMemberRole.CHAIRMAN == "chairman"
        assert BoardMemberRole.VICE_CHAIRMAN == "vice_chairman"
        assert BoardMemberRole.SECRETARY == "secretary"
        assert BoardMemberRole.TREASURER == "treasurer"
        assert BoardMemberRole.MEMBER == "member"
        assert BoardMemberRole.EXECUTIVE_DIRECTOR == "executive_director"


class TestBoardDecisionType:
    """Test BoardDecisionType enum."""
    
    def test_enum_values(self):
        """Test that all enum values are correctly defined."""
        assert BoardDecisionType.UNANIMOUS == "unanimous"
        assert BoardDecisionType.MAJORITY == "majority"
        assert BoardDecisionType.CONSENSUS == "consensus"
        assert BoardDecisionType.CHAIRMAN_DECISION == "chairman_decision"


class TestBoardMember:
    """Test BoardMember dataclass."""
    
    def test_board_member_creation(self, mock_agent):
        """Test creating a board member."""
        board_member = BoardMember(
            agent=mock_agent,
            role=BoardMemberRole.CHAIRMAN,
            voting_weight=1.5,
            expertise_areas=["leadership", "strategy"]
        )
        
        assert board_member.agent == mock_agent
        assert board_member.role == BoardMemberRole.CHAIRMAN
        assert board_member.voting_weight == 1.5
        assert board_member.expertise_areas == ["leadership", "strategy"]
    
    def test_board_member_defaults(self, mock_agent):
        """Test board member with default values."""
        board_member = BoardMember(
            agent=mock_agent,
            role=BoardMemberRole.MEMBER
        )
        
        assert board_member.voting_weight == 1.0
        assert board_member.expertise_areas == []
    
    def test_board_member_post_init(self, mock_agent):
        """Test board member post-init with None expertise areas."""
        board_member = BoardMember(
            agent=mock_agent,
            role=BoardMemberRole.MEMBER,
            expertise_areas=None
        )
        
        assert board_member.expertise_areas == []


class TestBoardOrder:
    """Test BoardOrder model."""
    
    def test_board_order_creation(self):
        """Test creating a board order."""
        order = BoardOrder(
            agent_name="TestAgent",
            task="Test task",
            priority=1,
            deadline="2024-01-01",
            assigned_by="Chairman"
        )
        
        assert order.agent_name == "TestAgent"
        assert order.task == "Test task"
        assert order.priority == 1
        assert order.deadline == "2024-01-01"
        assert order.assigned_by == "Chairman"
    
    def test_board_order_defaults(self):
        """Test board order with default values."""
        order = BoardOrder(
            agent_name="TestAgent",
            task="Test task"
        )
        
        assert order.priority == 3
        assert order.deadline is None
        assert order.assigned_by == "Board of Directors"
    
    def test_board_order_validation(self):
        """Test board order validation."""
        # Test priority validation
        with pytest.raises(ValueError):
            BoardOrder(
                agent_name="TestAgent",
                task="Test task",
                priority=0  # Invalid priority
            )
        
        with pytest.raises(ValueError):
            BoardOrder(
                agent_name="TestAgent",
                task="Test task",
                priority=6  # Invalid priority
            )


class TestBoardDecision:
    """Test BoardDecision model."""
    
    def test_board_decision_creation(self):
        """Test creating a board decision."""
        decision = BoardDecision(
            decision_type=BoardDecisionType.MAJORITY,
            decision="Approve the proposal",
            votes_for=3,
            votes_against=1,
            abstentions=0,
            reasoning="The proposal aligns with our strategic goals"
        )
        
        assert decision.decision_type == BoardDecisionType.MAJORITY
        assert decision.decision == "Approve the proposal"
        assert decision.votes_for == 3
        assert decision.votes_against == 1
        assert decision.abstentions == 0
        assert decision.reasoning == "The proposal aligns with our strategic goals"
    
    def test_board_decision_defaults(self):
        """Test board decision with default values."""
        decision = BoardDecision(
            decision_type=BoardDecisionType.CONSENSUS,
            decision="Test decision"
        )
        
        assert decision.votes_for == 0
        assert decision.votes_against == 0
        assert decision.abstentions == 0
        assert decision.reasoning == ""


class TestBoardSpec:
    """Test BoardSpec model."""
    
    def test_board_spec_creation(self):
        """Test creating a board spec."""
        orders = [
            BoardOrder(agent_name="Agent1", task="Task 1"),
            BoardOrder(agent_name="Agent2", task="Task 2")
        ]
        decisions = [
            BoardDecision(
                decision_type=BoardDecisionType.MAJORITY,
                decision="Decision 1"
            )
        ]
        
        spec = BoardSpec(
            plan="Test plan",
            orders=orders,
            decisions=decisions,
            meeting_summary="Test meeting summary"
        )
        
        assert spec.plan == "Test plan"
        assert len(spec.orders) == 2
        assert len(spec.decisions) == 1
        assert spec.meeting_summary == "Test meeting summary"
    
    def test_board_spec_defaults(self):
        """Test board spec with default values."""
        spec = BoardSpec(
            plan="Test plan",
            orders=[]
        )
        
        assert spec.decisions == []
        assert spec.meeting_summary == ""


# Unit tests for BoardOfDirectorsSwarm
class TestBoardOfDirectorsSwarmInitialization:
    """Test BoardOfDirectorsSwarm initialization."""
    
    def test_basic_initialization(self, sample_agents):
        """Test basic swarm initialization."""
        swarm = BoardOfDirectorsSwarm(
            name="TestSwarm",
            agents=sample_agents
        )
        
        assert swarm.name == "TestSwarm"
        assert len(swarm.agents) == 3
        assert swarm.max_loops == 1
        assert swarm.verbose is False
        assert swarm.decision_threshold == 0.6
    
    def test_configured_initialization(self, sample_agents, sample_board_members):
        """Test configured swarm initialization."""
        swarm = BoardOfDirectorsSwarm(
            name="ConfiguredSwarm",
            description="Test description",
            board_members=sample_board_members,
            agents=sample_agents,
            max_loops=3,
            verbose=True,
            decision_threshold=0.8,
            enable_voting=False,
            enable_consensus=False,
            max_workers=8
        )
        
        assert swarm.name == "ConfiguredSwarm"
        assert swarm.description == "Test description"
        assert len(swarm.board_members) == 3
        assert len(swarm.agents) == 3
        assert swarm.max_loops == 3
        assert swarm.verbose is True
        assert swarm.decision_threshold == 0.8
        assert swarm.enable_voting is False
        assert swarm.enable_consensus is False
        assert swarm.max_workers == 8
    
    def test_default_board_setup(self, sample_agents):
        """Test default board setup when no board members provided."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        
        assert len(swarm.board_members) == 3
        assert swarm.board_members[0].role == BoardMemberRole.CHAIRMAN
        assert swarm.board_members[1].role == BoardMemberRole.VICE_CHAIRMAN
        assert swarm.board_members[2].role == BoardMemberRole.SECRETARY
    
    def test_initialization_without_agents(self):
        """Test initialization without agents should raise error."""
        with pytest.raises(ValueError, match="No agents found in the swarm"):
            BoardOfDirectorsSwarm(agents=[])
    
    def test_initialization_with_invalid_max_loops(self, sample_agents):
        """Test initialization with invalid max_loops."""
        with pytest.raises(ValueError, match="Max loops must be greater than 0"):
            BoardOfDirectorsSwarm(agents=sample_agents, max_loops=0)
    
    def test_initialization_with_invalid_decision_threshold(self, sample_agents):
        """Test initialization with invalid decision threshold."""
        with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
            BoardOfDirectorsSwarm(agents=sample_agents, decision_threshold=1.5)


class TestBoardOfDirectorsSwarmMethods:
    """Test BoardOfDirectorsSwarm methods."""
    
    def test_setup_default_board(self, sample_agents):
        """Test default board setup."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        
        assert len(swarm.board_members) == 3
        assert all(hasattr(member.agent, 'agent_name') for member in swarm.board_members)
        assert all(hasattr(member.agent, 'run') for member in swarm.board_members)
    
    def test_get_chairman_prompt(self, sample_agents):
        """Test chairman prompt generation."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        prompt = swarm._get_chairman_prompt()
        
        assert "Chairman" in prompt
        assert "board meetings" in prompt
        assert "consensus" in prompt
    
    def test_get_vice_chairman_prompt(self, sample_agents):
        """Test vice chairman prompt generation."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        prompt = swarm._get_vice_chairman_prompt()
        
        assert "Vice Chairman" in prompt
        assert "supporting" in prompt
        assert "operational" in prompt
    
    def test_get_secretary_prompt(self, sample_agents):
        """Test secretary prompt generation."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        prompt = swarm._get_secretary_prompt()
        
        assert "Secretary" in prompt
        assert "documenting" in prompt
        assert "records" in prompt
    
    def test_format_board_members_info(self, configured_board_swarm):
        """Test board members info formatting."""
        info = configured_board_swarm._format_board_members_info()
        
        assert "Chairman" in info
        assert "Vice-Chairman" in info
        assert "Secretary" in info
        assert "expertise" in info
    
    def test_add_board_member(self, basic_board_swarm, mock_board_member):
        """Test adding a board member."""
        initial_count = len(basic_board_swarm.board_members)
        basic_board_swarm.add_board_member(mock_board_member)
        
        assert len(basic_board_swarm.board_members) == initial_count + 1
        assert mock_board_member in basic_board_swarm.board_members
    
    def test_remove_board_member(self, configured_board_swarm):
        """Test removing a board member."""
        member_to_remove = configured_board_swarm.board_members[0]
        member_name = member_to_remove.agent.agent_name
        
        initial_count = len(configured_board_swarm.board_members)
        configured_board_swarm.remove_board_member(member_name)
        
        assert len(configured_board_swarm.board_members) == initial_count - 1
        assert member_to_remove not in configured_board_swarm.board_members
    
    def test_get_board_member(self, configured_board_swarm):
        """Test getting a board member by name."""
        member = configured_board_swarm.board_members[0]
        member_name = member.agent.agent_name
        
        found_member = configured_board_swarm.get_board_member(member_name)
        assert found_member == member
        
        # Test with non-existent member
        not_found = configured_board_swarm.get_board_member("NonExistent")
        assert not_found is None
    
    def test_get_board_summary(self, configured_board_swarm):
        """Test getting board summary."""
        summary = configured_board_swarm.get_board_summary()
        
        assert "board_name" in summary
        assert "total_members" in summary
        assert "total_agents" in summary
        assert "max_loops" in summary
        assert "decision_threshold" in summary
        assert "members" in summary
        
        assert summary["board_name"] == "ConfiguredBoard"
        assert summary["total_members"] == 3
        assert summary["total_agents"] == 3


class TestBoardMeetingOperations:
    """Test board meeting operations."""
    
    def test_create_board_meeting_prompt(self, configured_board_swarm):
        """Test board meeting prompt creation."""
        task = "Test task for board meeting"
        prompt = configured_board_swarm._create_board_meeting_prompt(task)
        
        assert task in prompt
        assert "BOARD OF DIRECTORS MEETING" in prompt
        assert "INSTRUCTIONS" in prompt
        assert "plan" in prompt
        assert "orders" in prompt
    
    def test_conduct_board_discussion(self, configured_board_swarm):
        """Test board discussion conduction."""
        prompt = "Test board meeting prompt"
        
        with patch.object(configured_board_swarm.board_members[0].agent, 'run') as mock_run:
            mock_run.return_value = "Board discussion result"
            result = configured_board_swarm._conduct_board_discussion(prompt)
            
            assert result == "Board discussion result"
            mock_run.assert_called_once_with(task=prompt, img=None)
    
    def test_conduct_board_discussion_no_chairman(self, sample_agents):
        """Test board discussion when no chairman is found."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        # Remove all board members
        swarm.board_members = []
        
        with pytest.raises(ValueError, match="No chairman found in board members"):
            swarm._conduct_board_discussion("Test prompt")
    
    def test_parse_board_decisions_valid_json(self, configured_board_swarm):
        """Test parsing valid JSON board decisions."""
        valid_json = """
        {
            "plan": "Test plan",
            "orders": [
                {
                    "agent_name": "Agent1",
                    "task": "Task 1",
                    "priority": 1,
                    "assigned_by": "Chairman"
                }
            ],
            "decisions": [
                {
                    "decision_type": "majority",
                    "decision": "Test decision",
                    "votes_for": 2,
                    "votes_against": 1,
                    "abstentions": 0,
                    "reasoning": "Test reasoning"
                }
            ],
            "meeting_summary": "Test summary"
        }
        """
        
        result = configured_board_swarm._parse_board_decisions(valid_json)
        
        assert isinstance(result, BoardSpec)
        assert result.plan == "Test plan"
        assert len(result.orders) == 1
        assert len(result.decisions) == 1
        assert result.meeting_summary == "Test summary"
    
    def test_parse_board_decisions_invalid_json(self, configured_board_swarm):
        """Test parsing invalid JSON board decisions."""
        invalid_json = "Invalid JSON content"
        
        result = configured_board_swarm._parse_board_decisions(invalid_json)
        
        assert isinstance(result, BoardSpec)
        assert result.plan == invalid_json
        assert len(result.orders) == 0
        assert len(result.decisions) == 0
        assert result.meeting_summary == "Parsing failed, using raw output"
    
    def test_run_board_meeting(self, configured_board_swarm):
        """Test running a complete board meeting."""
        task = "Test board meeting task"
        
        with patch.object(configured_board_swarm, '_conduct_board_discussion') as mock_discuss:
            with patch.object(configured_board_swarm, '_parse_board_decisions') as mock_parse:
                mock_discuss.return_value = "Board discussion"
                mock_parse.return_value = BoardSpec(
                    plan="Test plan",
                    orders=[],
                    decisions=[],
                    meeting_summary="Test summary"
                )
                
                result = configured_board_swarm.run_board_meeting(task)
                
                assert isinstance(result, BoardSpec)
                mock_discuss.assert_called_once()
                mock_parse.assert_called_once_with("Board discussion")


class TestTaskExecution:
    """Test task execution methods."""
    
    def test_call_single_agent(self, configured_board_swarm):
        """Test calling a single agent."""
        agent_name = "Agent1"
        task = "Test task"
        
        with patch.object(configured_board_swarm.agents[0], 'run') as mock_run:
            mock_run.return_value = "Agent response"
            result = configured_board_swarm._call_single_agent(agent_name, task)
            
            assert result == "Agent response"
            mock_run.assert_called_once()
    
    def test_call_single_agent_not_found(self, configured_board_swarm):
        """Test calling a non-existent agent."""
        with pytest.raises(ValueError, match="Agent 'NonExistent' not found"):
            configured_board_swarm._call_single_agent("NonExistent", "Test task")
    
    def test_execute_single_order(self, configured_board_swarm):
        """Test executing a single order."""
        order = BoardOrder(
            agent_name="Agent1",
            task="Test order task",
            priority=1,
            assigned_by="Chairman"
        )
        
        with patch.object(configured_board_swarm, '_call_single_agent') as mock_call:
            mock_call.return_value = "Order execution result"
            result = configured_board_swarm._execute_single_order(order)
            
            assert result == "Order execution result"
            mock_call.assert_called_once_with(
                agent_name="Agent1",
                task="Test order task"
            )
    
    def test_execute_orders(self, configured_board_swarm):
        """Test executing multiple orders."""
        orders = [
            BoardOrder(agent_name="Agent1", task="Task 1", priority=1),
            BoardOrder(agent_name="Agent2", task="Task 2", priority=2),
        ]
        
        with patch.object(configured_board_swarm, '_execute_single_order') as mock_execute:
            mock_execute.side_effect = ["Result 1", "Result 2"]
            results = configured_board_swarm._execute_orders(orders)
            
            assert len(results) == 2
            assert results[0]["agent_name"] == "Agent1"
            assert results[0]["output"] == "Result 1"
            assert results[1]["agent_name"] == "Agent2"
            assert results[1]["output"] == "Result 2"
    
    def test_generate_board_feedback(self, configured_board_swarm):
        """Test generating board feedback."""
        outputs = [
            {"agent_name": "Agent1", "output": "Output 1"},
            {"agent_name": "Agent2", "output": "Output 2"}
        ]
        
        with patch.object(configured_board_swarm.board_members[0].agent, 'run') as mock_run:
            mock_run.return_value = "Board feedback"
            result = configured_board_swarm._generate_board_feedback(outputs)
            
            assert result == "Board feedback"
            mock_run.assert_called_once()
    
    def test_generate_board_feedback_no_chairman(self, sample_agents):
        """Test generating feedback when no chairman is found."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        swarm.board_members = []  # Remove all board members
        
        with pytest.raises(ValueError, match="No chairman found for feedback"):
            swarm._generate_board_feedback([])


class TestStepAndRunMethods:
    """Test step and run methods."""
    
    def test_step_method(self, configured_board_swarm):
        """Test the step method."""
        task = "Test step task"
        
        with patch.object(configured_board_swarm, 'run_board_meeting') as mock_meeting:
            with patch.object(configured_board_swarm, '_execute_orders') as mock_execute:
                with patch.object(configured_board_swarm, '_generate_board_feedback') as mock_feedback:
                    mock_meeting.return_value = BoardSpec(
                        plan="Test plan",
                        orders=[BoardOrder(agent_name="Agent1", task="Task 1")],
                        decisions=[],
                        meeting_summary="Test summary"
                    )
                    mock_execute.return_value = [{"agent_name": "Agent1", "output": "Result"}]
                    mock_feedback.return_value = "Board feedback"
                    
                    result = configured_board_swarm.step(task)
                    
                    assert result == "Board feedback"
                    mock_meeting.assert_called_once_with(task=task, img=None)
                    mock_execute.assert_called_once()
                    mock_feedback.assert_called_once()
    
    def test_step_method_no_feedback(self, configured_board_swarm):
        """Test the step method with feedback disabled."""
        configured_board_swarm.board_feedback_on = False
        task = "Test step task"
        
        with patch.object(configured_board_swarm, 'run_board_meeting') as mock_meeting:
            with patch.object(configured_board_swarm, '_execute_orders') as mock_execute:
                mock_meeting.return_value = BoardSpec(
                    plan="Test plan",
                    orders=[BoardOrder(agent_name="Agent1", task="Task 1")],
                    decisions=[],
                    meeting_summary="Test summary"
                )
                mock_execute.return_value = [{"agent_name": "Agent1", "output": "Result"}]
                
                result = configured_board_swarm.step(task)
                
                assert result == [{"agent_name": "Agent1", "output": "Result"}]
    
    def test_run_method(self, configured_board_swarm):
        """Test the run method."""
        task = "Test run task"
        
        with patch.object(configured_board_swarm, 'step') as mock_step:
            with patch.object(configured_board_swarm, 'conversation') as mock_conversation:
                mock_step.return_value = "Step result"
                mock_conversation.add = Mock()
                
                result = configured_board_swarm.run(task)
                
                assert mock_step.call_count == 2  # max_loops = 2
                assert mock_conversation.add.call_count == 2
    
    def test_arun_method(self, configured_board_swarm):
        """Test the async run method."""
        task = "Test async run task"
        
        with patch.object(configured_board_swarm, 'run') as mock_run:
            mock_run.return_value = "Async result"
            
            async def test_async():
                result = await configured_board_swarm.arun(task)
                return result
            
            result = asyncio.run(test_async())
            assert result == "Async result"
            mock_run.assert_called_once_with(task=task, img=None)


# Integration tests
class TestBoardOfDirectorsSwarmIntegration:
    """Integration tests for BoardOfDirectorsSwarm."""
    
    def test_full_workflow_integration(self, sample_agents):
        """Test full workflow integration."""
        swarm = BoardOfDirectorsSwarm(
            agents=sample_agents,
            verbose=False,
            max_loops=1
        )
        
        task = "Create a simple report"
        
        # Mock the board discussion to return structured output
        mock_board_output = """
        {
            "plan": "Create a comprehensive report",
            "orders": [
                {
                    "agent_name": "Agent1",
                    "task": "Research the topic",
                    "priority": 1,
                    "assigned_by": "Chairman"
                },
                {
                    "agent_name": "Agent2",
                    "task": "Write the report",
                    "priority": 2,
                    "assigned_by": "Chairman"
                }
            ],
            "decisions": [
                {
                    "decision_type": "consensus",
                    "decision": "Proceed with report creation",
                    "votes_for": 3,
                    "votes_against": 0,
                    "abstentions": 0,
                    "reasoning": "Report is needed for decision making"
                }
            ],
            "meeting_summary": "Board agreed to create a comprehensive report"
        }
        """
        
        with patch.object(swarm.board_members[0].agent, 'run') as mock_run:
            mock_run.return_value = mock_board_output
            result = swarm.run(task)
            
            assert result is not None
            assert isinstance(result, dict)
    
    def test_board_member_management_integration(self, sample_agents):
        """Test board member management integration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents)
        
        # Test adding a new board member
        new_member = BoardMember(
            agent=sample_agents[0],
            role=BoardMemberRole.MEMBER,
            voting_weight=1.0,
            expertise_areas=["testing"]
        )
        
        initial_count = len(swarm.board_members)
        swarm.add_board_member(new_member)
        assert len(swarm.board_members) == initial_count + 1
        
        # Test removing a board member
        member_name = swarm.board_members[0].agent.agent_name
        swarm.remove_board_member(member_name)
        assert len(swarm.board_members) == initial_count
        
        # Test getting board member
        member = swarm.get_board_member(swarm.board_members[0].agent.agent_name)
        assert member is not None


# Parameterized tests
@pytest.mark.parametrize("max_loops", [1, 2, 3])
def test_max_loops_parameterization(sample_agents, max_loops):
    """Test swarm with different max_loops values."""
    swarm = BoardOfDirectorsSwarm(agents=sample_agents, max_loops=max_loops)
    assert swarm.max_loops == max_loops


@pytest.mark.parametrize("decision_threshold", [0.5, 0.6, 0.7, 0.8, 0.9])
def test_decision_threshold_parameterization(sample_agents, decision_threshold):
    """Test swarm with different decision threshold values."""
    swarm = BoardOfDirectorsSwarm(agents=sample_agents, decision_threshold=decision_threshold)
    assert swarm.decision_threshold == decision_threshold


@pytest.mark.parametrize("board_model", ["gpt-4o-mini", "gpt-4", "claude-3-sonnet"])
def test_board_model_parameterization(sample_agents, board_model):
    """Test swarm with different board models."""
    swarm = BoardOfDirectorsSwarm(agents=sample_agents, board_model_name=board_model)
    assert swarm.board_model_name == board_model


# Error handling tests
class TestBoardOfDirectorsSwarmErrorHandling:
    """Test error handling in BoardOfDirectorsSwarm."""
    
    def test_initialization_error_handling(self):
        """Test error handling during initialization."""
        with pytest.raises(ValueError):
            BoardOfDirectorsSwarm(agents=[])
    
    def test_board_meeting_error_handling(self, configured_board_swarm):
        """Test error handling during board meeting."""
        with patch.object(configured_board_swarm, '_conduct_board_discussion') as mock_discuss:
            mock_discuss.side_effect = Exception("Board meeting failed")
            
            with pytest.raises(Exception, match="Board meeting failed"):
                configured_board_swarm.run_board_meeting("Test task")
    
    def test_task_execution_error_handling(self, configured_board_swarm):
        """Test error handling during task execution."""
        with patch.object(configured_board_swarm, '_call_single_agent') as mock_call:
            mock_call.side_effect = Exception("Task execution failed")
            
            with pytest.raises(Exception, match="Task execution failed"):
                configured_board_swarm._call_single_agent("Agent1", "Test task")
    
    def test_order_execution_error_handling(self, configured_board_swarm):
        """Test error handling during order execution."""
        orders = [BoardOrder(agent_name="Agent1", task="Task 1")]
        
        with patch.object(configured_board_swarm, '_execute_single_order') as mock_execute:
            mock_execute.side_effect = Exception("Order execution failed")
            
            # Should not raise exception, but log error
            results = configured_board_swarm._execute_orders(orders)
            assert len(results) == 1
            assert "Error" in results[0]["output"]


# Performance tests
class TestBoardOfDirectorsSwarmPerformance:
    """Test performance characteristics of BoardOfDirectorsSwarm."""
    
    def test_parallel_execution_performance(self, sample_agents):
        """Test parallel execution performance."""
        import time
        
        swarm = BoardOfDirectorsSwarm(
            agents=sample_agents,
            max_workers=3,
            verbose=False
        )
        
        # Create multiple orders
        orders = [
            BoardOrder(agent_name=f"Agent{i+1}", task=f"Task {i+1}")
            for i in range(3)
        ]
        
        start_time = time.time()
        
        with patch.object(swarm, '_execute_single_order') as mock_execute:
            mock_execute.side_effect = lambda order: f"Result for {order.task}"
            results = swarm._execute_orders(orders)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        assert len(results) == 3
        assert execution_time < 1.0  # Should complete quickly with parallel execution
    
    def test_memory_usage(self, sample_agents):
        """Test memory usage characteristics."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple swarms
        swarms = []
        for i in range(5):
            swarm = BoardOfDirectorsSwarm(
                agents=sample_agents,
                name=f"Swarm{i}",
                verbose=False
            )
            swarms.append(swarm)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


# Configuration tests
class TestBoardOfDirectorsSwarmConfiguration:
    """Test configuration options for BoardOfDirectorsSwarm."""
    
    def test_verbose_configuration(self, sample_agents):
        """Test verbose configuration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, verbose=True)
        assert swarm.verbose is True
        
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, verbose=False)
        assert swarm.verbose is False
    
    def test_collaboration_prompt_configuration(self, sample_agents):
        """Test collaboration prompt configuration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, add_collaboration_prompt=True)
        assert swarm.add_collaboration_prompt is True
        
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, add_collaboration_prompt=False)
        assert swarm.add_collaboration_prompt is False
    
    def test_board_feedback_configuration(self, sample_agents):
        """Test board feedback configuration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, board_feedback_on=True)
        assert swarm.board_feedback_on is True
        
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, board_feedback_on=False)
        assert swarm.board_feedback_on is False
    
    def test_voting_configuration(self, sample_agents):
        """Test voting configuration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, enable_voting=True)
        assert swarm.enable_voting is True
        
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, enable_voting=False)
        assert swarm.enable_voting is False
    
    def test_consensus_configuration(self, sample_agents):
        """Test consensus configuration."""
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, enable_consensus=True)
        assert swarm.enable_consensus is True
        
        swarm = BoardOfDirectorsSwarm(agents=sample_agents, enable_consensus=False)
        assert swarm.enable_consensus is False


# Real integration tests (skipped if no API key)
@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OpenAI API key not available"
)
class TestBoardOfDirectorsSwarmRealIntegration:
    """Real integration tests for BoardOfDirectorsSwarm."""
    
    def test_real_board_meeting(self):
        """Test real board meeting with actual API calls."""
        # Create real agents
        agents = [
            Agent(
                agent_name="Researcher",
                agent_description="Research analyst",
                model_name="gpt-4o-mini",
                max_loops=1
            ),
            Agent(
                agent_name="Writer",
                agent_description="Content writer",
                model_name="gpt-4o-mini",
                max_loops=1
            )
        ]
        
        swarm = BoardOfDirectorsSwarm(
            agents=agents,
            verbose=False,
            max_loops=1
        )
        
        task = "Create a brief market analysis report"
        
        result = swarm.run(task)
        
        assert result is not None
        assert isinstance(result, dict)
        assert "conversation_history" in result
    
    def test_real_board_member_management(self):
        """Test real board member management."""
        agents = [
            Agent(
                agent_name="TestAgent",
                agent_description="Test agent",
                model_name="gpt-4o-mini",
                max_loops=1
            )
        ]
        
        swarm = BoardOfDirectorsSwarm(agents=agents, verbose=False)
        
        # Test board summary
        summary = swarm.get_board_summary()
        assert summary["total_members"] == 3  # Default board
        assert summary["total_agents"] == 1


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 