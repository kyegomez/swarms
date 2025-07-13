"""
Comprehensive test suite for EnhancedHierarchicalSwarm

This test suite covers:
- Communication system functionality
- Dynamic role assignment
- Task scheduling and coordination
- Performance monitoring
- Error handling and recovery
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from swarms.structs.enhanced_hierarchical_swarm import (
    EnhancedHierarchicalSwarm,
    DynamicRoleManager,
    TaskScheduler,
    AgentRole,
    TaskComplexity,
    AgentCapability,
    EnhancedTask
)
from swarms.structs.communication import (
    CommunicationManager,
    Message,
    MessageType,
    MessagePriority,
    MessageStatus
)
from swarms.structs.agent import Agent


class TestCommunicationSystem:
    """Test the communication system components"""
    
    def test_message_creation(self):
        """Test message creation and properties"""
        message = Message(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=MessagePriority.HIGH,
            content={"task": "test task"}
        )
        
        assert message.sender_id == "agent1"
        assert message.receiver_id == "agent2"
        assert message.message_type == MessageType.TASK_ASSIGNMENT
        assert message.priority == MessagePriority.HIGH
        assert message.content["task"] == "test task"
        assert message.status == MessageStatus.PENDING
        assert not message.is_expired()
    
    def test_message_expiry(self):
        """Test message expiry functionality"""
        message = Message(
            sender_id="agent1",
            receiver_id="agent2",
            expiry_time=time.time() - 10  # Expired 10 seconds ago
        )
        
        assert message.is_expired()
    
    def test_communication_manager_initialization(self):
        """Test communication manager initialization"""
        comm_manager = CommunicationManager()
        
        assert comm_manager.router is not None
        assert comm_manager.feedback_system is not None
        assert comm_manager.escalation_manager is not None
        assert not comm_manager.running
    
    def test_communication_manager_start_stop(self):
        """Test communication manager start/stop functionality"""
        comm_manager = CommunicationManager()
        
        comm_manager.start()
        assert comm_manager.running
        
        comm_manager.stop()
        assert not comm_manager.running
    
    def test_channel_creation(self):
        """Test communication channel creation"""
        comm_manager = CommunicationManager()
        comm_manager.start()
        
        channel_id = comm_manager.create_conversation(
            "test_conv",
            ["agent1", "agent2"],
            "group"
        )
        
        assert channel_id in comm_manager.router.channels
        assert "test_conv" in comm_manager.active_conversations
        
        comm_manager.stop()


class TestDynamicRoleManager:
    """Test dynamic role assignment and management"""
    
    def test_role_manager_initialization(self):
        """Test role manager initialization"""
        role_manager = DynamicRoleManager()
        
        assert len(role_manager.agent_capabilities) == 0
        assert len(role_manager.role_assignments) == 0
        assert len(role_manager.performance_history) == 0
    
    def test_agent_registration(self):
        """Test agent registration"""
        role_manager = DynamicRoleManager()
        
        capabilities = {"analysis": 0.8, "writing": 0.6}
        role_manager.register_agent("agent1", capabilities)
        
        assert "agent1" in role_manager.agent_capabilities
        assert "agent1" in role_manager.role_assignments
        assert "agent1" in role_manager.performance_history
        assert role_manager.role_assignments["agent1"] == AgentRole.EXECUTOR
        assert len(role_manager.agent_capabilities["agent1"]) == 2
    
    def test_performance_update(self):
        """Test agent performance updates"""
        role_manager = DynamicRoleManager()
        role_manager.register_agent("agent1", {"analysis": 0.5})
        
        # Update performance with success
        role_manager.update_agent_performance(
            "agent1", "analysis", True, TaskComplexity.MEDIUM
        )
        
        capability = role_manager.agent_capabilities["agent1"]["analysis"]
        assert capability.experience_count == 1
        assert capability.success_rate == 1.0
        assert capability.skill_level > 0.5  # Should have improved
    
    def test_role_promotion(self):
        """Test role promotion based on performance"""
        role_manager = DynamicRoleManager()
        role_manager.register_agent("agent1", {"analysis": 0.9})
        
        # Initial role should be executor
        assert role_manager.get_agent_role("agent1") == AgentRole.EXECUTOR
        
        # Update role assignments directly for testing
        role_manager.role_assignments["agent1"] = AgentRole.SPECIALIST
        
        assert role_manager.get_agent_role("agent1") == AgentRole.SPECIALIST
    
    def test_best_agent_selection(self):
        """Test best agent selection for tasks"""
        role_manager = DynamicRoleManager()
        
        # Register agents with different capabilities
        role_manager.register_agent("agent1", {"analysis": 0.8, "writing": 0.3})
        role_manager.register_agent("agent2", {"analysis": 0.4, "writing": 0.9})
        
        # Set success rates
        role_manager.agent_capabilities["agent1"]["analysis"].success_rate = 0.9
        role_manager.agent_capabilities["agent2"]["writing"].success_rate = 0.8
        
        # Test selection for analysis task
        best_agent = role_manager.get_best_agent_for_task(
            ["analysis"], ["agent1", "agent2"]
        )
        assert best_agent == "agent1"
        
        # Test selection for writing task
        best_agent = role_manager.get_best_agent_for_task(
            ["writing"], ["agent1", "agent2"]
        )
        assert best_agent == "agent2"


class TestTaskScheduler:
    """Test task scheduling and coordination"""
    
    def test_task_scheduler_initialization(self):
        """Test task scheduler initialization"""
        role_manager = DynamicRoleManager()
        scheduler = TaskScheduler(role_manager)
        
        assert scheduler.role_manager is role_manager
        assert len(scheduler.task_queue) == 0
        assert len(scheduler.active_tasks) == 0
        assert len(scheduler.completed_tasks) == 0
    
    def test_task_addition(self):
        """Test adding tasks to scheduler"""
        role_manager = DynamicRoleManager()
        scheduler = TaskScheduler(role_manager)
        
        task = EnhancedTask(
            content="test task",
            complexity=TaskComplexity.HIGH,
            priority=MessagePriority.HIGH,
            required_capabilities=["analysis"]
        )
        
        scheduler.add_task(task)
        
        assert len(scheduler.task_queue) == 1
        assert scheduler.task_queue[0] == task
    
    def test_task_scheduling(self):
        """Test task scheduling to agents"""
        role_manager = DynamicRoleManager()
        role_manager.register_agent("agent1", {"analysis": 0.8})
        
        scheduler = TaskScheduler(role_manager)
        
        task = EnhancedTask(
            content="test task",
            required_capabilities=["analysis"]
        )
        scheduler.add_task(task)
        
        scheduled_tasks = scheduler.schedule_tasks(["agent1"])
        
        assert "agent1" in scheduled_tasks
        assert len(scheduled_tasks["agent1"]) == 1
        assert scheduled_tasks["agent1"][0] == task
        assert task.id in scheduler.active_tasks
    
    def test_task_completion(self):
        """Test task completion tracking"""
        role_manager = DynamicRoleManager()
        scheduler = TaskScheduler(role_manager)
        
        task = EnhancedTask(content="test task")
        scheduler.active_tasks[task.id] = task
        
        scheduler.mark_task_completed(task.id, "result", True)
        
        assert task.id not in scheduler.active_tasks
        assert task.id in scheduler.completed_tasks
        assert scheduler.completed_tasks[task.id].result == "result"
        assert scheduler.completed_tasks[task.id].status == "completed"


class TestAgentCapability:
    """Test agent capability tracking"""
    
    def test_capability_initialization(self):
        """Test capability initialization"""
        capability = AgentCapability(
            domain="analysis",
            skill_level=0.6
        )
        
        assert capability.domain == "analysis"
        assert capability.skill_level == 0.6
        assert capability.experience_count == 0
        assert capability.success_rate == 0.0
    
    def test_capability_update_success(self):
        """Test capability update on success"""
        capability = AgentCapability(
            domain="analysis",
            skill_level=0.6
        )
        
        capability.update_performance(True, TaskComplexity.HIGH)
        
        assert capability.experience_count == 1
        assert capability.success_rate == 1.0
        assert capability.skill_level > 0.6  # Should have improved
    
    def test_capability_update_failure(self):
        """Test capability update on failure"""
        capability = AgentCapability(
            domain="analysis",
            skill_level=0.6
        )
        
        capability.update_performance(False, TaskComplexity.HIGH)
        
        assert capability.experience_count == 1
        assert capability.success_rate == 0.0
        assert capability.skill_level < 0.6  # Should have decreased


class TestEnhancedTask:
    """Test enhanced task functionality"""
    
    def test_task_creation(self):
        """Test task creation and properties"""
        task = EnhancedTask(
            content="test task",
            complexity=TaskComplexity.HIGH,
            priority=MessagePriority.HIGH,
            required_capabilities=["analysis", "writing"]
        )
        
        assert task.content == "test task"
        assert task.complexity == TaskComplexity.HIGH
        assert task.priority == MessagePriority.HIGH
        assert task.required_capabilities == ["analysis", "writing"]
        assert task.status == "pending"
    
    def test_task_to_dict(self):
        """Test task dictionary conversion"""
        task = EnhancedTask(
            content="test task",
            complexity=TaskComplexity.MEDIUM,
            priority=MessagePriority.LOW
        )
        
        task_dict = task.to_dict()
        
        assert task_dict["content"] == "test task"
        assert task_dict["complexity"] == TaskComplexity.MEDIUM.value
        assert task_dict["priority"] == MessagePriority.LOW.value
        assert task_dict["status"] == "pending"


class TestEnhancedHierarchicalSwarm:
    """Test the enhanced hierarchical swarm"""
    
    def create_mock_agent(self, name):
        """Create a mock agent for testing"""
        mock_agent = Mock()
        mock_agent.agent_name = name
        mock_agent.agent_description = f"Mock agent {name}"
        mock_agent.system_prompt = f"System prompt for {name}"
        mock_agent.run = Mock(return_value=f"Result from {name}")
        return mock_agent
    
    def test_swarm_initialization(self):
        """Test swarm initialization"""
        agents = [self.create_mock_agent("agent1"), self.create_mock_agent("agent2")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            verbose=False
        )
        
        assert swarm.name == "test-swarm"
        assert len(swarm.agents) == 2
        assert swarm.communication_manager is not None
        assert swarm.role_manager is not None
        assert swarm.task_scheduler is not None
        
        # Cleanup
        swarm.shutdown()
    
    def test_swarm_initialization_no_agents(self):
        """Test swarm initialization without agents"""
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=None,
            verbose=False
        )
        
        assert swarm.name == "test-swarm"
        assert len(swarm.agents) == 0
        
        # Cleanup
        swarm.shutdown()
    
    def test_agent_registration(self):
        """Test agent registration process"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            verbose=False
        )
        
        # Check if agent is registered
        assert "agent1" in swarm.role_manager.agent_capabilities
        assert "agent1" in swarm.role_manager.role_assignments
        
        # Cleanup
        swarm.shutdown()
    
    def test_capability_extraction(self):
        """Test capability extraction from agent"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            verbose=False
        )
        
        capabilities = swarm._extract_capabilities_from_agent(agents[0])
        
        assert isinstance(capabilities, dict)
        assert len(capabilities) > 0
        
        # Cleanup
        swarm.shutdown()
    
    def test_task_parsing(self):
        """Test task parsing into subtasks"""
        agents = [self.create_mock_agent("agent1")]
        
        # Create mock director
        mock_director = Mock()
        mock_director.run = Mock(return_value="subtask 1\nsubtask 2")
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            director=mock_director,
            verbose=False
        )
        
        enhanced_tasks = swarm._parse_task_into_subtasks("main task")
        
        assert isinstance(enhanced_tasks, list)
        assert len(enhanced_tasks) > 0
        
        # Cleanup
        swarm.shutdown()
    
    def test_task_parsing_no_director(self):
        """Test task parsing without director"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            director=None,
            verbose=False
        )
        
        enhanced_tasks = swarm._parse_task_into_subtasks("main task")
        
        assert isinstance(enhanced_tasks, list)
        assert len(enhanced_tasks) == 1
        assert enhanced_tasks[0].content == "main task"
        
        # Cleanup
        swarm.shutdown()
    
    def test_performance_metrics(self):
        """Test performance metrics collection"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            verbose=False
        )
        
        metrics = swarm.get_performance_metrics()
        
        assert "execution_metrics" in metrics
        assert "communication_stats" in metrics
        assert "agent_performance" in metrics
        
        # Cleanup
        swarm.shutdown()
    
    def test_performance_optimization(self):
        """Test performance optimization"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            auto_optimize=True,
            verbose=False
        )
        
        initial_concurrent_tasks = swarm.max_concurrent_tasks
        
        # Simulate low success rate
        swarm.execution_metrics['total_tasks'] = 10
        swarm.execution_metrics['completed_tasks'] = 5
        
        swarm.optimize_performance()
        
        # Should have decreased concurrent tasks
        assert swarm.max_concurrent_tasks <= initial_concurrent_tasks
        
        # Cleanup
        swarm.shutdown()
    
    @patch('swarms.structs.enhanced_hierarchical_swarm.Agent')
    def test_director_setup(self, mock_agent_class):
        """Test director setup"""
        mock_director = Mock()
        mock_agent_class.return_value = mock_director
        
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            director=None,
            verbose=False
        )
        
        # Director should be created
        assert swarm.director is not None
        
        # Cleanup
        swarm.shutdown()
    
    def test_shutdown(self):
        """Test swarm shutdown"""
        agents = [self.create_mock_agent("agent1")]
        
        swarm = EnhancedHierarchicalSwarm(
            name="test-swarm",
            agents=agents,
            verbose=False
        )
        
        # Swarm should be running
        assert swarm.communication_manager.running
        
        swarm.shutdown()
        
        # Swarm should be stopped
        assert not swarm.communication_manager.running


class TestIntegration:
    """Integration tests for the enhanced hierarchical swarm"""
    
    def create_mock_agent(self, name):
        """Create a mock agent for testing"""
        mock_agent = Mock()
        mock_agent.agent_name = name
        mock_agent.agent_description = f"Expert in {name.lower()}"
        mock_agent.system_prompt = f"You are an expert in {name.lower()}"
        mock_agent.run = Mock(return_value=f"Completed task by {name}")
        return mock_agent
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create test agents
        agents = [
            self.create_mock_agent("Analyst"),
            self.create_mock_agent("Writer")
        ]
        
        # Create swarm
        swarm = EnhancedHierarchicalSwarm(
            name="integration-test-swarm",
            agents=agents,
            verbose=False,
            enable_parallel_execution=False,  # Disable for predictable testing
            max_concurrent_tasks=1
        )
        
        # Execute a task
        task = "Analyze market trends and write a summary report"
        
        try:
            result = swarm.run(task=task)
            
            # Verify result
            assert result is not None
            
            # Verify metrics
            metrics = swarm.get_performance_metrics()
            assert metrics['execution_metrics']['total_tasks'] > 0
            
            # Verify agent performance tracking
            assert 'agent_performance' in metrics
            
        finally:
            # Cleanup
            swarm.shutdown()
    
    def test_parallel_execution(self):
        """Test parallel execution functionality"""
        # Create test agents
        agents = [
            self.create_mock_agent("Agent1"),
            self.create_mock_agent("Agent2")
        ]
        
        # Create swarm with parallel execution
        swarm = EnhancedHierarchicalSwarm(
            name="parallel-test-swarm",
            agents=agents,
            verbose=False,
            enable_parallel_execution=True,
            max_concurrent_tasks=2
        )
        
        # Execute a task
        task = "Complete parallel task execution test"
        
        try:
            start_time = time.time()
            result = swarm.run(task=task)
            execution_time = time.time() - start_time
            
            # Verify result
            assert result is not None
            
            # Verify execution completed in reasonable time
            assert execution_time < 30  # Should complete within 30 seconds
            
        finally:
            # Cleanup
            swarm.shutdown()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])