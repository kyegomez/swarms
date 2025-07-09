"""
Enhanced Hierarchical Swarm with Improved Communication

This module integrates the enhanced communication system and hierarchical cooperation
protocols with the hierarchical swarm to provide a production-ready, highly reliable
multi-agent coordination system.
"""

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import uuid

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.output_types import OutputType
from swarms.utils.formatter import formatter
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.history_output_formatter import history_output_formatter

# Import enhanced communication components
try:
    from swarms.communication.enhanced_communication import (
        MessageBroker, EnhancedMessage, MessageType, MessagePriority,
        CommunicationProtocol, get_global_broker
    )
    from swarms.communication.hierarchical_cooperation import (
        HierarchicalCoordinator, HierarchicalAgent, HierarchicalRole,
        CooperationPattern, HierarchicalTask, AgentCapability,
        get_global_coordinator
    )
    HAS_ENHANCED_COMMUNICATION = True
except ImportError:
    HAS_ENHANCED_COMMUNICATION = False
    logger = initialize_logger(log_folder="enhanced_hierarchical_swarm")
    logger.warning("Enhanced communication modules not available, using fallback implementation")

logger = initialize_logger(log_folder="enhanced_hierarchical_swarm")


class EnhancedAgent(Agent):
    """Enhanced agent that integrates with the communication system"""
    
    def __init__(
        self,
        agent_name: str,
        role: str = "worker",
        specializations: Optional[List[str]] = None,
        max_concurrent_tasks: int = 3,
        **kwargs
    ):
        super().__init__(agent_name=agent_name, **kwargs)
        
        self.role = role
        self.specializations = specializations or []
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0
        self.performance_history = []
        
        # Communication enhancement
        if HAS_ENHANCED_COMMUNICATION:
            self._setup_enhanced_communication()
    
    def _setup_enhanced_communication(self):
        """Set up enhanced communication capabilities"""
        try:
            # Map role string to HierarchicalRole enum
            role_mapping = {
                "director": HierarchicalRole.DIRECTOR,
                "supervisor": HierarchicalRole.SUPERVISOR,
                "coordinator": HierarchicalRole.COORDINATOR,
                "worker": HierarchicalRole.WORKER,
                "specialist": HierarchicalRole.SPECIALIST
            }
            
            hierarchical_role = role_mapping.get(self.role.lower(), HierarchicalRole.WORKER)
            
            # Create hierarchical agent wrapper
            self.hierarchical_agent = HierarchicalAgent(
                agent_id=self.agent_name,
                role=hierarchical_role,
                broker=get_global_broker(),
                coordinator=get_global_coordinator()
            )
            
            # Create agent capability
            self.capability = AgentCapability(
                name=f"{self.agent_name}_capability",
                proficiency=0.8,  # Default proficiency
                availability=1.0,
                current_load=0.0,
                specializations=self.specializations,
                max_concurrent_tasks=self.max_concurrent_tasks
            )
            
            # Register with coordinator
            coordinator = get_global_coordinator()
            coordinator.register_agent(
                self.hierarchical_agent,
                hierarchical_role,
                capabilities=self.capability
            )
            
            logger.info(f"Enhanced communication enabled for agent {self.agent_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup enhanced communication: {e}")
            self.hierarchical_agent = None
            self.capability = None
    
    def run(self, task: str, *args, **kwargs) -> str:
        """Enhanced run method with communication integration"""
        start_time = time.time()
        
        try:
            # Update current load
            self.current_tasks += 1
            if self.capability:
                self.capability.current_load = min(1.0, self.current_tasks / self.max_concurrent_tasks)
            
            # Execute the task
            result = super().run(task, *args, **kwargs)
            
            # Track performance
            execution_time = time.time() - start_time
            self.performance_history.append({
                'timestamp': datetime.now(),
                'execution_time': execution_time,
                'success': True,
                'task_length': len(task)
            })
            
            # Keep only last 100 performance records
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)
            
            return result
            
        except Exception as e:
            # Track failure
            execution_time = time.time() - start_time
            self.performance_history.append({
                'timestamp': datetime.now(),
                'execution_time': execution_time,
                'success': False,
                'error': str(e),
                'task_length': len(task)
            })
            
            logger.error(f"Agent {self.agent_name} failed to execute task: {e}")
            raise
            
        finally:
            # Update current load
            self.current_tasks = max(0, self.current_tasks - 1)
            if self.capability:
                self.capability.current_load = min(1.0, self.current_tasks / self.max_concurrent_tasks)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the agent"""
        if not self.performance_history:
            return {
                'total_tasks': 0,
                'success_rate': 1.0,
                'avg_execution_time': 0.0,
                'current_load': 0.0
            }
        
        total_tasks = len(self.performance_history)
        successful_tasks = sum(1 for record in self.performance_history if record['success'])
        success_rate = successful_tasks / total_tasks
        avg_execution_time = sum(record['execution_time'] for record in self.performance_history) / total_tasks
        
        return {
            'total_tasks': total_tasks,
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'current_load': self.current_tasks / self.max_concurrent_tasks,
            'specializations': self.specializations,
            'role': self.role
        }


class EnhancedHierarchicalSwarm(BaseSwarm):
    """
    Enhanced hierarchical swarm with improved communication, reliability, and cooperation.
    
    Features:
    - Reliable message passing with retry mechanisms
    - Rate limiting and frequency management
    - Advanced hierarchical cooperation patterns
    - Real-time agent health monitoring
    - Intelligent task delegation and escalation
    - Load balancing and performance optimization
    - Comprehensive error handling and recovery
    """
    
    def __init__(
        self,
        name: str = "EnhancedHierarchicalSwarm",
        description: str = "Enhanced hierarchical swarm with improved communication",
        agents: Optional[List[Union[Agent, EnhancedAgent]]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        director_agent: Optional[Agent] = None,
        communication_rate_limit: Tuple[int, float] = (100, 60.0),  # 100 messages per 60 seconds
        task_timeout: int = 300,
        cooperation_pattern: CooperationPattern = CooperationPattern.COMMAND_CONTROL,
        enable_load_balancing: bool = True,
        enable_auto_escalation: bool = True,
        enable_collaboration: bool = True,
        health_check_interval: float = 30.0,
        max_concurrent_tasks: int = 10,
        *args,
        **kwargs
    ):
        """
        Initialize the enhanced hierarchical swarm.
        
        Args:
            name: Swarm name
            description: Swarm description
            agents: List of agents in the swarm
            max_loops: Maximum execution loops
            output_type: Output format type
            director_agent: Designated director agent
            communication_rate_limit: (max_messages, time_window) for rate limiting
            task_timeout: Default task timeout in seconds
            cooperation_pattern: Default cooperation pattern
            enable_load_balancing: Enable automatic load balancing
            enable_auto_escalation: Enable automatic task escalation
            enable_collaboration: Enable agent collaboration
            health_check_interval: Health check interval in seconds
            max_concurrent_tasks: Maximum concurrent tasks across swarm
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents or [],
            max_loops=max_loops,
            *args,
            **kwargs
        )
        
        self.output_type = output_type
        self.director_agent = director_agent
        self.communication_rate_limit = communication_rate_limit
        self.task_timeout = task_timeout
        self.cooperation_pattern = cooperation_pattern
        self.enable_load_balancing = enable_load_balancing
        self.enable_auto_escalation = enable_auto_escalation
        self.enable_collaboration = enable_collaboration
        self.health_check_interval = health_check_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Enhanced components
        self.conversation = Conversation(time_enabled=True)
        self.enhanced_agents: List[EnhancedAgent] = []
        self.message_broker: Optional[MessageBroker] = None
        self.coordinator: Optional[HierarchicalCoordinator] = None
        self.task_executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Performance tracking
        self.execution_stats = {
            'tasks_executed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'avg_execution_time': 0.0,
            'messages_sent': 0,
            'delegation_count': 0,
            'escalation_count': 0,
            'collaboration_count': 0
        }
        
        # Initialize enhanced features
        self._initialize_enhanced_features()
    
    def _initialize_enhanced_features(self):
        """Initialize enhanced communication and cooperation features"""
        if not HAS_ENHANCED_COMMUNICATION:
            logger.warning("Enhanced communication not available, using basic implementation")
            return
        
        try:
            # Initialize message broker and coordinator
            self.message_broker = get_global_broker()
            self.coordinator = get_global_coordinator()
            
            # Convert agents to enhanced agents
            self._setup_enhanced_agents()
            
            # Set up director
            self._setup_director()
            
            logger.info(f"Enhanced features initialized for swarm {self.name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced features: {e}")
    
    def _setup_enhanced_agents(self):
        """Convert regular agents to enhanced agents"""
        self.enhanced_agents = []
        
        for i, agent in enumerate(self.agents):
            if isinstance(agent, EnhancedAgent):
                enhanced_agent = agent
            else:
                # Convert regular agent to enhanced agent
                enhanced_agent = EnhancedAgent(
                    agent_name=agent.agent_name,
                    role="worker",
                    system_prompt=getattr(agent, 'system_prompt', None),
                    llm=getattr(agent, 'llm', None),
                    max_loops=getattr(agent, 'max_loops', 1),
                    specializations=getattr(agent, 'specializations', [])
                )
            
            self.enhanced_agents.append(enhanced_agent)
        
        # Replace original agents list
        self.agents = self.enhanced_agents
        
        logger.info(f"Converted {len(self.enhanced_agents)} agents to enhanced agents")
    
    def _setup_director(self):
        """Set up the director agent"""
        if self.director_agent:
            # Use specified director
            if not isinstance(self.director_agent, EnhancedAgent):
                self.director_agent = EnhancedAgent(
                    agent_name=self.director_agent.agent_name,
                    role="director",
                    system_prompt=getattr(self.director_agent, 'system_prompt', None),
                    llm=getattr(self.director_agent, 'llm', None),
                    specializations=["coordination", "planning", "management"]
                )
        else:
            # Use first agent as director
            if self.enhanced_agents:
                self.director_agent = self.enhanced_agents[0]
                self.director_agent.role = "director"
                if hasattr(self.director_agent, 'hierarchical_agent'):
                    self.director_agent.hierarchical_agent.role = HierarchicalRole.DIRECTOR
        
        logger.info(f"Director set to: {self.director_agent.agent_name if self.director_agent else 'None'}")
    
    def run(self, task: str, img: str = None, *args, **kwargs) -> Union[str, Dict, List]:
        """
        Enhanced run method with improved communication and cooperation.
        
        Args:
            task: Task description to execute
            img: Optional image data
            
        Returns:
            Formatted output based on output_type
        """
        logger.info(f"Starting enhanced hierarchical swarm execution: {task}")
        
        # Add task to conversation
        self.conversation.add(role="User", content=f"Task: {task}")
        
        try:
            if HAS_ENHANCED_COMMUNICATION and self.coordinator:
                # Use enhanced communication system
                result = self._run_with_enhanced_communication(task, img, *args, **kwargs)
            else:
                # Fall back to basic implementation
                result = self._run_basic(task, img, *args, **kwargs)
            
            # Update statistics
            self.execution_stats['tasks_executed'] += 1
            self.execution_stats['tasks_successful'] += 1
            
            logger.info("Enhanced hierarchical swarm execution completed successfully")
            return result
            
        except Exception as e:
            self.execution_stats['tasks_executed'] += 1
            self.execution_stats['tasks_failed'] += 1
            logger.error(f"Enhanced hierarchical swarm execution failed: {e}")
            
            return {
                "error": str(e),
                "partial_results": getattr(self, '_partial_results', {}),
                "stats": self.execution_stats
            }
    
    def _run_with_enhanced_communication(self, task: str, img: str = None, *args, **kwargs) -> Any:
        """Run using enhanced communication system"""
        start_time = time.time()
        results = {}
        
        for loop in range(self.max_loops):
            logger.info(f"Starting loop {loop + 1}/{self.max_loops}")
            
            # Create hierarchical task
            task_id = self.coordinator.create_task(
                description=task,
                requesting_agent=self.director_agent.agent_name if self.director_agent else None,
                priority=MessagePriority.NORMAL,
                deadline=datetime.now() + timedelta(seconds=self.task_timeout),
                metadata={
                    'loop': loop + 1,
                    'max_loops': self.max_loops,
                    'swarm_name': self.name,
                    'cooperation_pattern': self.cooperation_pattern.value,
                    'img': img
                }
            )
            
            # Wait for task completion with timeout
            completion_timeout = self.task_timeout + 30  # Extra buffer
            start_wait = time.time()
            
            while time.time() - start_wait < completion_timeout:
                task_obj = self.coordinator._tasks.get(task_id)
                if task_obj and task_obj.status.value in ['completed', 'failed']:
                    break
                time.sleep(1.0)
            
            # Get final task state
            task_obj = self.coordinator._tasks.get(task_id)
            if task_obj:
                if task_obj.status.value == 'completed':
                    results[f'loop_{loop + 1}'] = task_obj.result
                    self.conversation.add(
                        role="System",
                        content=f"Loop {loop + 1} completed: {task_obj.result}"
                    )
                else:
                    results[f'loop_{loop + 1}'] = {
                        'error': task_obj.error_details or 'Task failed',
                        'status': task_obj.status.value
                    }
                    self.conversation.add(
                        role="System",
                        content=f"Loop {loop + 1} failed: {task_obj.error_details}"
                    )
            
            # Brief pause between loops
            if loop < self.max_loops - 1:
                time.sleep(0.5)
        
        # Update execution time
        execution_time = time.time() - start_time
        self.execution_stats['avg_execution_time'] = execution_time
        
        # Get swarm metrics
        swarm_metrics = self._get_swarm_metrics()
        
        # Format output
        return self._format_output(results, swarm_metrics)
    
    def _run_basic(self, task: str, img: str = None, *args, **kwargs) -> Any:
        """Fallback basic implementation"""
        results = {}
        
        for loop in range(self.max_loops):
            logger.info(f"Starting basic loop {loop + 1}/{self.max_loops}")
            
            # Simple round-robin execution
            loop_results = {}
            
            for agent in self.enhanced_agents:
                try:
                    agent_context = f"Loop {loop + 1}/{self.max_loops}: {task}"
                    result = agent.run(agent_context)
                    loop_results[agent.agent_name] = result
                    
                    self.conversation.add(
                        role=agent.agent_name,
                        content=f"Loop {loop + 1}: {result}"
                    )
                    
                except Exception as e:
                    logger.error(f"Agent {agent.agent_name} failed: {e}")
                    loop_results[agent.agent_name] = f"Error: {e}"
            
            results[f'loop_{loop + 1}'] = loop_results
        
        return self._format_output(results, {})
    
    def _format_output(self, results: Dict, metrics: Dict) -> Any:
        """Format output based on output_type"""
        if self.output_type == "dict":
            return {
                "results": results,
                "metrics": metrics,
                "conversation": self.conversation.to_dict(),
                "stats": self.execution_stats
            }
        elif self.output_type == "str":
            return history_output_formatter(self.conversation, "str")
        elif self.output_type == "list":
            return history_output_formatter(self.conversation, "list")
        else:
            # Default to conversation history
            return history_output_formatter(self.conversation, self.output_type)
    
    def _get_swarm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive swarm metrics"""
        metrics = {
            'total_agents': len(self.enhanced_agents),
            'execution_stats': self.execution_stats.copy(),
            'agent_performance': {}
        }
        
        # Add agent performance metrics
        for agent in self.enhanced_agents:
            metrics['agent_performance'][agent.agent_name] = agent.get_performance_metrics()
        
        # Add coordinator metrics if available
        if self.coordinator:
            coordinator_stats = self.coordinator.get_hierarchy_status()
            metrics['hierarchy_stats'] = coordinator_stats
        
        # Add message broker metrics if available
        if self.message_broker:
            broker_stats = self.message_broker.get_stats()
            metrics['communication_stats'] = broker_stats
        
        return metrics
    
    def delegate_task(
        self,
        task_description: str,
        from_agent: str,
        to_agent: str,
        reason: str = "delegation"
    ) -> bool:
        """Delegate a task from one agent to another"""
        if not HAS_ENHANCED_COMMUNICATION or not self.coordinator:
            logger.warning("Enhanced communication not available for task delegation")
            return False
        
        try:
            # Create task
            task_id = self.coordinator.create_task(
                description=task_description,
                requesting_agent=from_agent,
                metadata={'delegation_reason': reason}
            )
            
            # Delegate to target agent
            success = self.coordinator.delegate_task(task_id, from_agent, to_agent, reason)
            
            if success:
                self.execution_stats['delegation_count'] += 1
                logger.info(f"Successfully delegated task from {from_agent} to {to_agent}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delegate task: {e}")
            return False
    
    def escalate_task(
        self,
        task_description: str,
        agent_name: str,
        reason: str = "escalation"
    ) -> bool:
        """Escalate a task up the hierarchy"""
        if not HAS_ENHANCED_COMMUNICATION or not self.coordinator:
            logger.warning("Enhanced communication not available for task escalation")
            return False
        
        try:
            # Create task
            task_id = self.coordinator.create_task(
                description=task_description,
                requesting_agent=agent_name,
                metadata={'escalation_reason': reason}
            )
            
            # Escalate task
            success = self.coordinator.escalate_task(task_id, agent_name, reason)
            
            if success:
                self.execution_stats['escalation_count'] += 1
                logger.info(f"Successfully escalated task from {agent_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to escalate task: {e}")
            return False
    
    def broadcast_message(
        self,
        message: str,
        sender_agent: str,
        priority: str = "normal"
    ) -> bool:
        """Broadcast a message to all agents"""
        if not HAS_ENHANCED_COMMUNICATION:
            logger.warning("Enhanced communication not available for broadcasting")
            return False
        
        try:
            # Find sender agent
            sender = None
            for agent in self.enhanced_agents:
                if agent.agent_name == sender_agent:
                    sender = agent
                    break
            
            if not sender or not hasattr(sender, 'hierarchical_agent'):
                return False
            
            # Map priority
            priority_mapping = {
                "low": MessagePriority.LOW,
                "normal": MessagePriority.NORMAL,
                "high": MessagePriority.HIGH,
                "urgent": MessagePriority.URGENT,
                "critical": MessagePriority.CRITICAL
            }
            msg_priority = priority_mapping.get(priority.lower(), MessagePriority.NORMAL)
            
            # Send broadcast
            message_id = sender.hierarchical_agent.broadcast_message(
                content=message,
                message_type=MessageType.BROADCAST,
                priority=msg_priority
            )
            
            if message_id:
                self.execution_stats['messages_sent'] += 1
                logger.info(f"Broadcast message sent by {sender_agent}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            return False
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents in the swarm"""
        status = {}
        
        for agent in self.enhanced_agents:
            agent_status = {
                'name': agent.agent_name,
                'role': agent.role,
                'current_tasks': agent.current_tasks,
                'max_tasks': agent.max_concurrent_tasks,
                'specializations': agent.specializations,
                'performance': agent.get_performance_metrics()
            }
            
            # Add hierarchical info if available
            if hasattr(agent, 'capability') and agent.capability:
                agent_status.update({
                    'proficiency': agent.capability.proficiency,
                    'availability': agent.capability.availability,
                    'current_load': agent.capability.current_load
                })
            
            status[agent.agent_name] = agent_status
        
        return status
    
    def shutdown(self):
        """Graceful shutdown of the swarm"""
        logger.info("Shutting down enhanced hierarchical swarm...")
        
        try:
            # Shutdown task executor
            self.task_executor.shutdown(wait=True)
            
            # Stop enhanced agents
            for agent in self.enhanced_agents:
                if hasattr(agent, 'hierarchical_agent') and agent.hierarchical_agent:
                    agent.hierarchical_agent.stop_listening()
            
            logger.info("Enhanced hierarchical swarm shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()


# Example usage function
def create_enhanced_swarm_example():
    """Create an example enhanced hierarchical swarm"""
    # Create enhanced agents with different roles and specializations
    director = EnhancedAgent(
        agent_name="Director",
        role="director",
        specializations=["planning", "coordination", "oversight"],
        system_prompt="You are a director agent responsible for coordinating and overseeing task execution.",
        max_concurrent_tasks=5
    )
    
    supervisor = EnhancedAgent(
        agent_name="Supervisor",
        role="supervisor",
        specializations=["management", "quality_control"],
        system_prompt="You are a supervisor agent responsible for managing workers and ensuring quality.",
        max_concurrent_tasks=4
    )
    
    workers = [
        EnhancedAgent(
            agent_name=f"Worker_{i}",
            role="worker",
            specializations=["data_analysis", "research"] if i % 2 == 0 else ["writing", "synthesis"],
            system_prompt=f"You are worker {i} specialized in your assigned tasks.",
            max_concurrent_tasks=3
        )
        for i in range(1, 4)
    ]
    
    # Create enhanced hierarchical swarm
    swarm = EnhancedHierarchicalSwarm(
        name="ExampleEnhancedSwarm",
        description="Example enhanced hierarchical swarm with improved communication",
        agents=[director, supervisor] + workers,
        director_agent=director,
        max_loops=2,
        cooperation_pattern=CooperationPattern.DELEGATION,
        enable_load_balancing=True,
        enable_auto_escalation=True,
        enable_collaboration=True,
        max_concurrent_tasks=15
    )
    
    return swarm