"""
Enhanced HierarchicalSwarm with advanced communication and coordination capabilities

This module provides an improved hierarchical swarm implementation that includes:
- Enhanced communication protocols with multi-directional message passing
- Dynamic role assignment and specialization
- Advanced coordination mechanisms
- Performance monitoring and optimization
- Error handling and recovery
- Adaptive planning and learning
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
from enum import Enum

from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.communication import (
    CommunicationManager,
    Message,
    MessageType,
    MessagePriority,
    MessageStatus
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.history_output_formatter import history_output_formatter
from swarms.utils.output_types import OutputType
from swarms.tools.base_tool import BaseTool
from swarms.prompts.hiearchical_system_prompt import HIEARCHICAL_SWARM_SYSTEM_PROMPT

logger = initialize_logger(log_folder="enhanced_hierarchical_swarm")


class AgentRole(Enum):
    """Roles that agents can take in the hierarchy"""
    DIRECTOR = "director"
    MIDDLE_MANAGER = "middle_manager"
    SPECIALIST = "specialist"
    COORDINATOR = "coordinator"
    ANALYST = "analyst"
    EXECUTOR = "executor"


class TaskComplexity(Enum):
    """Complexity levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentCapability:
    """Represents an agent's capability in a specific domain"""
    domain: str
    skill_level: float  # 0.0 to 1.0
    experience_count: int = 0
    success_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def update_performance(self, success: bool, task_complexity: TaskComplexity):
        """Update capability based on task performance"""
        self.experience_count += 1
        if success:
            # Increase skill level based on task complexity
            improvement = task_complexity.value * 0.01
            self.skill_level = min(1.0, self.skill_level + improvement)
        else:
            # Slight decrease for failures
            self.skill_level = max(0.0, self.skill_level - 0.005)
        
        # Update success rate
        current_successes = self.success_rate * (self.experience_count - 1)
        if success:
            current_successes += 1
        self.success_rate = current_successes / self.experience_count
        self.last_updated = time.time()


@dataclass
class EnhancedTask:
    """Enhanced task representation with metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    priority: MessagePriority = MessagePriority.MEDIUM
    required_capabilities: List[str] = field(default_factory=list)
    estimated_duration: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    feedback: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            'id': self.id,
            'content': self.content,
            'complexity': self.complexity.value,
            'priority': self.priority.value,
            'required_capabilities': self.required_capabilities,
            'estimated_duration': self.estimated_duration,
            'dependencies': self.dependencies,
            'assigned_agent': self.assigned_agent,
            'status': self.status,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'result': self.result,
            'feedback': self.feedback
        }


class DynamicRoleManager:
    """Manages dynamic role assignment and agent specialization"""
    
    def __init__(self):
        self.agent_capabilities: Dict[str, Dict[str, AgentCapability]] = {}
        self.role_assignments: Dict[str, AgentRole] = {}
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.specialization_thresholds = {
            AgentRole.SPECIALIST: 0.8,
            AgentRole.COORDINATOR: 0.7,
            AgentRole.MIDDLE_MANAGER: 0.6
        }
        
    def register_agent(self, agent_id: str, initial_capabilities: Optional[Dict[str, float]] = None):
        """Register agent with initial capabilities"""
        self.agent_capabilities[agent_id] = {}
        self.role_assignments[agent_id] = AgentRole.EXECUTOR
        self.performance_history[agent_id] = []
        
        if initial_capabilities:
            for domain, skill_level in initial_capabilities.items():
                self.agent_capabilities[agent_id][domain] = AgentCapability(
                    domain=domain,
                    skill_level=skill_level
                )
    
    def update_agent_performance(self, 
                               agent_id: str, 
                               domain: str, 
                               success: bool, 
                               task_complexity: TaskComplexity):
        """Update agent performance in a domain"""
        if agent_id not in self.agent_capabilities:
            self.register_agent(agent_id)
        
        capabilities = self.agent_capabilities[agent_id]
        if domain not in capabilities:
            capabilities[domain] = AgentCapability(domain=domain, skill_level=0.5)
        
        capabilities[domain].update_performance(success, task_complexity)
        
        # Record performance history
        self.performance_history[agent_id].append({
            'timestamp': time.time(),
            'domain': domain,
            'success': success,
            'task_complexity': task_complexity.value,
            'new_skill_level': capabilities[domain].skill_level
        })
        
        # Update role based on performance
        self._update_agent_role(agent_id)
    
    def _update_agent_role(self, agent_id: str):
        """Update agent role based on performance"""
        capabilities = self.agent_capabilities[agent_id]
        
        # Calculate average skill level
        if not capabilities:
            return
            
        avg_skill = sum(cap.skill_level for cap in capabilities.values()) / len(capabilities)
        
        # Determine appropriate role
        new_role = AgentRole.EXECUTOR
        for role, threshold in self.specialization_thresholds.items():
            if avg_skill >= threshold:
                new_role = role
                break
        
        # Update role if changed
        if self.role_assignments[agent_id] != new_role:
            old_role = self.role_assignments[agent_id]
            self.role_assignments[agent_id] = new_role
            logger.info(f"Agent {agent_id} role updated from {old_role} to {new_role}")
    
    def get_best_agent_for_task(self, 
                              required_capabilities: List[str],
                              available_agents: List[str]) -> Optional[str]:
        """Find the best agent for a task based on capabilities"""
        best_agent = None
        best_score = -1
        
        for agent_id in available_agents:
            if agent_id not in self.agent_capabilities:
                continue
                
            capabilities = self.agent_capabilities[agent_id]
            
            # Calculate match score
            score = 0
            for capability in required_capabilities:
                if capability in capabilities:
                    cap = capabilities[capability]
                    # Weight by skill level and success rate
                    score += cap.skill_level * cap.success_rate
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent
    
    def get_agent_capabilities(self, agent_id: str) -> Dict[str, AgentCapability]:
        """Get agent capabilities"""
        return self.agent_capabilities.get(agent_id, {})
    
    def get_agent_role(self, agent_id: str) -> AgentRole:
        """Get agent role"""
        return self.role_assignments.get(agent_id, AgentRole.EXECUTOR)


class TaskScheduler:
    """Intelligent task scheduling and coordination"""
    
    def __init__(self, role_manager: DynamicRoleManager):
        self.role_manager = role_manager
        self.task_queue: List[EnhancedTask] = []
        self.active_tasks: Dict[str, EnhancedTask] = {}
        self.completed_tasks: Dict[str, EnhancedTask] = {}
        self.agent_workload: Dict[str, int] = {}
        self.max_concurrent_tasks = 10
        
    def add_task(self, task: EnhancedTask):
        """Add task to scheduler"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: (t.priority.value, t.complexity.value))
        
    def schedule_tasks(self, available_agents: List[str]) -> Dict[str, List[EnhancedTask]]:
        """Schedule tasks to available agents"""
        scheduled_tasks = {}
        
        # Initialize agent workload
        for agent_id in available_agents:
            if agent_id not in self.agent_workload:
                self.agent_workload[agent_id] = 0
        
        # Schedule tasks
        remaining_tasks = []
        for task in self.task_queue:
            if len(self.active_tasks) >= self.max_concurrent_tasks:
                remaining_tasks.append(task)
                continue
                
            # Check dependencies
            if not self._dependencies_met(task):
                remaining_tasks.append(task)
                continue
            
            # Find best agent for task
            best_agent = self.role_manager.get_best_agent_for_task(
                task.required_capabilities, 
                available_agents
            )
            
            if best_agent and self.agent_workload[best_agent] < 3:  # Max 3 concurrent tasks per agent
                if best_agent not in scheduled_tasks:
                    scheduled_tasks[best_agent] = []
                
                scheduled_tasks[best_agent].append(task)
                self.active_tasks[task.id] = task
                self.agent_workload[best_agent] += 1
                task.assigned_agent = best_agent
                task.status = "assigned"
                task.started_at = time.time()
            else:
                remaining_tasks.append(task)
        
        self.task_queue = remaining_tasks
        return scheduled_tasks
    
    def _dependencies_met(self, task: EnhancedTask) -> bool:
        """Check if task dependencies are met"""
        for dep_id in task.dependencies:
            if dep_id not in self.completed_tasks:
                return False
        return True
    
    def mark_task_completed(self, task_id: str, result: Any, success: bool):
        """Mark task as completed"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            task.status = "completed" if success else "failed"
            task.completed_at = time.time()
            task.result = result
            self.completed_tasks[task_id] = task
            
            # Update agent workload
            if task.assigned_agent:
                self.agent_workload[task.assigned_agent] -= 1
            
            # Update agent performance
            if task.assigned_agent and task.required_capabilities:
                for capability in task.required_capabilities:
                    self.role_manager.update_agent_performance(
                        task.assigned_agent, 
                        capability, 
                        success, 
                        task.complexity
                    )
    
    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get task status"""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id].status
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id].status
        return None


class EnhancedHierarchicalSwarm(BaseSwarm):
    """Enhanced hierarchical swarm with advanced communication and coordination"""
    
    def __init__(self,
                 name: str = "EnhancedHierarchicalSwarm",
                 description: str = "Advanced hierarchical swarm with enhanced communication",
                 director: Optional[Union[Agent, Callable, Any]] = None,
                 agents: List[Union[Agent, Callable, Any]] = None,
                 max_loops: int = 1,
                 output_type: OutputType = "dict-all-except-first",
                 director_model_name: str = "gpt-4o-mini",
                 verbose: bool = False,
                 enable_parallel_execution: bool = True,
                 max_concurrent_tasks: int = 10,
                 auto_optimize: bool = True,
                 **kwargs):
        
        super().__init__(name=name, description=description, agents=agents or [])
        
        self.director = director
        self.max_loops = max_loops
        self.output_type = output_type
        self.director_model_name = director_model_name
        self.verbose = verbose
        self.enable_parallel_execution = enable_parallel_execution
        self.max_concurrent_tasks = max_concurrent_tasks
        self.auto_optimize = auto_optimize
        self.agents = agents or []
        
        # Initialize enhanced components
        self.communication_manager = CommunicationManager()
        self.role_manager = DynamicRoleManager()
        self.task_scheduler = TaskScheduler(self.role_manager)
        self.conversation = Conversation(time_enabled=True)
        
        # Performance tracking
        self.execution_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0,
            'agent_utilization': {}
        }
        
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # Initialize the swarm
        self.init_swarm()
    
    def init_swarm(self):
        """Initialize the enhanced swarm"""
        if self.verbose:
            logger.info(f"🚀 Initializing EnhancedHierarchicalSwarm: {self.name}")
        
        # Start communication manager
        self.communication_manager.start()
        
        # Register agents
        self._register_agents()
        
        # Setup director
        self._setup_director()
        
        # Setup communication channels
        self._setup_communication_channels()
        
        if self.verbose:
            logger.success(f"✅ EnhancedHierarchicalSwarm initialized: {self.name}")
    
    def _register_agents(self):
        """Register all agents with role manager"""
        for agent in self.agents:
            agent_id = getattr(agent, 'agent_name', str(id(agent)))
            
            # Extract initial capabilities from agent description
            initial_capabilities = self._extract_capabilities_from_agent(agent)
            self.role_manager.register_agent(agent_id, initial_capabilities)
            
            # Create communication channel for agent
            self.communication_manager.create_agent_channel(agent_id)
    
    def _extract_capabilities_from_agent(self, agent) -> Dict[str, float]:
        """Extract capabilities from agent description"""
        # Simple heuristic - could be enhanced with NLP
        capabilities = {}
        
        description = getattr(agent, 'agent_description', '').lower()
        system_prompt = getattr(agent, 'system_prompt', '').lower()
        
        combined_text = f"{description} {system_prompt}"
        
        # Define capability keywords and their weights
        capability_keywords = {
            'analysis': ['analysis', 'analyze', 'analytical'],
            'writing': ['writing', 'write', 'content', 'documentation'],
            'research': ['research', 'investigate', 'study'],
            'coding': ['code', 'programming', 'development', 'software'],
            'planning': ['plan', 'planning', 'strategy', 'roadmap'],
            'communication': ['communication', 'presentation', 'report']
        }
        
        for capability, keywords in capability_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    score += 0.2
            
            if score > 0:
                capabilities[capability] = min(1.0, score)
        
        # Default capabilities if none found
        if not capabilities:
            capabilities = {'general': 0.5}
        
        return capabilities
    
    def _setup_director(self):
        """Setup director agent"""
        if not self.director:
            self.director = Agent(
                agent_name="Director",
                agent_description="Director agent that coordinates and manages the swarm",
                model_name=self.director_model_name,
                max_loops=1,
                system_prompt=HIEARCHICAL_SWARM_SYSTEM_PROMPT
            )
        
        # Register director with role manager
        director_id = getattr(self.director, 'agent_name', 'Director')
        self.role_manager.register_agent(director_id, {'coordination': 0.9, 'planning': 0.9})
        self.role_manager.role_assignments[director_id] = AgentRole.DIRECTOR
    
    def _setup_communication_channels(self):
        """Setup communication channels between agents"""
        director_id = getattr(self.director, 'agent_name', 'Director')
        
        # Create channels for each agent to communicate with director
        for agent in self.agents:
            agent_id = getattr(agent, 'agent_name', str(id(agent)))
            
            # Director-Agent channel
            self.communication_manager.router.create_channel(
                f"director_{agent_id}",
                [director_id, agent_id],
                "hierarchical"
            )
        
        # Create peer-to-peer channels for coordination
        if self.enable_parallel_execution:
            for i, agent1 in enumerate(self.agents):
                for agent2 in self.agents[i+1:]:
                    agent1_id = getattr(agent1, 'agent_name', str(id(agent1)))
                    agent2_id = getattr(agent2, 'agent_name', str(id(agent2)))
                    
                    self.communication_manager.router.create_channel(
                        f"peer_{agent1_id}_{agent2_id}",
                        [agent1_id, agent2_id],
                        "peer"
                    )
    
    def run(self, task: str, img: str = None, *args, **kwargs):
        """Execute the enhanced hierarchical swarm"""
        try:
            start_time = time.time()
            
            if self.verbose:
                logger.info(f"🚀 Starting enhanced swarm execution: {self.name}")
            
            # Create conversation for this execution
            conversation_id = f"exec_{uuid.uuid4()}"
            
            # Parse task into enhanced tasks
            enhanced_tasks = self._parse_task_into_subtasks(task)
            
            # Add tasks to scheduler
            for enhanced_task in enhanced_tasks:
                self.task_scheduler.add_task(enhanced_task)
            
            # Execute tasks
            results = self._execute_tasks_with_coordination(conversation_id, img)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_execution_metrics(execution_time, len(enhanced_tasks))
            
            if self.verbose:
                logger.success(f"✅ Enhanced swarm execution completed in {execution_time:.2f}s")
            
            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type
            )
            
        except Exception as e:
            logger.error(f"❌ Enhanced swarm execution failed: {str(e)}")
            raise
    
    def _parse_task_into_subtasks(self, task: str) -> List[EnhancedTask]:
        """Parse main task into enhanced subtasks"""
        # Use director to break down the task
        if not self.director:
            # Fallback: create single task if no director
            return [EnhancedTask(
                content=task,
                complexity=TaskComplexity.MEDIUM,
                priority=MessagePriority.MEDIUM,
                required_capabilities=['general']
            )]
            
        director_response = self.director.run(
            task=f"Break down this task into specific subtasks with required capabilities: {task}"
        )
        
        # Parse director response into enhanced tasks
        enhanced_tasks = []
        
        # Simple parsing - could be enhanced with structured output
        if isinstance(director_response, list):
            for item in director_response:
                if isinstance(item, dict) and 'content' in item:
                    content = item['content']
                    if isinstance(content, str):
                        enhanced_task = EnhancedTask(
                            content=content,
                            complexity=TaskComplexity.MEDIUM,
                            priority=MessagePriority.MEDIUM,
                            required_capabilities=self._extract_required_capabilities(content)
                        )
                        enhanced_tasks.append(enhanced_task)
        else:
            # Fallback: create single task
            enhanced_task = EnhancedTask(
                content=task,
                complexity=TaskComplexity.MEDIUM,
                priority=MessagePriority.MEDIUM,
                required_capabilities=['general']
            )
            enhanced_tasks.append(enhanced_task)
        
        return enhanced_tasks
    
    def _extract_required_capabilities(self, task_content: str) -> List[str]:
        """Extract required capabilities from task content"""
        capabilities = []
        content_lower = task_content.lower()
        
        capability_keywords = {
            'analysis': ['analyze', 'analysis', 'evaluate', 'assess'],
            'writing': ['write', 'draft', 'create', 'document'],
            'research': ['research', 'investigate', 'find', 'study'],
            'coding': ['code', 'program', 'develop', 'implement'],
            'planning': ['plan', 'design', 'strategy', 'organize'],
            'communication': ['present', 'report', 'communicate', 'explain']
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                capabilities.append(capability)
        
        return capabilities or ['general']
    
    def _execute_tasks_with_coordination(self, conversation_id: str, img: str = None) -> List[Any]:
        """Execute tasks with coordination and communication"""
        results = []
        
        # Get available agents
        available_agents = []
        for agent in self.agents:
            agent_id = getattr(agent, 'agent_name', None)
            if agent_id is None:
                agent_id = str(id(agent))
            available_agents.append(agent_id)
        
        # Execute tasks in batches
        while self.task_scheduler.task_queue or self.task_scheduler.active_tasks:
            # Schedule next batch of tasks
            scheduled_tasks = self.task_scheduler.schedule_tasks(available_agents)
            
            if not scheduled_tasks and self.task_scheduler.active_tasks:
                # Wait for active tasks to complete
                time.sleep(0.1)
                continue
            
            # Execute scheduled tasks
            if self.enable_parallel_execution:
                batch_results = self._execute_tasks_parallel(scheduled_tasks, conversation_id, img)
            else:
                batch_results = self._execute_tasks_sequential(scheduled_tasks, conversation_id, img)
            
            results.extend(batch_results)
        
        return results
    
    def _execute_tasks_parallel(self, scheduled_tasks: Dict[str, List[EnhancedTask]], 
                               conversation_id: str, img: str = None) -> List[Any]:
        """Execute tasks in parallel"""
        futures = []
        
        for agent_id, tasks in scheduled_tasks.items():
            for task in tasks:
                future = self.executor.submit(
                    self._execute_single_task,
                    agent_id, task, conversation_id, img
                )
                futures.append((future, task))
        
        results = []
        for future, task in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results.append(result)
                self.task_scheduler.mark_task_completed(task.id, result, True)
            except Exception as e:
                logger.error(f"Task {task.id} failed: {str(e)}")
                self.task_scheduler.mark_task_completed(task.id, str(e), False)
        
        return results
    
    def _execute_tasks_sequential(self, scheduled_tasks: Dict[str, List[EnhancedTask]], 
                                 conversation_id: str, img: str = None) -> List[Any]:
        """Execute tasks sequentially"""
        results = []
        
        for agent_id, tasks in scheduled_tasks.items():
            for task in tasks:
                try:
                    result = self._execute_single_task(agent_id, task, conversation_id, img)
                    results.append(result)
                    self.task_scheduler.mark_task_completed(task.id, result, True)
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {str(e)}")
                    self.task_scheduler.mark_task_completed(task.id, str(e), False)
        
        return results
    
    def _execute_single_task(self, agent_id: str, task: EnhancedTask, 
                            conversation_id: str, img: str = None) -> Any:
        """Execute a single task"""
        # Find agent
        agent = None
        for a in self.agents:
            if getattr(a, 'agent_name', str(id(a))) == agent_id:
                agent = a
                break
        
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create task message
        task_message = Message(
            sender_id="Director",
            receiver_id=agent_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            priority=task.priority,
            content={
                'task': task.content,
                'task_id': task.id,
                'required_capabilities': task.required_capabilities
            },
            conversation_id=conversation_id
        )
        
        # Send task message
        self.communication_manager.send_message(task_message)
        
        # Execute task
        result = agent.run(task=task.content, img=img)
        
        # Record in conversation
        self.conversation.add(role=agent_id, content=result)
        
        # Send completion message
        completion_message = Message(
            sender_id=agent_id,
            receiver_id="Director",
            message_type=MessageType.TASK_COMPLETION,
            priority=task.priority,
            content={
                'task_id': task.id,
                'result': result,
                'success': True
            },
            conversation_id=conversation_id
        )
        
        self.communication_manager.send_message(completion_message)
        
        return result
    
    def _update_execution_metrics(self, execution_time: float, task_count: int):
        """Update execution metrics"""
        self.execution_metrics['total_tasks'] += task_count
        self.execution_metrics['completed_tasks'] += len(self.task_scheduler.completed_tasks)
        self.execution_metrics['failed_tasks'] = self.execution_metrics['total_tasks'] - self.execution_metrics['completed_tasks']
        
        # Update average execution time
        current_avg = self.execution_metrics['avg_execution_time']
        self.execution_metrics['avg_execution_time'] = (current_avg + execution_time) / 2
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {
            'execution_metrics': self.execution_metrics,
            'communication_stats': self.communication_manager.get_channel_statistics(),
            'agent_performance': {}
        }
        
        # Add agent performance
        for agent in self.agents:
            agent_id = getattr(agent, 'agent_name', str(id(agent)))
            agent_perf = self.communication_manager.get_agent_performance(agent_id)
            agent_capabilities = self.role_manager.get_agent_capabilities(agent_id)
            agent_role = self.role_manager.get_agent_role(agent_id)
            
            metrics['agent_performance'][agent_id] = {
                'performance_metrics': agent_perf,
                'capabilities': {cap: {'skill_level': data.skill_level, 'success_rate': data.success_rate} 
                               for cap, data in agent_capabilities.items()},
                'role': agent_role.value
            }
        
        return metrics
    
    def optimize_performance(self):
        """Optimize swarm performance based on metrics"""
        if not self.auto_optimize:
            return
        
        # Analyze performance and adjust parameters
        metrics = self.get_performance_metrics()
        
        # Adjust concurrent task limits based on performance
        success_rate = metrics['execution_metrics']['completed_tasks'] / max(1, metrics['execution_metrics']['total_tasks'])
        
        if success_rate < 0.7:  # Low success rate
            self.max_concurrent_tasks = max(1, self.max_concurrent_tasks - 1)
        elif success_rate > 0.9:  # High success rate
            self.max_concurrent_tasks = min(20, self.max_concurrent_tasks + 1)
        
        if self.verbose:
            logger.info(f"Performance optimization: concurrent tasks adjusted to {self.max_concurrent_tasks}")
    
    def shutdown(self):
        """Shutdown the swarm"""
        if self.verbose:
            logger.info("🛑 Shutting down EnhancedHierarchicalSwarm")
        
        self.communication_manager.stop()
        self.executor.shutdown(wait=True)
        
        if self.verbose:
            logger.success("✅ EnhancedHierarchicalSwarm shutdown complete")