"""
Hierarchical Cooperation System

This module provides advanced hierarchical cooperation protocols for multi-agent systems,
building on the enhanced communication infrastructure to enable sophisticated coordination
patterns, delegation chains, and cooperative task execution.
"""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union, Callable, Any, Tuple
from datetime import datetime, timedelta
import uuid

from swarms.communication.enhanced_communication import (
    CommunicationAgent, MessageBroker, EnhancedMessage, MessageType, 
    MessagePriority, CommunicationProtocol, MessageMetadata,
    AgentID, MessageID, get_global_broker
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="hierarchical_cooperation")


class HierarchicalRole(Enum):
    """Roles in hierarchical structure"""
    DIRECTOR = "director"
    SUPERVISOR = "supervisor"
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"


class CooperationPattern(Enum):
    """Cooperation patterns for task execution"""
    COMMAND_CONTROL = "command_control"
    DELEGATION = "delegation"
    COLLABORATION = "collaboration"
    CONSENSUS = "consensus"
    PIPELINE = "pipeline"
    BROADCAST_GATHER = "broadcast_gather"


class TaskStatus(Enum):
    """Status of tasks in the hierarchical system"""
    CREATED = "created"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    WAITING_DEPENDENCIES = "waiting_dependencies"
    COMPLETED = "completed"
    FAILED = "failed"
    DELEGATED = "delegated"
    ESCALATED = "escalated"


@dataclass
class HierarchicalTask:
    """Task representation in hierarchical system"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    assigned_agent: Optional[AgentID] = None
    requesting_agent: Optional[AgentID] = None
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.CREATED
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    result: Optional[Any] = None
    error_details: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_overdue(self) -> bool:
        """Check if task is overdue"""
        return self.deadline is not None and datetime.now() > self.deadline

    def can_start(self, completed_tasks: Set[str]) -> bool:
        """Check if task can start based on dependencies"""
        return all(dep_id in completed_tasks for dep_id in self.dependencies)


@dataclass
class AgentCapability:
    """Capability description for an agent"""
    name: str
    proficiency: float  # 0.0 to 1.0
    availability: float  # 0.0 to 1.0
    current_load: float  # 0.0 to 1.0
    specializations: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 3


@dataclass
class DelegationChain:
    """Represents a delegation chain from director to workers"""
    task_id: str
    chain: List[AgentID]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"


class HierarchicalCoordinator:
    """
    Coordinator for hierarchical cooperation patterns
    
    Manages task delegation, dependency resolution, escalation,
    and coordination across the hierarchical structure.
    """

    def __init__(self, broker: Optional[MessageBroker] = None):
        self.broker = broker or get_global_broker()
        
        # Hierarchical structure
        self._agents: Dict[AgentID, 'HierarchicalAgent'] = {}
        self._hierarchy: Dict[AgentID, List[AgentID]] = {}  # agent -> subordinates
        self._supervisors: Dict[AgentID, AgentID] = {}  # agent -> supervisor
        self._roles: Dict[AgentID, HierarchicalRole] = {}
        
        # Task management
        self._tasks: Dict[str, HierarchicalTask] = {}
        self._agent_tasks: Dict[AgentID, Set[str]] = defaultdict(set)
        self._completed_tasks: Set[str] = set()
        self._delegation_chains: Dict[str, DelegationChain] = {}
        
        # Capabilities and load balancing
        self._capabilities: Dict[AgentID, AgentCapability] = {}
        self._workload: Dict[AgentID, float] = defaultdict(float)
        
        # Cooperation patterns
        self._active_cooperations: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring and statistics
        self._stats = defaultdict(int)
        self._performance_history: List[Dict] = []
        
        # Control
        self._running = False
        self._workers: List[threading.Thread] = []
        
        # Start coordinator
        self.start()

    def register_agent(
        self, 
        agent: 'HierarchicalAgent',
        role: HierarchicalRole,
        supervisor: Optional[AgentID] = None,
        capabilities: Optional[AgentCapability] = None
    ):
        """Register an agent in the hierarchical structure"""
        agent_id = agent.agent_id
        
        # Register agent
        self._agents[agent_id] = agent
        self._roles[agent_id] = role
        
        # Set up hierarchical relationships
        if supervisor and supervisor in self._agents:
            self._supervisors[agent_id] = supervisor
            if supervisor not in self._hierarchy:
                self._hierarchy[supervisor] = []
            self._hierarchy[supervisor].append(agent_id)
        
        # Register capabilities
        if capabilities:
            self._capabilities[agent_id] = capabilities
        else:
            # Default capabilities
            self._capabilities[agent_id] = AgentCapability(
                name=f"{agent_id}_default",
                proficiency=0.7,
                availability=1.0,
                current_load=0.0
            )
        
        # Initialize workload tracking
        self._workload[agent_id] = 0.0
        
        logger.info(f"Registered agent {agent_id} with role {role.value}")

    def create_task(
        self,
        description: str,
        requesting_agent: Optional[AgentID] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        deadline: Optional[datetime] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new hierarchical task"""
        task = HierarchicalTask(
            description=description,
            requesting_agent=requesting_agent,
            priority=priority,
            deadline=deadline,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self._tasks[task.id] = task
        self._stats['tasks_created'] += 1
        
        logger.info(f"Created task {task.id}: {description}")
        
        # Auto-assign if possible
        self._try_assign_task(task.id)
        
        return task.id

    def assign_task(
        self,
        task_id: str,
        agent_id: AgentID,
        pattern: CooperationPattern = CooperationPattern.COMMAND_CONTROL
    ) -> bool:
        """Assign a task to an agent using specified cooperation pattern"""
        if task_id not in self._tasks or agent_id not in self._agents:
            return False
        
        task = self._tasks[task_id]
        if task.status != TaskStatus.CREATED:
            return False
        
        # Check if agent can handle the task
        if not self._can_agent_handle_task(agent_id, task):
            return False
        
        # Assign task
        task.assigned_agent = agent_id
        task.assigned_at = datetime.now()
        task.status = TaskStatus.ASSIGNED
        
        self._agent_tasks[agent_id].add(task_id)
        self._update_workload(agent_id)
        
        # Execute cooperation pattern
        self._execute_cooperation_pattern(task_id, pattern)
        
        self._stats['tasks_assigned'] += 1
        logger.info(f"Assigned task {task_id} to agent {agent_id} using {pattern.value}")
        
        return True

    def delegate_task(
        self,
        task_id: str,
        from_agent: AgentID,
        to_agent: AgentID,
        reason: str = "delegation"
    ) -> bool:
        """Delegate a task from one agent to another"""
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        # Validate delegation
        if task.assigned_agent != from_agent:
            logger.error(f"Task {task_id} not assigned to {from_agent}")
            return False
        
        if to_agent not in self._agents:
            logger.error(f"Target agent {to_agent} not found")
            return False
        
        # Check if target agent can handle the task
        if not self._can_agent_handle_task(to_agent, task):
            logger.warning(f"Agent {to_agent} cannot handle task {task_id}")
            return False
        
        # Update task assignment
        old_agent = task.assigned_agent
        task.assigned_agent = to_agent
        task.status = TaskStatus.DELEGATED
        task.metadata['delegation_reason'] = reason
        task.metadata['delegated_from'] = from_agent
        task.metadata['delegated_at'] = datetime.now().isoformat()
        
        # Update agent task tracking
        if old_agent:
            self._agent_tasks[old_agent].discard(task_id)
            self._update_workload(old_agent)
        
        self._agent_tasks[to_agent].add(task_id)
        self._update_workload(to_agent)
        
        # Create delegation chain
        if task_id not in self._delegation_chains:
            self._delegation_chains[task_id] = DelegationChain(
                task_id=task_id,
                chain=[from_agent, to_agent]
            )
        else:
            self._delegation_chains[task_id].chain.append(to_agent)
        
        # Notify agents
        self._notify_delegation(task_id, from_agent, to_agent)
        
        self._stats['tasks_delegated'] += 1
        logger.info(f"Delegated task {task_id} from {from_agent} to {to_agent}")
        
        return True

    def escalate_task(
        self,
        task_id: str,
        agent_id: AgentID,
        reason: str = "escalation"
    ) -> bool:
        """Escalate a task up the hierarchy"""
        if task_id not in self._tasks or agent_id not in self._supervisors:
            return False
        
        supervisor_id = self._supervisors[agent_id]
        task = self._tasks[task_id]
        
        # Update task
        task.status = TaskStatus.ESCALATED
        task.metadata['escalation_reason'] = reason
        task.metadata['escalated_from'] = agent_id
        task.metadata['escalated_at'] = datetime.now().isoformat()
        
        # Delegate to supervisor
        success = self.delegate_task(task_id, agent_id, supervisor_id, f"escalated: {reason}")
        
        if success:
            self._stats['tasks_escalated'] += 1
            logger.info(f"Escalated task {task_id} from {agent_id} to {supervisor_id}")
        
        return success

    def complete_task(
        self,
        task_id: str,
        agent_id: AgentID,
        result: Any = None
    ) -> bool:
        """Mark a task as completed"""
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        if task.assigned_agent != agent_id:
            logger.error(f"Task {task_id} not assigned to {agent_id}")
            return False
        
        # Complete task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        task.result = result
        
        # Update tracking
        self._completed_tasks.add(task_id)
        self._agent_tasks[agent_id].discard(task_id)
        self._update_workload(agent_id)
        
        # Check for dependent tasks
        self._check_dependent_tasks(task_id)
        
        # Notify completion in delegation chain
        if task_id in self._delegation_chains:
            self._notify_completion_chain(task_id, result)
        
        self._stats['tasks_completed'] += 1
        logger.info(f"Completed task {task_id} by agent {agent_id}")
        
        return True

    def fail_task(
        self,
        task_id: str,
        agent_id: AgentID,
        error: str,
        retry: bool = True
    ) -> bool:
        """Mark a task as failed and optionally retry"""
        if task_id not in self._tasks:
            return False
        
        task = self._tasks[task_id]
        
        if task.assigned_agent != agent_id:
            return False
        
        # Update task
        task.retry_count += 1
        task.error_details = error
        
        # Decide on retry or failure
        if retry and task.retry_count <= task.max_retries:
            task.status = TaskStatus.CREATED
            task.assigned_agent = None
            task.assigned_at = None
            
            # Remove from current agent
            self._agent_tasks[agent_id].discard(task_id)
            self._update_workload(agent_id)
            
            # Try to reassign
            self._try_assign_task(task_id)
            
            logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
            
        else:
            # Permanent failure
            task.status = TaskStatus.FAILED
            self._agent_tasks[agent_id].discard(task_id)
            self._update_workload(agent_id)
            
            # Try escalation
            if agent_id in self._supervisors:
                logger.info(f"Escalating failed task {task_id}")
                self.escalate_task(task_id, agent_id, f"task failed: {error}")
            
            self._stats['tasks_failed'] += 1
            logger.error(f"Failed task {task_id}: {error}")
        
        return True

    def _try_assign_task(self, task_id: str) -> bool:
        """Try to automatically assign a task to the best available agent"""
        task = self._tasks[task_id]
        
        # Check dependencies
        if not task.can_start(self._completed_tasks):
            task.status = TaskStatus.WAITING_DEPENDENCIES
            return False
        
        # Find best agent for the task
        best_agent = self._find_best_agent_for_task(task)
        
        if best_agent:
            return self.assign_task(task_id, best_agent)
        
        return False

    def _find_best_agent_for_task(self, task: HierarchicalTask) -> Optional[AgentID]:
        """Find the best agent to handle a task"""
        candidates = []
        
        for agent_id, capability in self._capabilities.items():
            if not self._can_agent_handle_task(agent_id, task):
                continue
            
            # Calculate score based on multiple factors
            score = self._calculate_agent_score(agent_id, task)
            candidates.append((score, agent_id))
        
        if candidates:
            # Sort by score (higher is better)
            candidates.sort(reverse=True)
            return candidates[0][1]
        
        return None

    def _calculate_agent_score(self, agent_id: AgentID, task: HierarchicalTask) -> float:
        """Calculate suitability score for agent-task pairing"""
        capability = self._capabilities[agent_id]
        
        # Base score from proficiency
        score = capability.proficiency
        
        # Adjust for availability
        score *= capability.availability
        
        # Penalize high current load
        score *= (1.0 - capability.current_load)
        
        # Boost for specialization match
        task_type = task.metadata.get('type', 'general')
        if task_type in capability.specializations:
            score *= 1.5
        
        # Priority adjustment
        if task.priority == MessagePriority.URGENT:
            score *= 1.3
        elif task.priority == MessagePriority.HIGH:
            score *= 1.1
        
        return score

    def _can_agent_handle_task(self, agent_id: AgentID, task: HierarchicalTask) -> bool:
        """Check if an agent can handle a specific task"""
        if agent_id not in self._capabilities:
            return False
        
        capability = self._capabilities[agent_id]
        
        # Check availability
        if capability.availability < 0.1:
            return False
        
        # Check current load
        if capability.current_load >= 1.0:
            return False
        
        # Check if agent has too many tasks
        current_tasks = len(self._agent_tasks[agent_id])
        if current_tasks >= capability.max_concurrent_tasks:
            return False
        
        # Check deadline constraints
        if task.deadline and datetime.now() > task.deadline:
            return False
        
        return True

    def _update_workload(self, agent_id: AgentID):
        """Update workload metrics for an agent"""
        if agent_id not in self._capabilities:
            return
        
        capability = self._capabilities[agent_id]
        current_tasks = len(self._agent_tasks[agent_id])
        
        # Calculate load as ratio of current tasks to max capacity
        capability.current_load = min(1.0, current_tasks / capability.max_concurrent_tasks)
        self._workload[agent_id] = capability.current_load

    def _execute_cooperation_pattern(self, task_id: str, pattern: CooperationPattern):
        """Execute specific cooperation pattern for task"""
        task = self._tasks[task_id]
        
        if pattern == CooperationPattern.COMMAND_CONTROL:
            self._execute_command_control(task_id)
        elif pattern == CooperationPattern.DELEGATION:
            self._execute_delegation_pattern(task_id)
        elif pattern == CooperationPattern.COLLABORATION:
            self._execute_collaboration_pattern(task_id)
        elif pattern == CooperationPattern.CONSENSUS:
            self._execute_consensus_pattern(task_id)
        elif pattern == CooperationPattern.PIPELINE:
            self._execute_pipeline_pattern(task_id)
        elif pattern == CooperationPattern.BROADCAST_GATHER:
            self._execute_broadcast_gather_pattern(task_id)

    def _execute_command_control(self, task_id: str):
        """Execute command and control pattern"""
        task = self._tasks[task_id]
        if not task.assigned_agent:
            return
        
        agent = self._agents[task.assigned_agent]
        
        # Send direct command
        message_content = {
            'task_id': task_id,
            'task_description': task.description,
            'priority': task.priority.value,
            'deadline': task.deadline.isoformat() if task.deadline else None,
            'metadata': task.metadata
        }
        
        agent.send_message(
            content=message_content,
            receiver_id=task.assigned_agent,
            message_type=MessageType.TASK,
            priority=task.priority,
            requires_ack=True
        )

    def _execute_delegation_pattern(self, task_id: str):
        """Execute delegation pattern with chain of responsibility"""
        task = self._tasks[task_id]
        if not task.assigned_agent:
            return
        
        # Create delegation context
        delegation_context = {
            'task_id': task_id,
            'delegation_allowed': True,
            'escalation_path': self._get_escalation_path(task.assigned_agent),
            'delegation_criteria': {
                'max_delegation_depth': 3,
                'required_capabilities': task.metadata.get('required_capabilities', [])
            }
        }
        
        task.metadata.update(delegation_context)
        self._execute_command_control(task_id)

    def _execute_collaboration_pattern(self, task_id: str):
        """Execute collaborative pattern with peer agents"""
        task = self._tasks[task_id]
        if not task.assigned_agent:
            return
        
        # Find collaborative peers
        peers = self._find_collaborative_peers(task.assigned_agent, task)
        
        if peers:
            # Notify peers about collaboration opportunity
            collaboration_id = str(uuid.uuid4())
            self._active_cooperations[collaboration_id] = {
                'type': 'collaboration',
                'task_id': task_id,
                'lead_agent': task.assigned_agent,
                'participants': peers,
                'created_at': datetime.now()
            }
            
            # Send collaboration invitations
            for peer_id in peers:
                self._send_collaboration_invitation(
                    collaboration_id, task_id, peer_id, task.assigned_agent
                )

    def _send_collaboration_invitation(
        self, 
        collaboration_id: str, 
        task_id: str, 
        peer_id: AgentID, 
        lead_agent: AgentID
    ):
        """Send collaboration invitation to peer agent"""
        if peer_id not in self._agents:
            return
        
        agent = self._agents[peer_id]
        invitation_content = {
            'collaboration_id': collaboration_id,
            'task_id': task_id,
            'lead_agent': lead_agent,
            'invitation_type': 'collaboration',
            'task_description': self._tasks[task_id].description
        }
        
        agent.send_message(
            content=invitation_content,
            receiver_id=peer_id,
            message_type=MessageType.COORDINATION,
            priority=MessagePriority.HIGH
        )

    def _find_collaborative_peers(self, agent_id: AgentID, task: HierarchicalTask) -> List[AgentID]:
        """Find suitable peer agents for collaboration"""
        peers = []
        
        # Look for agents with complementary capabilities
        required_skills = task.metadata.get('required_capabilities', [])
        
        for peer_id, capability in self._capabilities.items():
            if peer_id == agent_id:
                continue
            
            # Check if peer has relevant specializations
            if any(skill in capability.specializations for skill in required_skills):
                if capability.current_load < 0.8:  # Not too busy
                    peers.append(peer_id)
        
        return peers[:3]  # Limit to 3 collaborators

    def _execute_consensus_pattern(self, task_id: str):
        """Execute consensus pattern (placeholder implementation)"""
        # For now, fall back to collaboration pattern
        self._execute_collaboration_pattern(task_id)

    def _execute_pipeline_pattern(self, task_id: str):
        """Execute pipeline pattern (placeholder implementation)"""
        # For now, fall back to command control
        self._execute_command_control(task_id)

    def _execute_broadcast_gather_pattern(self, task_id: str):
        """Execute broadcast-gather pattern (placeholder implementation)"""
        # For now, fall back to collaboration pattern
        self._execute_collaboration_pattern(task_id)

    def _get_escalation_path(self, agent_id: AgentID) -> List[AgentID]:
        """Get escalation path for an agent"""
        path = []
        current_agent = agent_id
        
        while current_agent in self._supervisors:
            supervisor = self._supervisors[current_agent]
            path.append(supervisor)
            current_agent = supervisor
        
        return path

    def _check_dependent_tasks(self, completed_task_id: str):
        """Check and potentially start tasks that depended on this one"""
        for task_id, task in self._tasks.items():
            if (completed_task_id in task.dependencies and 
                task.status == TaskStatus.WAITING_DEPENDENCIES):
                
                if task.can_start(self._completed_tasks):
                    task.status = TaskStatus.CREATED
                    self._try_assign_task(task_id)

    def _notify_delegation(self, task_id: str, from_agent: AgentID, to_agent: AgentID):
        """Notify agents about task delegation"""
        task = self._tasks[task_id]
        
        # Notify the receiving agent
        if to_agent in self._agents:
            delegation_content = {
                'task_id': task_id,
                'task_description': task.description,
                'delegated_from': from_agent,
                'priority': task.priority.value,
                'delegation_reason': task.metadata.get('delegation_reason', 'delegation')
            }
            
            self._agents[to_agent].send_message(
                content=delegation_content,
                receiver_id=to_agent,
                message_type=MessageType.TASK,
                priority=task.priority
            )

    def _notify_completion_chain(self, task_id: str, result: Any):
        """Notify the delegation chain about task completion"""
        if task_id not in self._delegation_chains:
            return
        
        chain = self._delegation_chains[task_id]
        completion_content = {
            'task_id': task_id,
            'result': result,
            'completed_by': self._tasks[task_id].assigned_agent,
            'delegation_chain': chain.chain
        }
        
        # Notify all agents in the delegation chain
        for agent_id in chain.chain[:-1]:  # Exclude the final executor
            if agent_id in self._agents:
                self._agents[agent_id].send_message(
                    content=completion_content,
                    receiver_id=agent_id,
                    message_type=MessageType.RESPONSE,
                    priority=MessagePriority.HIGH
                )

    def get_hierarchy_status(self) -> Dict[str, Any]:
        """Get status of the hierarchical structure"""
        return {
            'total_agents': len(self._agents),
            'roles_distribution': {
                role.value: sum(1 for r in self._roles.values() if r == role)
                for role in HierarchicalRole
            },
            'hierarchy_depth': self._calculate_hierarchy_depth(),
            'active_tasks': len([t for t in self._tasks.values() if t.status != TaskStatus.COMPLETED]),
            'completed_tasks': len(self._completed_tasks),
            'delegation_chains': len(self._delegation_chains),
            'average_workload': sum(self._workload.values()) / len(self._workload) if self._workload else 0,
            'statistics': dict(self._stats)
        }

    def _calculate_hierarchy_depth(self) -> int:
        """Calculate the depth of the hierarchy"""
        max_depth = 0
        
        for agent_id in self._agents:
            depth = 0
            current = agent_id
            while current in self._supervisors:
                depth += 1
                current = self._supervisors[current]
            max_depth = max(max_depth, depth)
        
        return max_depth

    def start(self):
        """Start the hierarchical coordinator"""
        if self._running:
            return
        
        self._running = True
        
        # Start monitoring worker
        monitor_worker = threading.Thread(target=self._monitor_worker, daemon=True)
        monitor_worker.start()
        self._workers.append(monitor_worker)
        
        # Start task scheduler
        scheduler_worker = threading.Thread(target=self._scheduler_worker, daemon=True)
        scheduler_worker.start()
        self._workers.append(scheduler_worker)
        
        logger.info("Hierarchical coordinator started")

    def stop(self):
        """Stop the hierarchical coordinator"""
        self._running = False
        
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        logger.info("Hierarchical coordinator stopped")

    def _monitor_worker(self):
        """Worker thread for monitoring system health"""
        while self._running:
            try:
                # Check for overdue tasks
                current_time = datetime.now()
                for task_id, task in self._tasks.items():
                    if task.is_overdue() and task.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                        logger.warning(f"Task {task_id} is overdue")
                        
                        # Try escalation
                        if task.assigned_agent and task.assigned_agent in self._supervisors:
                            self.escalate_task(task_id, task.assigned_agent, "task overdue")
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(30.0)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Monitor worker error: {e}")
                time.sleep(60.0)

    def _scheduler_worker(self):
        """Worker thread for task scheduling"""
        while self._running:
            try:
                # Check for tasks waiting on dependencies
                for task_id, task in self._tasks.items():
                    if task.status == TaskStatus.WAITING_DEPENDENCIES:
                        if task.can_start(self._completed_tasks):
                            task.status = TaskStatus.CREATED
                            self._try_assign_task(task_id)
                
                # Rebalance workloads if needed
                self._rebalance_workloads()
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Scheduler worker error: {e}")
                time.sleep(30.0)

    def _update_performance_metrics(self):
        """Update performance metrics"""
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'active_tasks': len([t for t in self._tasks.values() if t.status != TaskStatus.COMPLETED]),
            'completed_tasks': len(self._completed_tasks),
            'failed_tasks': len([t for t in self._tasks.values() if t.status == TaskStatus.FAILED]),
            'average_workload': sum(self._workload.values()) / len(self._workload) if self._workload else 0,
            'delegation_efficiency': self._calculate_delegation_efficiency()
        }
        
        self._performance_history.append(metrics)
        
        # Keep only last 100 metrics
        if len(self._performance_history) > 100:
            self._performance_history.pop(0)

    def _calculate_delegation_efficiency(self) -> float:
        """Calculate delegation efficiency metric"""
        if not self._delegation_chains:
            return 1.0
        
        successful_delegations = sum(
            1 for task_id in self._delegation_chains 
            if self._tasks[task_id].status == TaskStatus.COMPLETED
        )
        
        return successful_delegations / len(self._delegation_chains)

    def _rebalance_workloads(self):
        """Rebalance workloads across agents if needed"""
        # Find overloaded and underloaded agents
        overloaded = []
        underloaded = []
        
        for agent_id, workload in self._workload.items():
            if workload > 0.9:
                overloaded.append(agent_id)
            elif workload < 0.3:
                underloaded.append(agent_id)
        
        # Try to delegate tasks from overloaded to underloaded agents
        for overloaded_agent in overloaded:
            if not underloaded:
                break
            
            # Find tasks that can be delegated
            delegatable_tasks = [
                task_id for task_id in self._agent_tasks[overloaded_agent]
                if self._tasks[task_id].status == TaskStatus.ASSIGNED
            ]
            
            for task_id in delegatable_tasks[:1]:  # Delegate one task at a time
                target_agent = underloaded[0]
                if self._can_agent_handle_task(target_agent, self._tasks[task_id]):
                    self.delegate_task(task_id, overloaded_agent, target_agent, "load balancing")
                    underloaded.pop(0)
                    break


class HierarchicalAgent(CommunicationAgent):
    """
    Enhanced agent with hierarchical cooperation capabilities
    """
    
    def __init__(
        self, 
        agent_id: AgentID, 
        role: HierarchicalRole = HierarchicalRole.WORKER,
        broker: Optional[MessageBroker] = None,
        coordinator: Optional[HierarchicalCoordinator] = None
    ):
        super().__init__(agent_id, broker)
        
        self.role = role
        self.coordinator = coordinator
        self._current_tasks: Dict[str, HierarchicalTask] = {}
        self._collaboration_sessions: Dict[str, Dict] = {}
        
        # Register hierarchical message handlers
        self.register_message_handler(MessageType.TASK, self._handle_task_message)
        self.register_message_handler(MessageType.COORDINATION, self._handle_coordination_message)
        self.register_message_handler(MessageType.RESPONSE, self._handle_response_message)
        
        # Start listening
        self.start_listening()
        
        logger.info(f"Hierarchical agent {agent_id} created with role {role.value}")

    def _handle_task_message(self, message: EnhancedMessage):
        """Handle incoming task messages"""
        try:
            content = message.content
            if isinstance(content, dict) and 'task_id' in content:
                task_id = content['task_id']
                
                # Create local task representation
                task = HierarchicalTask(
                    id=task_id,
                    description=content.get('task_description', ''),
                    assigned_agent=self.agent_id,
                    status=TaskStatus.IN_PROGRESS,
                    priority=MessagePriority(content.get('priority', 2)),
                    metadata=content.get('metadata', {})
                )
                
                self._current_tasks[task_id] = task
                
                # Execute task
                result = self.execute_task(task)
                
                # Report completion
                if self.coordinator:
                    if result.get('success', False):
                        self.coordinator.complete_task(task_id, self.agent_id, result)
                    else:
                        self.coordinator.fail_task(
                            task_id, self.agent_id, result.get('error', 'Unknown error')
                        )
                
                # Clean up
                self._current_tasks.pop(task_id, None)
                
        except Exception as e:
            logger.error(f"Error handling task message: {e}")

    def _handle_coordination_message(self, message: EnhancedMessage):
        """Handle coordination messages (collaboration invitations, etc.)"""
        try:
            content = message.content
            if isinstance(content, dict):
                if content.get('invitation_type') == 'collaboration':
                    self._handle_collaboration_invitation(content, message.sender_id)
                    
        except Exception as e:
            logger.error(f"Error handling coordination message: {e}")

    def _handle_collaboration_invitation(self, content: Dict, sender_id: AgentID):
        """Handle collaboration invitation"""
        collaboration_id = content.get('collaboration_id')
        task_id = content.get('task_id')
        
        if collaboration_id and task_id:
            # Decide whether to accept collaboration
            accept = self._should_accept_collaboration(content)
            
            if accept:
                self._collaboration_sessions[collaboration_id] = {
                    'task_id': task_id,
                    'lead_agent': content.get('lead_agent'),
                    'participants': [],
                    'status': 'active'
                }
                
                # Send acceptance
                response = {
                    'collaboration_id': collaboration_id,
                    'response': 'accept',
                    'capabilities': self._get_relevant_capabilities(task_id)
                }
                
                self.send_message(
                    content=response,
                    receiver_id=sender_id,
                    message_type=MessageType.COORDINATION,
                    priority=MessagePriority.HIGH
                )
                
                logger.info(f"Accepted collaboration {collaboration_id}")
            else:
                # Send decline
                response = {
                    'collaboration_id': collaboration_id,
                    'response': 'decline',
                    'reason': 'insufficient capacity'
                }
                
                self.send_message(
                    content=response,
                    receiver_id=sender_id,
                    message_type=MessageType.COORDINATION,
                    priority=MessagePriority.NORMAL
                )

    def _should_accept_collaboration(self, invitation: Dict) -> bool:
        """Decide whether to accept a collaboration invitation"""
        # Simple heuristic: accept if not too busy
        current_load = len(self._current_tasks)
        max_capacity = 3  # Could be configurable
        
        return current_load < max_capacity

    def _get_relevant_capabilities(self, task_id: str) -> List[str]:
        """Get capabilities relevant to a specific task"""
        # This would be implemented based on the agent's actual capabilities
        return ["analysis", "research", "coordination"]

    def _handle_response_message(self, message: EnhancedMessage):
        """Handle response messages from other agents"""
        try:
            content = message.content
            if isinstance(content, dict):
                if 'collaboration_id' in content:
                    self._handle_collaboration_response(content)
                elif 'task_id' in content:
                    self._handle_task_response(content)
                    
        except Exception as e:
            logger.error(f"Error handling response message: {e}")

    def _handle_collaboration_response(self, content: Dict):
        """Handle collaboration response"""
        collaboration_id = content.get('collaboration_id')
        response = content.get('response')
        
        if collaboration_id in self._collaboration_sessions:
            session = self._collaboration_sessions[collaboration_id]
            
            if response == 'accept':
                # Add participant to collaboration
                participant_id = content.get('participant_id')
                if participant_id:
                    session['participants'].append(participant_id)
                
                logger.info(f"Participant {participant_id} joined collaboration {collaboration_id}")

    def _handle_task_response(self, content: Dict):
        """Handle task response from subordinates"""
        task_id = content.get('task_id')
        result = content.get('result')
        
        logger.info(f"Received task response for {task_id}: {result}")

    def delegate_task_to_subordinate(
        self, 
        task: HierarchicalTask, 
        subordinate_id: AgentID,
        reason: str = "delegation"
    ) -> bool:
        """Delegate a task to a subordinate agent"""
        if not self.coordinator:
            return False
        
        return self.coordinator.delegate_task(
            task.id, self.agent_id, subordinate_id, reason
        )

    def escalate_task_to_supervisor(
        self, 
        task: HierarchicalTask, 
        reason: str = "escalation"
    ) -> bool:
        """Escalate a task to supervisor"""
        if not self.coordinator:
            return False
        
        return self.coordinator.escalate_task(task.id, self.agent_id, reason)

    def request_collaboration(
        self, 
        task: HierarchicalTask, 
        peer_agents: List[AgentID]
    ) -> str:
        """Request collaboration with peer agents"""
        collaboration_id = str(uuid.uuid4())
        
        # Send collaboration requests
        for peer_id in peer_agents:
            invitation = {
                'collaboration_id': collaboration_id,
                'task_id': task.id,
                'invitation_type': 'collaboration',
                'task_description': task.description,
                'lead_agent': self.agent_id
            }
            
            self.send_message(
                content=invitation,
                receiver_id=peer_id,
                message_type=MessageType.COORDINATION,
                priority=MessagePriority.HIGH
            )
        
        # Track collaboration session
        self._collaboration_sessions[collaboration_id] = {
            'task_id': task.id,
            'lead_agent': self.agent_id,
            'invited_agents': peer_agents,
            'participants': [],
            'status': 'pending'
        }
        
        return collaboration_id

    @abstractmethod
    def execute_task(self, task: HierarchicalTask) -> Dict[str, Any]:
        """Execute a task - to be implemented by subclasses"""
        pass

    def process_task_message(self, message: EnhancedMessage):
        """Process task message - delegated to _handle_task_message"""
        self._handle_task_message(message)

    def process_response_message(self, message: EnhancedMessage):
        """Process response message - delegated to _handle_response_message"""
        self._handle_response_message(message)


# Global coordinator instance
_global_coordinator = None

def get_global_coordinator() -> HierarchicalCoordinator:
    """Get or create global hierarchical coordinator instance"""
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = HierarchicalCoordinator()
    return _global_coordinator

def shutdown_global_coordinator():
    """Shutdown global hierarchical coordinator"""
    global _global_coordinator
    if _global_coordinator:
        _global_coordinator.stop()
        _global_coordinator = None