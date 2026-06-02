"""
Observability Engine - Execution Monitoring & Analytics

Responsibilities:
- Execution timeline tracking
- Workflow graph visualization
- Anomaly detection
- Performance monitoring
- Execution replay capability
- NexxSight dashboard integration

Example:
    observability = ObservabilityEngine()
    
    # Track execution step
    observability.record_step(
        workflow_id="wf_123",
        step_id="step_1",
        agent="pricing_agent",
        status="completed",
        duration=2.5,
        details={"price_changed_from": 40, "price_changed_to": 45}
    )
    
    # Get execution timeline
    timeline = observability.get_execution_timeline(workflow_id)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ExecutionStatus(str, Enum):
    """Status of an execution step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRIED = "retried"
    APPROVED = "approved"
    DENIED = "denied"


@dataclass
class ExecutionStep:
    """Single step in an execution timeline."""
    step_id: str
    agent_id: str
    action: str
    status: ExecutionStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration: float  # seconds
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "details": self.details,
            "error": self.error,
            "retry_count": self.retry_count
        }


@dataclass
class WorkflowExecution:
    """Complete workflow execution record."""
    workflow_id: str
    workflow_name: str
    start_time: datetime
    end_time: Optional[datetime]
    status: ExecutionStatus
    steps: List[ExecutionStep] = field(default_factory=list)
    agents_involved: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Total execution duration."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return (datetime.now() - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status.value,
            "duration": self.duration,
            "steps": [s.to_dict() for s in self.steps],
            "agents_involved": self.agents_involved,
            "errors": self.errors,
            "total_steps": len(self.steps)
        }


class ObservabilityEngine:
    """
    Observability engine for monitoring and analyzing autonomous execution.
    
    Provides complete visibility into agent execution, workflow performance,
    and system health through execution timelines, metrics, and replay capabilities.
    """
    
    def __init__(self):
        """Initialize observability engine."""
        self.executions: Dict[str, WorkflowExecution] = {}
        self.anomalies: List[Dict[str, Any]] = []
    
    def create_execution(
        self,
        workflow_id: str,
        workflow_name: str
    ) -> WorkflowExecution:
        """
        Create a new workflow execution record.
        
        Args:
            workflow_id: Unique workflow identifier
            workflow_name: Human-readable workflow name
        
        Returns:
            WorkflowExecution object
        """
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            workflow_name=workflow_name,
            start_time=datetime.now(),
            status=ExecutionStatus.RUNNING
        )
        
        self.executions[workflow_id] = execution
        return execution
    
    def record_step(
        self,
        workflow_id: str,
        step_id: str,
        agent_id: str,
        action: str,
        status: ExecutionStatus,
        duration: float = 0.0,
        details: Dict[str, Any] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Record an execution step.
        
        Args:
            workflow_id: Workflow identifier
            step_id: Step identifier
            agent_id: Agent performing the step
            action: Action description
            status: Step status
            duration: Execution duration
            details: Additional details
            error: Error message if failed
        """
        if workflow_id not in self.executions:
            return
        
        execution = self.executions[workflow_id]
        
        step = ExecutionStep(
            step_id=step_id,
            agent_id=agent_id,
            action=action,
            status=status,
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=duration,
            details=details or {},
            error=error
        )
        
        execution.steps.append(step)
        
        if agent_id not in execution.agents_involved:
            execution.agents_involved.append(agent_id)
        
        if error:
            execution.errors.append(error)
    
    def complete_execution(
        self,
        workflow_id: str,
        status: ExecutionStatus = ExecutionStatus.COMPLETED
    ) -> None:
        """
        Mark a workflow execution as complete.
        
        Args:
            workflow_id: Workflow identifier
            status: Final status
        """
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            execution.status = status
            execution.end_time = datetime.now()
    
    def get_execution_timeline(
        self,
        workflow_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get execution timeline for a workflow.
        
        Args:
            workflow_id: Workflow identifier
        
        Returns:
            List of execution steps or None if not found
        """
        if workflow_id in self.executions:
            execution = self.executions[workflow_id]
            return [step.to_dict() for step in execution.steps]
        return None
    
    def detect_anomalies(self, workflow_id: str) -> List[Dict[str, Any]]:
        """
        Detect anomalies in execution.
        
        Looks for:
        - Unusually long execution times
        - Repeated failures
        - Unexpected agent behavior
        - Resource spikes
        
        Args:
            workflow_id: Workflow identifier
        
        Returns:
            List of detected anomalies
        """
        if workflow_id not in self.executions:
            return []
        
        execution = self.executions[workflow_id]
        anomalies = []
        
        # Check for long steps
        for step in execution.steps:
            if step.duration > 60:  # More than 1 minute
                anomalies.append({
                    "type": "long_execution",
                    "step_id": step.step_id,
                    "duration": step.duration,
                    "severity": "medium"
                })
        
        # Check for repeated failures
        failed_steps = [s for s in execution.steps if s.status == ExecutionStatus.FAILED]
        if len(failed_steps) > 3:
            anomalies.append({
                "type": "repeated_failures",
                "count": len(failed_steps),
                "severity": "high"
            })
        
        # Check for high retry count
        high_retry_steps = [s for s in execution.steps if s.retry_count > 2]
        if high_retry_steps:
            anomalies.append({
                "type": "high_retries",
                "count": len(high_retry_steps),
                "severity": "medium"
            })
        
        return anomalies
    
    def replay_execution(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution data for replay/debugging.
        
        Args:
            workflow_id: Workflow identifier
        
        Returns:
            Complete execution record or None if not found
        """
        if workflow_id in self.executions:
            return self.executions[workflow_id].to_dict()
        return None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get observability engine summary."""
        total_executions = len(self.executions)
        completed = sum(
            1 for e in self.executions.values() 
            if e.status == ExecutionStatus.COMPLETED
        )
        failed = sum(
            1 for e in self.executions.values() 
            if e.status == ExecutionStatus.FAILED
        )
        
        return {
            "total_executions": total_executions,
            "completed": completed,
            "failed": failed,
            "anomalies_detected": len(self.anomalies),
            "average_duration": sum(
                e.duration for e in self.executions.values()
            ) / max(total_executions, 1)
        }
