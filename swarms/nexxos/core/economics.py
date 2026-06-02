"""
Economics Engine - Cost Tracking & ROI Analysis

Responsibilities:
- Token usage tracking
- Compute cost calculation
- Revenue impact analysis
- Workflow profitability
- Agent efficiency metrics
- Automation ROI calculation

Example:
    economics = EconomicsEngine()
    
    # Track workflow execution
    workflow_cost = economics.calculate_workflow_cost(
        tokens_used=1500,
        compute_time=2.5,
        api_calls=3
    )
    
    # Calculate ROI
    roi = economics.calculate_roi(
        cost=workflow_cost,
        revenue_generated=500.0
    )
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


@dataclass
class CostBreakdown:
    """Cost breakdown for a workflow or agent execution."""
    token_cost: float
    compute_cost: float
    api_cost: float
    total_cost: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "token_cost": self.token_cost,
            "compute_cost": self.compute_cost,
            "api_cost": self.api_cost,
            "total_cost": self.total_cost,
        }


@dataclass
class WorkflowMetrics:
    """Economic metrics for a workflow execution."""
    workflow_id: str
    start_time: datetime
    end_time: datetime
    tokens_used: int
    compute_time: float  # seconds
    api_calls: int
    revenue_generated: float
    cost_breakdown: CostBreakdown
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def roi(self) -> float:
        """Return on investment."""
        if self.cost_breakdown.total_cost == 0:
            return 0
        return (self.revenue_generated - self.cost_breakdown.total_cost) / self.cost_breakdown.total_cost
    
    @property
    def roi_percentage(self) -> float:
        """ROI as percentage."""
        return self.roi * 100
    
    @property
    def cost_per_revenue(self) -> float:
        """Cost per dollar of revenue generated."""
        if self.revenue_generated == 0:
            return float('inf')
        return self.cost_breakdown.total_cost / self.revenue_generated


class EconomicsEngine:
    """
    Economic intelligence engine for tracking costs and ROI.
    
    Provides comprehensive cost tracking, profitability analysis,
    and economic optimization insights for autonomous operations.
    """
    
    # Cost per unit (configurable)
    TOKEN_COST = 0.00001  # $0.00001 per token
    COMPUTE_COST = 0.02   # $0.02 per second
    API_CALL_COST = 0.01  # $0.01 per API call
    
    def __init__(self):
        """Initialize economics engine."""
        self.workflow_metrics: List[WorkflowMetrics] = []
        self.agent_costs: Dict[str, CostBreakdown] = {}
    
    def calculate_cost(
        self,
        tokens_used: int = 0,
        compute_time: float = 0.0,
        api_calls: int = 0
    ) -> CostBreakdown:
        """
        Calculate cost breakdown for an execution.
        
        Args:
            tokens_used: Number of tokens used
            compute_time: Computation time in seconds
            api_calls: Number of API calls made
        
        Returns:
            CostBreakdown with itemized costs
        """
        token_cost = tokens_used * self.TOKEN_COST
        compute_cost = compute_time * self.COMPUTE_COST
        api_cost = api_calls * self.API_CALL_COST
        
        return CostBreakdown(
            token_cost=token_cost,
            compute_cost=compute_cost,
            api_cost=api_cost,
            total_cost=token_cost + compute_cost + api_cost
        )
    
    def track_workflow(
        self,
        workflow_id: str,
        start_time: datetime,
        end_time: datetime,
        tokens_used: int,
        api_calls: int,
        revenue_generated: float
    ) -> WorkflowMetrics:
        """
        Track a workflow's economic metrics.
        
        Args:
            workflow_id: Unique workflow identifier
            start_time: Workflow start time
            end_time: Workflow end time
            tokens_used: Tokens consumed
            api_calls: API calls made
            revenue_generated: Revenue this workflow generated
        
        Returns:
            WorkflowMetrics with complete economic data
        """
        compute_time = (end_time - start_time).total_seconds()
        cost = self.calculate_cost(
            tokens_used=tokens_used,
            compute_time=compute_time,
            api_calls=api_calls
        )
        
        metrics = WorkflowMetrics(
            workflow_id=workflow_id,
            start_time=start_time,
            end_time=end_time,
            tokens_used=tokens_used,
            compute_time=compute_time,
            api_calls=api_calls,
            revenue_generated=revenue_generated,
            cost_breakdown=cost
        )
        
        self.workflow_metrics.append(metrics)
        return metrics
    
    def track_agent_cost(self, agent_id: str, cost: CostBreakdown) -> None:
        """
        Accumulate cost for an agent.
        
        Args:
            agent_id: Agent identifier
            cost: Cost breakdown
        """
        if agent_id not in self.agent_costs:
            self.agent_costs[agent_id] = CostBreakdown(
                token_cost=0, compute_cost=0, api_cost=0, total_cost=0
            )
        
        existing = self.agent_costs[agent_id]
        self.agent_costs[agent_id] = CostBreakdown(
            token_cost=existing.token_cost + cost.token_cost,
            compute_cost=existing.compute_cost + cost.compute_cost,
            api_cost=existing.api_cost + cost.api_cost,
            total_cost=existing.total_cost + cost.total_cost
        )
    
    def get_workflow_roi(self, workflow_id: str) -> Optional[float]:
        """
        Get ROI percentage for a specific workflow.
        
        Args:
            workflow_id: Workflow identifier
        
        Returns:
            ROI percentage or None if not found
        """
        for metrics in self.workflow_metrics:
            if metrics.workflow_id == workflow_id:
                return metrics.roi_percentage
        return None
    
    def get_total_costs(self) -> CostBreakdown:
        """
        Get total costs across all workflows.
        
        Returns:
            Aggregated cost breakdown
        """
        total_token = sum(m.cost_breakdown.token_cost for m in self.workflow_metrics)
        total_compute = sum(m.cost_breakdown.compute_cost for m in self.workflow_metrics)
        total_api = sum(m.cost_breakdown.api_cost for m in self.workflow_metrics)
        
        return CostBreakdown(
            token_cost=total_token,
            compute_cost=total_compute,
            api_cost=total_api,
            total_cost=total_token + total_compute + total_api
        )
    
    def get_total_revenue(self) -> float:
        """
        Get total revenue generated by all workflows.
        
        Returns:
            Total revenue
        """
        return sum(m.revenue_generated for m in self.workflow_metrics)
    
    def get_overall_roi(self) -> float:
        """
        Get overall ROI percentage across all workflows.
        
        Returns:
            Overall ROI percentage
        """
        total_cost = self.get_total_costs().total_cost
        total_revenue = self.get_total_revenue()
        
        if total_cost == 0:
            return 0
        
        return ((total_revenue - total_cost) / total_cost) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get economics engine summary."""
        total_cost = self.get_total_costs()
        
        return {
            "total_workflows": len(self.workflow_metrics),
            "total_cost": total_cost.total_cost,
            "total_revenue": self.get_total_revenue(),
            "overall_roi_percentage": self.get_overall_roi(),
            "cost_breakdown": total_cost.to_dict(),
            "agents_tracked": len(self.agent_costs)
        }
