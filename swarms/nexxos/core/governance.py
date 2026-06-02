"""
Governance Engine - Policy Enforcement & Access Control

Responsibilities:
- Permission management (role-based access control)
- Policy enforcement (execution scope, financial limits, API access)
- Approval workflows (multi-level approval for sensitive actions)
- Risk scoring (operational, financial, security, compliance)
- Audit logging (complete action trail)

Example:
    governance = GovernanceEngine()
    
    # Check permission before action
    if governance.has_permission(agent_id, action="price_change"):
        execute_action()
    
    # Score risk for approval workflow
    risk = governance.score_risk(
        action="refund_customer",
        amount=500,
        agent_id="pricing_agent"
    )
    
    # Log action for audit trail
    governance.audit_log(
        agent="pricing_agent",
        action="price_change",
        old_value=40,
        new_value=45,
        reason="competitor_price_increase"
    )
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class ActionType(str, Enum):
    """Types of actions that require governance control."""
    SEND_EMAIL = "send_email"
    MODIFY_PRICING = "modify_pricing"
    REFUND_CUSTOMER = "refund_customer"
    DELETE_RECORDS = "delete_records"
    API_CALL = "api_call"
    TOOL_EXECUTION = "tool_execution"
    FINANCIAL_TRANSFER = "financial_transfer"


class RiskLevel(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskScore:
    """Risk assessment for an action."""
    financial_risk: float = 0.0  # 0-1
    security_risk: float = 0.0   # 0-1
    operational_risk: float = 0.0 # 0-1
    compliance_risk: float = 0.0  # 0-1
    
    def total_score(self) -> float:
        """Calculate weighted risk score."""
        return (
            self.financial_risk * 0.4 +
            self.security_risk * 0.3 +
            self.operational_risk * 0.2 +
            self.compliance_risk * 0.1
        )
    
    def level(self) -> RiskLevel:
        """Determine risk level from score."""
        score = self.total_score()
        if score < 0.25:
            return RiskLevel.LOW
        elif score < 0.5:
            return RiskLevel.MEDIUM
        elif score < 0.75:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


@dataclass
class Policy:
    """Policy definition for action control."""
    action: ActionType
    allowed_agents: List[str] = field(default_factory=list)
    requires_approval: bool = False
    approval_level: str = "supervisor"  # supervisor, director, cfo, etc.
    max_value: Optional[float] = None
    rate_limit: Optional[int] = None  # actions per hour
    description: str = ""


@dataclass
class AuditLog:
    """Audit log entry for compliance and debugging."""
    timestamp: datetime
    agent_id: str
    action: ActionType
    parameters: Dict[str, Any]
    status: str  # "approved", "executed", "denied"
    risk_score: float
    approval_chain: List[str] = field(default_factory=list)
    notes: str = ""


class GovernanceEngine:
    """
    Enterprise governance engine for autonomous agent control.
    
    Ensures that autonomous agents operate within defined policies,
    with proper approval workflows and complete audit trails.
    """
    
    def __init__(self):
        """Initialize governance engine."""
        self.policies: Dict[ActionType, Policy] = {}
        self.audit_logs: List[AuditLog] = []
        self._setup_default_policies()
    
    def _setup_default_policies(self):
        """Set up default governance policies."""
        # Email campaigns require marketing supervisor approval
        self.add_policy(Policy(
            action=ActionType.SEND_EMAIL,
            allowed_agents=["marketing_agent"],
            requires_approval=True,
            approval_level="marketing_supervisor",
            description="Send email campaign"
        ))
        
        # Pricing changes require ecommerce director approval
        self.add_policy(Policy(
            action=ActionType.MODIFY_PRICING,
            allowed_agents=["pricing_agent"],
            requires_approval=True,
            approval_level="ecommerce_director",
            max_value=0.2,  # Max 20% price change
            description="Modify product pricing"
        ))
        
        # Refunds require finance approval
        self.add_policy(Policy(
            action=ActionType.REFUND_CUSTOMER,
            allowed_agents=["customer_service_agent"],
            requires_approval=True,
            approval_level="finance_approval",
            max_value=5000,  # Max $5000 per refund
            description="Process customer refund"
        ))
        
        # Data deletion requires human approval
        self.add_policy(Policy(
            action=ActionType.DELETE_RECORDS,
            allowed_agents=[],  # No autonomous agents
            requires_approval=True,
            approval_level="data_admin",
            description="Delete records from database"
        ))
    
    def add_policy(self, policy: Policy) -> None:
        """Add or update a governance policy."""
        self.policies[policy.action] = policy
    
    def has_permission(self, agent_id: str, action: ActionType) -> bool:
        """
        Check if an agent has permission to perform an action.
        
        Args:
            agent_id: The agent attempting the action
            action: The action type
        
        Returns:
            True if agent has permission, False otherwise
        """
        if action not in self.policies:
            # No policy defined = allowed by default
            return True
        
        policy = self.policies[action]
        
        # If allowed_agents is empty, deny by default (unless it's an approval action)
        if not policy.allowed_agents and action == ActionType.DELETE_RECORDS:
            return False
        
        # Check if agent is in allowed list
        return agent_id in policy.allowed_agents if policy.allowed_agents else True
    
    def score_risk(self, agent_id: str, action: ActionType, 
                   parameters: Dict[str, Any]) -> RiskScore:
        """
        Score the risk of an action.
        
        Args:
            agent_id: The agent performing the action
            action: The action type
            parameters: Action parameters (amount, etc.)
        
        Returns:
            RiskScore object with component scores
        """
        risk = RiskScore()
        
        # Financial risk based on action amount
        if "amount" in parameters:
            amount = parameters["amount"]
            risk.financial_risk = min(1.0, amount / 10000)  # Normalize to 0-1
        
        # Security risk based on action type
        if action in [ActionType.DELETE_RECORDS, ActionType.API_CALL]:
            risk.security_risk = 0.8
        elif action == ActionType.SEND_EMAIL:
            risk.security_risk = 0.3
        
        # Operational risk based on agent and action
        if action == ActionType.MODIFY_PRICING:
            risk.operational_risk = 0.6
        elif action == ActionType.REFUND_CUSTOMER:
            risk.operational_risk = 0.5
        
        # Compliance risk
        if action in [ActionType.DELETE_RECORDS, ActionType.FINANCIAL_TRANSFER]:
            risk.compliance_risk = 0.7
        
        return risk
    
    def requires_approval(self, agent_id: str, action: ActionType) -> bool:
        """
        Check if an action requires approval.
        
        Args:
            agent_id: The agent performing the action
            action: The action type
        
        Returns:
            True if approval is required
        """
        if action not in self.policies:
            return False
        return self.policies[action].requires_approval
    
    def get_approval_level(self, action: ActionType) -> str:
        """
        Get the required approval level for an action.
        
        Args:
            action: The action type
        
        Returns:
            Approval level required (supervisor, director, etc.)
        """
        if action in self.policies:
            return self.policies[action].approval_level
        return "none"
    
    def audit_log(self, agent_id: str, action: ActionType, 
                  parameters: Dict[str, Any], status: str,
                  approval_chain: List[str] = None,
                  notes: str = "") -> None:
        """
        Log an action for audit trail.
        
        Args:
            agent_id: The agent performing the action
            action: The action type
            parameters: Action parameters
            status: Action status (approved, executed, denied)
            approval_chain: List of approvers
            notes: Additional notes
        """
        risk_score = self.score_risk(agent_id, action, parameters).total_score()
        
        log_entry = AuditLog(
            timestamp=datetime.now(),
            agent_id=agent_id,
            action=action,
            parameters=parameters,
            status=status,
            risk_score=risk_score,
            approval_chain=approval_chain or [],
            notes=notes
        )
        
        self.audit_logs.append(log_entry)
    
    def get_audit_logs(self, agent_id: str = None, 
                      action: ActionType = None) -> List[AuditLog]:
        """
        Retrieve audit logs with optional filtering.
        
        Args:
            agent_id: Filter by agent ID
            action: Filter by action type
        
        Returns:
            List of matching audit log entries
        """
        logs = self.audit_logs
        
        if agent_id:
            logs = [l for l in logs if l.agent_id == agent_id]
        if action:
            logs = [l for l in logs if l.action == action]
        
        return logs
    
    def get_summary(self) -> Dict[str, Any]:
        """Get governance engine summary."""
        return {
            "total_policies": len(self.policies),
            "total_audit_logs": len(self.audit_logs),
            "policies": [
                {
                    "action": p.action.value,
                    "requires_approval": p.requires_approval,
                    "allowed_agents": p.allowed_agents
                }
                for p in self.policies.values()
            ]
        }
