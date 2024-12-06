from typing import Dict, List, Optional
from pydantic import BaseModel
import numpy as np
from datetime import datetime

class UpdateMetadata(BaseModel):
    """Metadata for vector DB updates"""
    topic: str
    importance: float 
    timestamp: datetime
    embedding: Optional[List[float]] = None
    affected_areas: List[str] = []

class AgentProfile(BaseModel):
    """Profile for agent notification preferences"""
    agent_id: str
    expertise_areas: List[str]
    importance_threshold: float = 0.5
    current_task_context: Optional[str] = None
    embedding: Optional[List[float]] = None
    
class NotificationManager:
    """Manages selective notifications for vector DB updates"""
    
    def __init__(self):
        self.agent_profiles: Dict[str, AgentProfile] = {}
        
    def register_agent(self, profile: AgentProfile):
        """Register an agent's notification preferences"""
        self.agent_profiles[profile.agent_id] = profile
        
    def unregister_agent(self, agent_id: str):
        """Remove an agent's notification preferences"""
        if agent_id in self.agent_profiles:
            del self.agent_profiles[agent_id]
            
    def calculate_relevance(
        self,
        update_metadata: UpdateMetadata,
        agent_profile: AgentProfile
    ) -> float:
        """Calculate relevance score between update and agent"""
        # Topic/expertise overlap score
        topic_score = len(
            set(agent_profile.expertise_areas) & 
            set(update_metadata.affected_areas)
        ) / max(
            len(agent_profile.expertise_areas),
            len(update_metadata.affected_areas)
        )
        
        # Embedding similarity if available
        embedding_score = 0.0
        if update_metadata.embedding and agent_profile.embedding:
            embedding_score = np.dot(
                update_metadata.embedding,
                agent_profile.embedding
            )
            
        # Combine scores (can be tuned)
        relevance = 0.7 * topic_score + 0.3 * embedding_score
        
        return relevance
        
    def should_notify_agent(
        self,
        update_metadata: UpdateMetadata,
        agent_profile: AgentProfile
    ) -> bool:
        """Determine if an agent should be notified of an update"""
        # Check importance threshold
        if update_metadata.importance < agent_profile.importance_threshold:
            return False
            
        # Calculate relevance
        relevance = self.calculate_relevance(update_metadata, agent_profile)
        
        # Notification threshold (can be tuned)
        return relevance > 0.5
        
    def get_agents_to_notify(
        self,
        update_metadata: UpdateMetadata
    ) -> List[str]:
        """Get list of agent IDs that should be notified of an update"""
        agents_to_notify = []
        
        for agent_id, profile in self.agent_profiles.items():
            if self.should_notify_agent(update_metadata, profile):
                agents_to_notify.append(agent_id)
                
        return agents_to_notify 