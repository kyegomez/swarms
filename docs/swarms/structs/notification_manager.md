# Notification Manager

The `NotificationManager` is responsible for managing selective notifications for vector DB updates between agents. It determines which agents should be notified of specific updates based on their expertise areas and notification preferences.

## Overview

The notification system uses relevance scoring and importance thresholds to determine which agents should receive updates. This prevents overwhelming agents with irrelevant notifications and ensures efficient information flow.

## Classes

### UpdateMetadata

Metadata for vector DB updates.

```python
class UpdateMetadata(BaseModel):
    topic: str                              # Topic of the update
    importance: float                       # Importance score (0-1)
    timestamp: datetime                     # When update occurred  
    embedding: Optional[List[float]] = None # Vector embedding
    affected_areas: List[str] = []          # Areas affected by update
```

### AgentProfile 

Profile containing agent notification preferences.

```python
class AgentProfile(BaseModel):
    agent_id: str                           # Unique agent identifier
    expertise_areas: List[str]              # Areas of expertise
    importance_threshold: float = 0.5       # Min importance threshold
    current_task_context: Optional[str]     # Current task being worked on
    embedding: Optional[List[float]]        # Vector embedding of expertise
```

### NotificationManager

Main class for managing notifications.

```python
class NotificationManager:
    def __init__(self):
        self.agent_profiles: Dict[str, AgentProfile] = {}
```

## Key Methods

| Method | Description |
|--------|-------------|
| `register_agent(profile: AgentProfile)` | Register an agent's notification preferences |
| `unregister_agent(agent_id: str)` | Remove an agent's notification preferences |
| `calculate_relevance(update_metadata: UpdateMetadata, agent_profile: AgentProfile) -> float` | Calculate relevance score between update and agent |
| `should_notify_agent(update_metadata: UpdateMetadata, agent_profile: AgentProfile) -> bool` | Determine if agent should be notified |
| `get_agents_to_notify(update_metadata: UpdateMetadata) -> List[str]` | Get list of agents to notify |

## Usage Example

```python
from swarms.structs.notification_manager import NotificationManager, UpdateMetadata, AgentProfile
from datetime import datetime

# Create notification manager
manager = NotificationManager()

# Register an agent
profile = AgentProfile(
    agent_id="financial_agent",
    expertise_areas=["finance", "stocks", "trading"],
    importance_threshold=0.6
)
manager.register_agent(profile)

# Create an update
update = UpdateMetadata(
    topic="stock_market",
    importance=0.8,
    timestamp=datetime.now(),
    affected_areas=["finance", "trading"]
)

# Get agents to notify
agents_to_notify = manager.get_agents_to_notify(update)
print(f"Agents to notify: {agents_to_notify}")
```

## Relevance Calculation

The relevance score between an update and an agent is calculated using:

1. Topic/expertise overlap score (70% weight)
   - Measures overlap between agent's expertise areas and update's affected areas

2. Embedding similarity score (30% weight) 
   - Cosine similarity between update and agent embeddings if available

The final relevance score determines if an agent should be notified based on:
- Relevance score > 0.5
- Update importance > agent's importance threshold

## Best Practices

1. Set appropriate importance thresholds
   - Higher thresholds (e.g. 0.8) for critical updates only
   - Lower thresholds (e.g. 0.3) for broader awareness

2. Define focused expertise areas
   - Use specific areas rather than broad categories
   - Include related areas for better matching

3. Update task context
   - Keep current_task_context updated for better relevance
   - Clear context when task complete

4. Monitor notification patterns
   - Adjust thresholds if agents receive too many/few updates
   - Refine expertise areas based on relevance scores