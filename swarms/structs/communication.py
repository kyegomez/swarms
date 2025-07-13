"""
Enhanced Communication Protocol System for HierarchicalSwarm

This module provides advanced communication capabilities including:
- Multi-directional message passing
- Priority-based routing
- Message queuing and buffering
- Communication channels with different protocols
- Advanced feedback mechanisms
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union, Callable
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the communication system"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_COMPLETION = "task_completion"
    FEEDBACK = "feedback"
    QUERY = "query"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    ESCALATION = "escalation"
    COORDINATION = "coordination"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    HANDOFF = "handoff"
    COLLABORATION = "collaboration"


class MessagePriority(Enum):
    """Priority levels for messages"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class MessageStatus(Enum):
    """Status of messages"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class Message:
    """Enhanced message structure for hierarchical communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""
    message_type: MessageType = MessageType.TASK_ASSIGNMENT
    priority: MessagePriority = MessagePriority.MEDIUM
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    expiry_time: Optional[float] = None
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    conversation_id: Optional[str] = None
    status: MessageStatus = MessageStatus.PENDING
    retry_count: int = 0
    max_retries: int = 3
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expiry_time is not None and time.time() > self.expiry_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'expiry_time': self.expiry_time,
            'requires_response': self.requires_response,
            'parent_message_id': self.parent_message_id,
            'conversation_id': self.conversation_id,
            'status': self.status.value,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }


class CommunicationProtocol(Protocol):
    """Protocol for communication handlers"""
    
    def handle_message(self, message: Message) -> Optional[Message]:
        """Handle an incoming message"""
        pass
    
    def can_handle(self, message_type: MessageType) -> bool:
        """Check if this protocol can handle the message type"""
        pass


class MessageQueue:
    """Thread-safe message queue with priority support"""
    
    def __init__(self, max_size: int = 1000):
        self.queues = {
            MessagePriority.CRITICAL: deque(),
            MessagePriority.HIGH: deque(),
            MessagePriority.MEDIUM: deque(),
            MessagePriority.LOW: deque()
        }
        self.max_size = max_size
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.size = 0
        
    def put(self, message: Message, timeout: Optional[float] = None) -> bool:
        """Add message to queue with timeout"""
        with self.condition:
            while self.size >= self.max_size:
                if timeout is None:
                    return False
                if not self.condition.wait(timeout):
                    return False
            
            if message.is_expired():
                return False
                
            self.queues[message.priority].append(message)
            self.size += 1
            self.condition.notify()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get message from queue with timeout"""
        with self.condition:
            while self.size == 0:
                if timeout is None:
                    self.condition.wait()
                elif not self.condition.wait(timeout):
                    return None
            
            # Get highest priority message
            for priority in MessagePriority:
                if self.queues[priority]:
                    message = self.queues[priority].popleft()
                    self.size -= 1
                    self.condition.notify()
                    return message
            
            return None
    
    def peek(self) -> Optional[Message]:
        """Peek at next message without removing it"""
        with self.lock:
            for priority in MessagePriority:
                if self.queues[priority]:
                    return self.queues[priority][0]
            return None
    
    def clear_expired(self):
        """Remove expired messages from queue"""
        with self.lock:
            for priority in MessagePriority:
                queue = self.queues[priority]
                expired_count = 0
                while queue and queue[0].is_expired():
                    queue.popleft()
                    expired_count += 1
                self.size -= expired_count


class CommunicationChannel:
    """Communication channel between agents"""
    
    def __init__(self, 
                 channel_id: str,
                 participants: List[str],
                 channel_type: str = "direct",
                 max_queue_size: int = 100):
        self.channel_id = channel_id
        self.participants = set(participants)
        self.channel_type = channel_type
        self.message_queue = MessageQueue(max_queue_size)
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.active = True
        self.created_at = time.time()
        
    def add_participant(self, participant_id: str):
        """Add participant to channel"""
        self.participants.add(participant_id)
        
    def remove_participant(self, participant_id: str):
        """Remove participant from channel"""
        self.participants.discard(participant_id)
        
    def send_message(self, message: Message) -> bool:
        """Send message through channel"""
        if not self.active:
            return False
            
        if message.sender_id not in self.participants:
            return False
            
        if message.receiver_id and message.receiver_id not in self.participants:
            return False
            
        return self.message_queue.put(message)
    
    def receive_message(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Receive message from channel"""
        if not self.active:
            return None
            
        return self.message_queue.get(timeout)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type].append(handler)
    
    def handle_message(self, message: Message) -> List[Message]:
        """Handle message using registered handlers"""
        responses = []
        for handler in self.message_handlers.get(message.message_type, []):
            try:
                response = handler(message)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Error handling message {message.id}: {e}")
                
        return responses


class CommunicationRouter:
    """Routes messages between agents and channels"""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.agent_channels: Dict[str, List[str]] = defaultdict(list)
        self.message_history: Dict[str, List[Message]] = defaultdict(list)
        self.routing_table: Dict[str, str] = {}  # agent_id -> preferred_channel
        self.lock = threading.Lock()
        
    def create_channel(self, 
                      channel_id: str,
                      participants: List[str],
                      channel_type: str = "direct") -> CommunicationChannel:
        """Create new communication channel"""
        with self.lock:
            channel = CommunicationChannel(channel_id, participants, channel_type)
            self.channels[channel_id] = channel
            
            for participant in participants:
                self.agent_channels[participant].append(channel_id)
                
            return channel
    
    def get_channel(self, channel_id: str) -> Optional[CommunicationChannel]:
        """Get communication channel"""
        return self.channels.get(channel_id)
    
    def route_message(self, message: Message) -> bool:
        """Route message to appropriate channel"""
        with self.lock:
            # Find appropriate channel
            sender_channels = self.agent_channels.get(message.sender_id, [])
            receiver_channels = self.agent_channels.get(message.receiver_id, [])
            
            # Find common channel
            common_channels = set(sender_channels) & set(receiver_channels)
            
            if not common_channels:
                # Create direct channel if none exists
                channel_id = f"{message.sender_id}_{message.receiver_id}"
                channel = self.create_channel(
                    channel_id, 
                    [message.sender_id, message.receiver_id],
                    "direct"
                )
                common_channels = {channel_id}
            
            # Use first available channel
            channel_id = next(iter(common_channels))
            channel = self.channels[channel_id]
            
            # Store message in history
            self.message_history[message.conversation_id or "default"].append(message)
            
            return channel.send_message(message)
    
    def broadcast_message(self, message: Message, channel_ids: List[str]) -> Dict[str, bool]:
        """Broadcast message to multiple channels"""
        results = {}
        for channel_id in channel_ids:
            channel = self.channels.get(channel_id)
            if channel:
                results[channel_id] = channel.send_message(message)
            else:
                results[channel_id] = False
        return results
    
    def get_agent_channels(self, agent_id: str) -> List[str]:
        """Get channels for an agent"""
        return self.agent_channels.get(agent_id, [])
    
    def get_conversation_history(self, conversation_id: str) -> List[Message]:
        """Get conversation history"""
        return self.message_history.get(conversation_id, [])


class FeedbackSystem:
    """Advanced feedback system for hierarchical communication"""
    
    def __init__(self):
        self.feedback_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.feedback_processors: Dict[str, Callable] = {}
        
    def register_feedback_processor(self, feedback_type: str, processor: Callable):
        """Register feedback processor"""
        self.feedback_processors[feedback_type] = processor
        
    def process_feedback(self, 
                        agent_id: str,
                        feedback_type: str,
                        feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback for an agent"""
        processor = self.feedback_processors.get(feedback_type)
        if processor:
            processed_feedback = processor(feedback_data)
        else:
            processed_feedback = feedback_data
            
        # Store feedback history
        feedback_entry = {
            'timestamp': time.time(),
            'type': feedback_type,
            'data': processed_feedback,
            'agent_id': agent_id
        }
        self.feedback_history[agent_id].append(feedback_entry)
        
        # Update performance metrics
        self._update_performance_metrics(agent_id, feedback_type, processed_feedback)
        
        return processed_feedback
    
    def _update_performance_metrics(self, 
                                   agent_id: str, 
                                   feedback_type: str,
                                   feedback_data: Dict[str, Any]):
        """Update performance metrics based on feedback"""
        metrics = self.performance_metrics[agent_id]
        
        # Extract numeric metrics from feedback
        for key, value in feedback_data.items():
            if isinstance(value, (int, float)):
                metric_key = f"{feedback_type}_{key}"
                if metric_key in metrics:
                    # Simple moving average
                    metrics[metric_key] = (metrics[metric_key] + value) / 2
                else:
                    metrics[metric_key] = value
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """Get performance metrics for an agent"""
        return self.performance_metrics.get(agent_id, {})
    
    def get_agent_feedback_history(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get feedback history for an agent"""
        return self.feedback_history.get(agent_id, [])


class EscalationManager:
    """Manages escalation of issues in hierarchical communication"""
    
    def __init__(self):
        self.escalation_rules: Dict[str, Dict[str, Any]] = {}
        self.escalation_history: List[Dict[str, Any]] = []
        self.escalation_handlers: Dict[str, Callable] = {}
        
    def register_escalation_rule(self, 
                                rule_id: str,
                                condition: Callable,
                                escalation_target: str,
                                escalation_level: int = 1):
        """Register escalation rule"""
        self.escalation_rules[rule_id] = {
            'condition': condition,
            'target': escalation_target,
            'level': escalation_level
        }
        
    def register_escalation_handler(self, level: int, handler: Callable):
        """Register escalation handler for specific level"""
        self.escalation_handlers[f"level_{level}"] = handler
        
    def check_escalation(self, message: Message) -> Optional[str]:
        """Check if message should be escalated"""
        for rule_id, rule in self.escalation_rules.items():
            if rule['condition'](message):
                return rule['target']
        return None
    
    def escalate_message(self, message: Message, escalation_target: str) -> Message:
        """Escalate message to higher level"""
        escalation_message = Message(
            sender_id=message.receiver_id,
            receiver_id=escalation_target,
            message_type=MessageType.ESCALATION,
            priority=MessagePriority.HIGH,
            content={
                'original_message': message.to_dict(),
                'escalation_reason': "Automatic escalation triggered",
                'escalation_timestamp': time.time()
            },
            parent_message_id=message.id,
            conversation_id=message.conversation_id
        )
        
        # Record escalation
        self.escalation_history.append({
            'timestamp': time.time(),
            'original_message_id': message.id,
            'escalation_message_id': escalation_message.id,
            'escalation_target': escalation_target
        })
        
        return escalation_message


class CommunicationManager:
    """Main communication manager for hierarchical swarm"""
    
    def __init__(self):
        self.router = CommunicationRouter()
        self.feedback_system = FeedbackSystem()
        self.escalation_manager = EscalationManager()
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        self.message_processors: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.running = False
        
    def start(self):
        """Start the communication manager"""
        self.running = True
        
    def stop(self):
        """Stop the communication manager"""
        self.running = False
        self.executor.shutdown(wait=True)
        
    def register_message_processor(self, processor: Callable):
        """Register message processor"""
        self.message_processors.append(processor)
        
    def send_message(self, message: Message) -> bool:
        """Send message through the system"""
        if not self.running:
            return False
            
        # Process message through registered processors
        for processor in self.message_processors:
            try:
                message = processor(message) or message
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                return False
        
        # Check for escalation
        escalation_target = self.escalation_manager.check_escalation(message)
        if escalation_target:
            escalation_message = self.escalation_manager.escalate_message(message, escalation_target)
            self.router.route_message(escalation_message)
        
        # Route message
        return self.router.route_message(message)
    
    def create_conversation(self, 
                          conversation_id: str,
                          participants: List[str],
                          conversation_type: str = "group") -> str:
        """Create new conversation"""
        channel_id = f"conv_{conversation_id}"
        channel = self.router.create_channel(channel_id, participants, conversation_type)
        
        self.active_conversations[conversation_id] = {
            'channel_id': channel_id,
            'participants': participants,
            'type': conversation_type,
            'created_at': time.time()
        }
        
        return channel_id
    
    def get_conversation_messages(self, conversation_id: str) -> List[Message]:
        """Get messages from conversation"""
        return self.router.get_conversation_history(conversation_id)
    
    def process_feedback(self, 
                        agent_id: str,
                        feedback_type: str,
                        feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process feedback for an agent"""
        return self.feedback_system.process_feedback(agent_id, feedback_type, feedback_data)
    
    def get_agent_performance(self, agent_id: str) -> Dict[str, float]:
        """Get performance metrics for an agent"""
        return self.feedback_system.get_agent_performance(agent_id)
    
    async def async_send_message(self, message: Message) -> bool:
        """Send message asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.send_message, message)
    
    def broadcast_to_all_agents(self, 
                               message: Message,
                               exclude_agents: Optional[List[str]] = None) -> Dict[str, bool]:
        """Broadcast message to all agents"""
        exclude_agents = exclude_agents or []
        all_agents = set()
        
        for agent_id in self.router.agent_channels.keys():
            if agent_id not in exclude_agents:
                all_agents.add(agent_id)
        
        # Create broadcast message for each agent
        results = {}
        for agent_id in all_agents:
            agent_message = Message(
                sender_id=message.sender_id,
                receiver_id=agent_id,
                message_type=MessageType.BROADCAST,
                priority=message.priority,
                content=message.content,
                metadata=message.metadata,
                conversation_id=message.conversation_id
            )
            results[agent_id] = self.send_message(agent_message)
        
        return results
    
    def create_agent_channel(self, agent_id: str) -> str:
        """Create dedicated channel for agent"""
        channel_id = f"agent_{agent_id}"
        self.router.create_channel(channel_id, [agent_id], "agent")
        return channel_id
    
    def get_channel_statistics(self) -> Dict[str, Any]:
        """Get communication statistics"""
        stats = {
            'total_channels': len(self.router.channels),
            'active_conversations': len(self.active_conversations),
            'total_agents': len(self.router.agent_channels),
            'message_history_size': sum(len(history) for history in self.router.message_history.values()),
            'escalation_count': len(self.escalation_manager.escalation_history)
        }
        return stats