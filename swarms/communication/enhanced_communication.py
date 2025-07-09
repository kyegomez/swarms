"""
Enhanced Multi-Agent Communication System

This module provides a robust, reliable communication infrastructure for multi-agent systems
with advanced features including message queuing, retry mechanisms, frequency management,
and hierarchical cooperation protocols.
"""

import asyncio
import time
import threading
import uuid
import json
import logging
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Set, Union, Tuple, 
    Protocol, TypeVar, Generic
)
from queue import Queue, PriorityQueue
import heapq
from datetime import datetime, timedelta

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="enhanced_communication")

# Type definitions
AgentID = str
MessageID = str
T = TypeVar('T')


class MessagePriority(Enum):
    """Message priority levels for queue management"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class MessageType(Enum):
    """Types of messages in the system"""
    TASK = "task"
    RESPONSE = "response"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    ERROR = "error"
    BROADCAST = "broadcast"
    DIRECT = "direct"
    FEEDBACK = "feedback"
    ACKNOWLEDGMENT = "acknowledgment"


class MessageStatus(Enum):
    """Message delivery status"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRY = "retry"


class CommunicationProtocol(Enum):
    """Communication protocols for different scenarios"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    HIERARCHICAL = "hierarchical"
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"


@dataclass
class MessageMetadata:
    """Metadata for message tracking and management"""
    created_at: datetime = field(default_factory=datetime.now)
    ttl: Optional[timedelta] = None
    retry_count: int = 0
    max_retries: int = 3
    requires_ack: bool = False
    reply_to: Optional[MessageID] = None
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None


@dataclass
class EnhancedMessage:
    """Enhanced message class with comprehensive metadata"""
    id: MessageID = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: AgentID = ""
    receiver_id: Optional[AgentID] = None
    receiver_ids: Optional[List[AgentID]] = None
    content: Union[str, Dict, List, Any] = ""
    message_type: MessageType = MessageType.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    protocol: CommunicationProtocol = CommunicationProtocol.DIRECT
    metadata: MessageMetadata = field(default_factory=MessageMetadata)
    status: MessageStatus = MessageStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    processing_time: Optional[float] = None
    error_details: Optional[str] = None

    def __lt__(self, other):
        """Support for priority queue ordering"""
        if not isinstance(other, EnhancedMessage):
            return NotImplemented
        return (self.priority.value, self.timestamp) < (other.priority.value, other.timestamp)

    def is_expired(self) -> bool:
        """Check if message has expired"""
        if self.metadata.ttl is None:
            return False
        return datetime.now() > self.timestamp + self.metadata.ttl

    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return (
            self.status == MessageStatus.FAILED and
            self.metadata.retry_count < self.metadata.max_retries and
            not self.is_expired()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "receiver_ids": self.receiver_ids,
            "content": self.content,
            "message_type": self.message_type.value,
            "priority": self.priority.value,
            "protocol": self.protocol.value,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "processing_time": self.processing_time,
            "error_details": self.error_details,
            "metadata": {
                "created_at": self.metadata.created_at.isoformat(),
                "ttl": self.metadata.ttl.total_seconds() if self.metadata.ttl else None,
                "retry_count": self.metadata.retry_count,
                "max_retries": self.metadata.max_retries,
                "requires_ack": self.metadata.requires_ack,
                "reply_to": self.metadata.reply_to,
                "conversation_id": self.metadata.conversation_id,
                "trace_id": self.metadata.trace_id,
            }
        }


class RateLimiter:
    """Rate limiter for message frequency control"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
        self._lock = threading.Lock()
    
    def is_allowed(self) -> bool:
        """Check if request is allowed under rate limit"""
        with self._lock:
            now = time.time()
            
            # Remove old requests outside time window
            while self.requests and self.requests[0] <= now - self.time_window:
                self.requests.popleft()
            
            # Check if we can make a new request
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    def wait_time(self) -> float:
        """Get time to wait before next request is allowed"""
        with self._lock:
            if len(self.requests) < self.max_requests:
                return 0.0
            
            oldest_request = self.requests[0]
            return max(0.0, oldest_request + self.time_window - time.time())


class MessageQueue:
    """Enhanced message queue with priority and persistence"""
    
    def __init__(self, max_size: int = 10000, persist: bool = False):
        self.max_size = max_size
        self.persist = persist
        self._queue = PriorityQueue(maxsize=max_size)
        self._pending_messages: Dict[MessageID, EnhancedMessage] = {}
        self._completed_messages: Dict[MessageID, EnhancedMessage] = {}
        self._lock = threading.Lock()
        self._stats = defaultdict(int)
    
    def put(self, message: EnhancedMessage, block: bool = True, timeout: Optional[float] = None) -> bool:
        """Add message to queue"""
        try:
            # Create priority tuple (negative priority for max-heap behavior)
            priority_item = (-message.priority.value, message.timestamp, message)
            self._queue.put(priority_item, block=block, timeout=timeout)
            
            with self._lock:
                self._pending_messages[message.id] = message
                self._stats['messages_queued'] += 1
            
            logger.debug(f"Message {message.id} queued with priority {message.priority.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue message {message.id}: {e}")
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[EnhancedMessage]:
        """Get next message from queue"""
        try:
            priority_item = self._queue.get(block=block, timeout=timeout)
            message = priority_item[2]  # Extract message from priority tuple
            
            with self._lock:
                if message.id in self._pending_messages:
                    del self._pending_messages[message.id]
                self._stats['messages_dequeued'] += 1
            
            return message
            
        except Exception:
            return None
    
    def mark_completed(self, message_id: MessageID, message: EnhancedMessage):
        """Mark message as completed"""
        with self._lock:
            self._completed_messages[message_id] = message
            self._stats['messages_completed'] += 1
            
            # Clean up old completed messages
            if len(self._completed_messages) > self.max_size // 2:
                # Remove oldest 25% of completed messages
                sorted_messages = sorted(
                    self._completed_messages.items(),
                    key=lambda x: x[1].timestamp
                )
                remove_count = len(sorted_messages) // 4
                for mid, _ in sorted_messages[:remove_count]:
                    del self._completed_messages[mid]
    
    def get_pending_count(self) -> int:
        """Get number of pending messages"""
        return len(self._pending_messages)
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self._lock:
            return dict(self._stats)


class MessageBroker:
    """Central message broker for routing and delivery"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._agents: Dict[AgentID, 'CommunicationAgent'] = {}
        self._message_queues: Dict[AgentID, MessageQueue] = {}
        self._rate_limiters: Dict[AgentID, RateLimiter] = {}
        self._subscribers: Dict[str, Set[AgentID]] = defaultdict(set)
        self._running = False
        self._workers: List[threading.Thread] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._stats = defaultdict(int)
        self._lock = threading.Lock()
        
        # Message delivery tracking
        self._pending_acks: Dict[MessageID, EnhancedMessage] = {}
        self._retry_queue: List[Tuple[float, EnhancedMessage]] = []
        
        # Start message processing
        self.start()
    
    def register_agent(self, agent: 'CommunicationAgent', rate_limit: Tuple[int, float] = (100, 60.0)):
        """Register an agent with the broker"""
        with self._lock:
            self._agents[agent.agent_id] = agent
            self._message_queues[agent.agent_id] = MessageQueue()
            max_requests, time_window = rate_limit
            self._rate_limiters[agent.agent_id] = RateLimiter(max_requests, time_window)
        
        logger.info(f"Registered agent {agent.agent_id} with rate limit {rate_limit}")
    
    def unregister_agent(self, agent_id: AgentID):
        """Unregister an agent"""
        with self._lock:
            self._agents.pop(agent_id, None)
            self._message_queues.pop(agent_id, None)
            self._rate_limiters.pop(agent_id, None)
            
            # Remove from all subscriptions
            for topic_subscribers in self._subscribers.values():
                topic_subscribers.discard(agent_id)
        
        logger.info(f"Unregistered agent {agent_id}")
    
    def send_message(self, message: EnhancedMessage) -> bool:
        """Send a message through the broker"""
        try:
            # Validate message
            if not self._validate_message(message):
                return False
            
            # Check rate limits for sender
            if message.sender_id in self._rate_limiters:
                rate_limiter = self._rate_limiters[message.sender_id]
                if not rate_limiter.is_allowed():
                    wait_time = rate_limiter.wait_time()
                    logger.warning(f"Rate limit exceeded for {message.sender_id}. Wait {wait_time:.2f}s")
                    message.status = MessageStatus.FAILED
                    message.error_details = f"Rate limit exceeded. Wait {wait_time:.2f}s"
                    return False
            
            # Route message based on protocol
            return self._route_message(message)
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            message.status = MessageStatus.FAILED
            message.error_details = str(e)
            return False
    
    def _validate_message(self, message: EnhancedMessage) -> bool:
        """Validate message before processing"""
        if not message.sender_id:
            logger.error(f"Message {message.id} missing sender_id")
            return False
        
        if message.protocol == CommunicationProtocol.DIRECT and not message.receiver_id:
            logger.error(f"Direct message {message.id} missing receiver_id")
            return False
        
        if message.protocol in [CommunicationProtocol.BROADCAST, CommunicationProtocol.MULTICAST]:
            if not message.receiver_ids:
                logger.error(f"Broadcast/Multicast message {message.id} missing receiver_ids")
                return False
        
        if message.is_expired():
            logger.warning(f"Message {message.id} has expired")
            message.status = MessageStatus.EXPIRED
            return False
        
        return True
    
    def _route_message(self, message: EnhancedMessage) -> bool:
        """Route message based on protocol"""
        try:
            if message.protocol == CommunicationProtocol.DIRECT:
                return self._route_direct_message(message)
            elif message.protocol == CommunicationProtocol.BROADCAST:
                return self._route_broadcast_message(message)
            elif message.protocol == CommunicationProtocol.MULTICAST:
                return self._route_multicast_message(message)
            elif message.protocol == CommunicationProtocol.PUBLISH_SUBSCRIBE:
                return self._route_pubsub_message(message)
            elif message.protocol == CommunicationProtocol.HIERARCHICAL:
                return self._route_hierarchical_message(message)
            else:
                logger.error(f"Unknown protocol {message.protocol} for message {message.id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to route message {message.id}: {e}")
            return False
    
    def _route_direct_message(self, message: EnhancedMessage) -> bool:
        """Route direct message to specific agent"""
        receiver_id = message.receiver_id
        if receiver_id not in self._message_queues:
            logger.error(f"Receiver {receiver_id} not found for message {message.id}")
            return False
        
        queue = self._message_queues[receiver_id]
        success = queue.put(message, block=False)
        
        if success:
            message.status = MessageStatus.SENT
            self._stats['direct_messages_sent'] += 1
            
            # Track acknowledgment if required
            if message.metadata.requires_ack:
                self._pending_acks[message.id] = message
        
        return success
    
    def _route_broadcast_message(self, message: EnhancedMessage) -> bool:
        """Route broadcast message to all agents"""
        success_count = 0
        total_count = 0
        
        for agent_id, queue in self._message_queues.items():
            if agent_id == message.sender_id:  # Don't send to sender
                continue
                
            total_count += 1
            # Create copy for each receiver
            msg_copy = EnhancedMessage(
                id=str(uuid.uuid4()),
                sender_id=message.sender_id,
                receiver_id=agent_id,
                content=message.content,
                message_type=message.message_type,
                priority=message.priority,
                protocol=message.protocol,
                metadata=message.metadata,
                timestamp=message.timestamp
            )
            
            if queue.put(msg_copy, block=False):
                success_count += 1
        
        if success_count > 0:
            message.status = MessageStatus.SENT
            self._stats['broadcast_messages_sent'] += 1
        
        return success_count == total_count
    
    def _route_multicast_message(self, message: EnhancedMessage) -> bool:
        """Route multicast message to specific agents"""
        if not message.receiver_ids:
            return False
        
        success_count = 0
        total_count = len(message.receiver_ids)
        
        for receiver_id in message.receiver_ids:
            if receiver_id not in self._message_queues:
                logger.warning(f"Receiver {receiver_id} not found for multicast message {message.id}")
                continue
            
            queue = self._message_queues[receiver_id]
            # Create copy for each receiver
            msg_copy = EnhancedMessage(
                id=str(uuid.uuid4()),
                sender_id=message.sender_id,
                receiver_id=receiver_id,
                content=message.content,
                message_type=message.message_type,
                priority=message.priority,
                protocol=message.protocol,
                metadata=message.metadata,
                timestamp=message.timestamp
            )
            
            if queue.put(msg_copy, block=False):
                success_count += 1
        
        if success_count > 0:
            message.status = MessageStatus.SENT
            self._stats['multicast_messages_sent'] += 1
        
        return success_count == total_count
    
    def _route_pubsub_message(self, message: EnhancedMessage) -> bool:
        """Route publish-subscribe message"""
        # Extract topic from message content
        topic = message.content.get('topic') if isinstance(message.content, dict) else 'default'
        
        if topic not in self._subscribers:
            logger.warning(f"No subscribers for topic {topic}")
            return True  # Not an error if no subscribers
        
        success_count = 0
        subscribers = list(self._subscribers[topic])
        
        for subscriber_id in subscribers:
            if subscriber_id == message.sender_id:  # Don't send to sender
                continue
                
            if subscriber_id not in self._message_queues:
                continue
            
            queue = self._message_queues[subscriber_id]
            # Create copy for each subscriber
            msg_copy = EnhancedMessage(
                id=str(uuid.uuid4()),
                sender_id=message.sender_id,
                receiver_id=subscriber_id,
                content=message.content,
                message_type=message.message_type,
                priority=message.priority,
                protocol=message.protocol,
                metadata=message.metadata,
                timestamp=message.timestamp
            )
            
            if queue.put(msg_copy, block=False):
                success_count += 1
        
        if success_count > 0:
            message.status = MessageStatus.SENT
            self._stats['pubsub_messages_sent'] += 1
        
        return True
    
    def _route_hierarchical_message(self, message: EnhancedMessage) -> bool:
        """Route hierarchical message following chain of command"""
        # This would implement hierarchical routing logic
        # For now, fall back to direct routing
        return self._route_direct_message(message)
    
    def receive_message(self, agent_id: AgentID, timeout: Optional[float] = None) -> Optional[EnhancedMessage]:
        """Receive message for an agent"""
        if agent_id not in self._message_queues:
            return None
        
        queue = self._message_queues[agent_id]
        message = queue.get(block=timeout is not None, timeout=timeout)
        
        if message:
            message.status = MessageStatus.DELIVERED
            self._stats['messages_delivered'] += 1
        
        return message
    
    def acknowledge_message(self, message_id: MessageID, agent_id: AgentID) -> bool:
        """Acknowledge message receipt"""
        if message_id in self._pending_acks:
            message = self._pending_acks[message_id]
            message.status = MessageStatus.ACKNOWLEDGED
            del self._pending_acks[message_id]
            self._stats['messages_acknowledged'] += 1
            logger.debug(f"Message {message_id} acknowledged by {agent_id}")
            return True
        
        return False
    
    def subscribe(self, agent_id: AgentID, topic: str):
        """Subscribe agent to topic"""
        self._subscribers[topic].add(agent_id)
        logger.info(f"Agent {agent_id} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_id: AgentID, topic: str):
        """Unsubscribe agent from topic"""
        self._subscribers[topic].discard(agent_id)
        logger.info(f"Agent {agent_id} unsubscribed from topic {topic}")
    
    def start(self):
        """Start the message broker"""
        if self._running:
            return
        
        self._running = True
        
        # Start retry worker
        retry_worker = threading.Thread(target=self._retry_worker, daemon=True)
        retry_worker.start()
        self._workers.append(retry_worker)
        
        # Start cleanup worker
        cleanup_worker = threading.Thread(target=self._cleanup_worker, daemon=True)
        cleanup_worker.start()
        self._workers.append(cleanup_worker)
        
        logger.info("Message broker started")
    
    def stop(self):
        """Stop the message broker"""
        self._running = False
        
        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)
        
        self._executor.shutdown(wait=True)
        logger.info("Message broker stopped")
    
    def _retry_worker(self):
        """Worker thread for handling message retries"""
        while self._running:
            try:
                # Check for failed messages that should be retried
                retry_messages = []
                current_time = time.time()
                
                for message_id, message in list(self._pending_acks.items()):
                    if message.should_retry():
                        # Add exponential backoff
                        retry_delay = min(2 ** message.metadata.retry_count, 60)
                        retry_time = current_time + retry_delay
                        heapq.heappush(self._retry_queue, (retry_time, message))
                        del self._pending_acks[message_id]
                
                # Process retry queue
                while self._retry_queue:
                    retry_time, message = heapq.heappop(self._retry_queue)
                    if retry_time <= current_time:
                        message.metadata.retry_count += 1
                        message.status = MessageStatus.RETRY
                        self.send_message(message)
                    else:
                        # Put back in queue for later
                        heapq.heappush(self._retry_queue, (retry_time, message))
                        break
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Retry worker error: {e}")
                time.sleep(5.0)
    
    def _cleanup_worker(self):
        """Worker thread for cleaning up expired messages"""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Clean up expired pending acknowledgments
                expired_acks = []
                for message_id, message in self._pending_acks.items():
                    if message.is_expired():
                        expired_acks.append(message_id)
                
                for message_id in expired_acks:
                    message = self._pending_acks[message_id]
                    message.status = MessageStatus.EXPIRED
                    del self._pending_acks[message_id]
                    self._stats['messages_expired'] += 1
                
                time.sleep(30.0)  # Clean up every 30 seconds
                
            except Exception as e:
                logger.error(f"Cleanup worker error: {e}")
                time.sleep(60.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics"""
        stats: Dict[str, Any] = dict(self._stats)
        stats['registered_agents'] = len(self._agents)
        stats['pending_acknowledgments'] = len(self._pending_acks)
        stats['retry_queue_size'] = len(self._retry_queue)
        stats['subscribers_by_topic'] = {
            topic: len(subscribers) 
            for topic, subscribers in self._subscribers.items()
        }
        
        # Add queue stats
        queue_stats = {}
        for agent_id, queue in self._message_queues.items():
            queue_stats[agent_id] = queue.get_stats()
        stats['queue_stats'] = queue_stats
        
        return stats


class CommunicationAgent(ABC):
    """Abstract base class for agents with communication capabilities"""
    
    def __init__(self, agent_id: AgentID, broker: Optional[MessageBroker] = None):
        self.agent_id = agent_id
        self.broker = broker
        self._running = False
        self._message_handlers: Dict[MessageType, Callable] = {}
        self._receive_worker: Optional[threading.Thread] = None
        
        # Register default handlers
        self._register_default_handlers()
        
        # Register with broker if provided
        if self.broker:
            self.broker.register_agent(self)
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self._message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self._message_handlers[MessageType.ACKNOWLEDGMENT] = self._handle_acknowledgment
        self._message_handlers[MessageType.STATUS] = self._handle_status
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for a specific message type"""
        self._message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type.value} messages")
    
    def send_message(
        self,
        content: Union[str, Dict, List, Any],
        receiver_id: Optional[AgentID] = None,
        receiver_ids: Optional[List[AgentID]] = None,
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        protocol: CommunicationProtocol = CommunicationProtocol.DIRECT,
        requires_ack: bool = False,
        ttl: Optional[timedelta] = None,
        **kwargs
    ) -> Optional[MessageID]:
        """Send a message through the broker"""
        if not self.broker:
            logger.error("No broker available for sending message")
            return None
        
        message = EnhancedMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            receiver_ids=receiver_ids,
            content=content,
            message_type=message_type,
            priority=priority,
            protocol=protocol,
            metadata=MessageMetadata(
                requires_ack=requires_ack,
                ttl=ttl,
                **kwargs
            )
        )
        
        success = self.broker.send_message(message)
        if success:
            logger.debug(f"Message {message.id} sent successfully")
            return message.id
        else:
            logger.error(f"Failed to send message {message.id}")
            return None
    
    def broadcast_message(
        self,
        content: Union[str, Dict, List, Any],
        message_type: MessageType = MessageType.BROADCAST,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> Optional[MessageID]:
        """Broadcast message to all agents"""
        return self.send_message(
            content=content,
            message_type=message_type,
            priority=priority,
            protocol=CommunicationProtocol.BROADCAST,
            **kwargs
        )
    
    def publish_message(
        self,
        topic: str,
        content: Union[str, Dict, List, Any],
        message_type: MessageType = MessageType.DIRECT,
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> Optional[MessageID]:
        """Publish message to topic"""
        publish_content = {
            'topic': topic,
            'data': content
        }
        
        return self.send_message(
            content=publish_content,
            message_type=message_type,
            priority=priority,
            protocol=CommunicationProtocol.PUBLISH_SUBSCRIBE,
            **kwargs
        )
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe to a topic"""
        if self.broker:
            self.broker.subscribe(self.agent_id, topic)
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from a topic"""
        if self.broker:
            self.broker.unsubscribe(self.agent_id, topic)
    
    def start_listening(self):
        """Start listening for incoming messages"""
        if self._running:
            return
        
        self._running = True
        self._receive_worker = threading.Thread(target=self._message_receiver, daemon=True)
        self._receive_worker.start()
        logger.info(f"Agent {self.agent_id} started listening for messages")
    
    def stop_listening(self):
        """Stop listening for incoming messages"""
        self._running = False
        if self._receive_worker:
            self._receive_worker.join(timeout=5.0)
        logger.info(f"Agent {self.agent_id} stopped listening for messages")
    
    def _message_receiver(self):
        """Worker thread for receiving and processing messages"""
        while self._running:
            try:
                if not self.broker:
                    time.sleep(1.0)
                    continue
                
                message = self.broker.receive_message(self.agent_id, timeout=1.0)
                if message:
                    self._process_message(message)
                    
            except Exception as e:
                logger.error(f"Error in message receiver for {self.agent_id}: {e}")
                time.sleep(1.0)
    
    def _process_message(self, message: EnhancedMessage):
        """Process received message"""
        try:
            start_time = time.time()
            
            # Check if handler exists for message type
            if message.message_type in self._message_handlers:
                handler = self._message_handlers[message.message_type]
                result = handler(message)
                
                # Send acknowledgment if required
                if message.metadata.requires_ack and self.broker:
                    self.broker.acknowledge_message(message.id, self.agent_id)
                
                # Update processing time
                message.processing_time = time.time() - start_time
                
                logger.debug(
                    f"Processed message {message.id} of type {message.message_type.value} "
                    f"in {message.processing_time:.3f}s"
                )
                
            else:
                logger.warning(
                    f"No handler for message type {message.message_type.value} "
                    f"in agent {self.agent_id}"
                )
                
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            message.error_details = str(e)
    
    def _handle_heartbeat(self, message: EnhancedMessage):
        """Handle heartbeat message"""
        logger.debug(f"Heartbeat received from {message.sender_id}")
        
        # Send heartbeat response
        response_content = {
            'status': 'alive',
            'timestamp': datetime.now().isoformat(),
            'agent_id': self.agent_id
        }
        
        self.send_message(
            content=response_content,
            receiver_id=message.sender_id,
            message_type=MessageType.STATUS,
            priority=MessagePriority.HIGH
        )
    
    def _handle_acknowledgment(self, message: EnhancedMessage):
        """Handle acknowledgment message"""
        logger.debug(f"Acknowledgment received from {message.sender_id}")
    
    def _handle_status(self, message: EnhancedMessage):
        """Handle status message"""
        logger.debug(f"Status received from {message.sender_id}: {message.content}")
    
    @abstractmethod
    def process_task_message(self, message: EnhancedMessage):
        """Process task message - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def process_response_message(self, message: EnhancedMessage):
        """Process response message - to be implemented by subclasses"""
        pass


# Global message broker instance
_global_broker = None

def get_global_broker() -> MessageBroker:
    """Get or create global message broker instance"""
    global _global_broker
    if _global_broker is None:
        _global_broker = MessageBroker()
    return _global_broker

def shutdown_global_broker():
    """Shutdown global message broker"""
    global _global_broker
    if _global_broker:
        _global_broker.stop()
        _global_broker = None