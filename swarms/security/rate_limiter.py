"""
Rate limiting and abuse prevention for Swarms framework.

This module provides comprehensive rate limiting, request tracking,
and abuse prevention for all swarm operations.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

from loguru import logger

from swarms.utils.loguru_logger import initialize_logger

# Initialize logger for rate limiting
rate_logger = initialize_logger(log_folder="rate_limiting")


class RateLimiter:
    """
    Rate limiting and abuse prevention for swarm security
    
    Features:
    - Per-agent rate limiting
    - Token-based limiting
    - Request tracking
    - Abuse detection
    - Automatic blocking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize rate limiter with configuration
        
        Args:
            config: Rate limiting configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        self.max_requests_per_minute = config.get("max_requests_per_minute", 60)
        self.max_tokens_per_request = config.get("max_tokens_per_request", 10000)
        self.window = config.get("window", 60)  # seconds
        
        # Request tracking
        self._request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self._token_usage: Dict[str, int] = defaultdict(int)
        self._blocked_agents: Dict[str, float] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = False
        
        if self.enabled:
            self._start_cleanup_thread()
        
        rate_logger.info(f"RateLimiter initialized: {self.max_requests_per_minute} req/min, {self.max_tokens_per_request} tokens/req")
    
    def check_rate_limit(self, agent_name: str, request_size: int = 1) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits
        
        Args:
            agent_name: Name of the agent making the request
            request_size: Size of the request (tokens or complexity)
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        if not self.enabled:
            return True, None
        
        try:
            with self._lock:
                current_time = time.time()
                
                # Check if agent is blocked
                if agent_name in self._blocked_agents:
                    block_until = self._blocked_agents[agent_name]
                    if current_time < block_until:
                        remaining = block_until - current_time
                        return False, f"Agent {agent_name} is blocked for {remaining:.1f} more seconds"
                    else:
                        # Unblock agent
                        del self._blocked_agents[agent_name]
                
                # Check token limit
                if request_size > self.max_tokens_per_request:
                    return False, f"Request size {request_size} exceeds token limit {self.max_tokens_per_request}"
                
                # Get agent's request history
                history = self._request_history[agent_name]
                
                # Remove old requests outside the window
                cutoff_time = current_time - self.window
                while history and history[0] < cutoff_time:
                    history.popleft()
                
                # Check request count limit
                if len(history) >= self.max_requests_per_minute:
                    # Block agent temporarily
                    block_duration = min(300, self.window * 2)  # Max 5 minutes
                    self._blocked_agents[agent_name] = current_time + block_duration
                    
                    rate_logger.warning(f"Agent {agent_name} rate limit exceeded, blocked for {block_duration}s")
                    return False, f"Rate limit exceeded. Agent blocked for {block_duration} seconds"
                
                # Add current request
                history.append(current_time)
                
                # Update token usage
                self._token_usage[agent_name] += request_size
                
                rate_logger.debug(f"Rate limit check passed for {agent_name}")
                return True, None
                
        except Exception as e:
            rate_logger.error(f"Rate limit check error: {e}")
            return False, f"Rate limit check error: {str(e)}"
    
    def check_agent_limit(self, agent_name: str) -> Tuple[bool, Optional[str]]:
        """Check agent-specific rate limits"""
        return self.check_rate_limit(agent_name, 1)
    
    def check_token_limit(self, agent_name: str, token_count: int) -> Tuple[bool, Optional[str]]:
        """Check token-based rate limits"""
        return self.check_rate_limit(agent_name, token_count)
    
    def track_request(self, agent_name: str, request_type: str = "general", metadata: Dict[str, Any] = None) -> None:
        """
        Track a request for monitoring purposes
        
        Args:
            agent_name: Name of the agent
            request_type: Type of request
            metadata: Additional request metadata
        """
        if not self.enabled:
            return
        
        try:
            with self._lock:
                current_time = time.time()
                
                # Store request metadata
                request_info = {
                    "timestamp": current_time,
                    "type": request_type,
                    "metadata": metadata or {}
                }
                
                # Add to history (we only store timestamps for rate limiting)
                self._request_history[agent_name].append(current_time)
                
                rate_logger.debug(f"Tracked request: {agent_name} - {request_type}")
                
        except Exception as e:
            rate_logger.error(f"Request tracking error: {e}")
    
    def get_agent_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get rate limiting statistics for an agent"""
        try:
            with self._lock:
                current_time = time.time()
                history = self._request_history[agent_name]
                
                # Clean old requests
                cutoff_time = current_time - self.window
                while history and history[0] < cutoff_time:
                    history.popleft()
                
                # Calculate statistics
                recent_requests = len(history)
                total_tokens = self._token_usage.get(agent_name, 0)
                is_blocked = agent_name in self._blocked_agents
                block_remaining = 0
                
                if is_blocked:
                    block_remaining = self._blocked_agents[agent_name] - current_time
                
                return {
                    "agent_name": agent_name,
                    "recent_requests": recent_requests,
                    "max_requests": self.max_requests_per_minute,
                    "total_tokens": total_tokens,
                    "max_tokens_per_request": self.max_tokens_per_request,
                    "is_blocked": is_blocked,
                    "block_remaining": max(0, block_remaining),
                    "window_seconds": self.window,
                }
                
        except Exception as e:
            rate_logger.error(f"Error getting agent stats: {e}")
            return {}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics for all agents"""
        try:
            with self._lock:
                stats = {
                    "enabled": self.enabled,
                    "max_requests_per_minute": self.max_requests_per_minute,
                    "max_tokens_per_request": self.max_tokens_per_request,
                    "window_seconds": self.window,
                    "total_agents": len(self._request_history),
                    "blocked_agents": len(self._blocked_agents),
                    "agents": {}
                }
                
                for agent_name in self._request_history:
                    stats["agents"][agent_name] = self.get_agent_stats(agent_name)
                
                return stats
                
        except Exception as e:
            rate_logger.error(f"Error getting all stats: {e}")
            return {}
    
    def reset_agent(self, agent_name: str) -> bool:
        """Reset rate limiting for an agent"""
        try:
            with self._lock:
                if agent_name in self._request_history:
                    self._request_history[agent_name].clear()
                
                if agent_name in self._token_usage:
                    self._token_usage[agent_name] = 0
                
                if agent_name in self._blocked_agents:
                    del self._blocked_agents[agent_name]
                
                rate_logger.info(f"Reset rate limiting for agent: {agent_name}")
                return True
                
        except Exception as e:
            rate_logger.error(f"Error resetting agent: {e}")
            return False
    
    def block_agent(self, agent_name: str, duration: int = 300) -> bool:
        """Manually block an agent"""
        try:
            with self._lock:
                current_time = time.time()
                self._blocked_agents[agent_name] = current_time + duration
                
                rate_logger.warning(f"Manually blocked agent {agent_name} for {duration}s")
                return True
                
        except Exception as e:
            rate_logger.error(f"Error blocking agent: {e}")
            return False
    
    def unblock_agent(self, agent_name: str) -> bool:
        """Unblock an agent"""
        try:
            with self._lock:
                if agent_name in self._blocked_agents:
                    del self._blocked_agents[agent_name]
                    rate_logger.info(f"Unblocked agent: {agent_name}")
                    return True
                return False
                
        except Exception as e:
            rate_logger.error(f"Error unblocking agent: {e}")
            return False
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self._stop_cleanup:
                try:
                    time.sleep(60)  # Cleanup every minute
                    self._cleanup_old_data()
                except Exception as e:
                    rate_logger.error(f"Cleanup thread error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_data(self) -> None:
        """Clean up old rate limiting data"""
        try:
            with self._lock:
                current_time = time.time()
                cutoff_time = current_time - (self.window * 2)  # Keep 2x window
                
                # Clean request history
                for agent_name in list(self._request_history.keys()):
                    history = self._request_history[agent_name]
                    while history and history[0] < cutoff_time:
                        history.popleft()
                    
                    # Remove empty histories
                    if not history:
                        del self._request_history[agent_name]
                
                # Clean blocked agents
                for agent_name in list(self._blocked_agents.keys()):
                    if self._blocked_agents[agent_name] < current_time:
                        del self._blocked_agents[agent_name]
                
                # Reset token usage periodically
                if current_time % 3600 < 60:  # Reset every hour
                    self._token_usage.clear()
                
        except Exception as e:
            rate_logger.error(f"Cleanup error: {e}")
    
    def stop(self) -> None:
        """Stop the rate limiter and cleanup thread"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        
        rate_logger.info("RateLimiter stopped")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop() 