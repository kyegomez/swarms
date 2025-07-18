import json
import pickle
import hashlib
import threading
import time
from functools import lru_cache, wraps
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import weakref
from concurrent.futures import ThreadPoolExecutor
import os

from loguru import logger

# Import the Agent class - adjust path as needed
try:
    from swarms.structs.agent import Agent
except ImportError:
    # Fallback for development/testing
    Agent = Any


class AgentCache:
    """
    A comprehensive caching system for Agent objects with multiple strategies:
    - Memory-based LRU cache
    - Weak reference cache to prevent memory leaks
    - Persistent disk cache for agent configurations
    - Lazy loading with background preloading
    """

    def __init__(
        self,
        max_memory_cache_size: int = 128,
        cache_dir: Optional[str] = None,
        enable_persistent_cache: bool = True,
        auto_save_interval: int = 300,  # 5 minutes
        enable_weak_refs: bool = True,
    ):
        """
        Initialize the AgentCache.

        Args:
            max_memory_cache_size: Maximum number of agents to keep in memory cache
            cache_dir: Directory for persistent cache storage
            enable_persistent_cache: Whether to enable disk-based caching
            auto_save_interval: Interval in seconds for auto-saving cache
            enable_weak_refs: Whether to use weak references to prevent memory leaks
        """
        self.max_memory_cache_size = max_memory_cache_size
        self.cache_dir = Path(cache_dir or "agent_cache")
        self.enable_persistent_cache = enable_persistent_cache
        self.auto_save_interval = auto_save_interval
        self.enable_weak_refs = enable_weak_refs

        # Memory caches
        self._memory_cache: Dict[str, Agent] = {}
        self._weak_cache: weakref.WeakValueDictionary = (
            weakref.WeakValueDictionary()
        )
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()

        # Cache statistics
        self._hits = 0
        self._misses = 0
        self._load_times: Dict[str, float] = {}

        # Background tasks
        self._auto_save_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Initialize cache directory
        if self.enable_persistent_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Start auto-save thread
        self._start_auto_save_thread()

    def _start_auto_save_thread(self):
        """Start the auto-save background thread."""
        if (
            self.enable_persistent_cache
            and self.auto_save_interval > 0
        ):
            self._auto_save_thread = threading.Thread(
                target=self._auto_save_loop,
                daemon=True,
                name="AgentCache-AutoSave",
            )
            self._auto_save_thread.start()

    def _auto_save_loop(self):
        """Background loop for auto-saving cache."""
        while not self._shutdown_event.wait(self.auto_save_interval):
            try:
                self.save_cache_to_disk()
            except Exception as e:
                logger.error(f"Error in auto-save: {e}")

    def _generate_cache_key(
        self, agent_config: Dict[str, Any]
    ) -> str:
        """Generate a unique cache key from agent configuration."""
        # Create a stable hash from the configuration
        config_str = json.dumps(
            agent_config, sort_keys=True, default=str
        )
        return hashlib.md5(config_str.encode()).hexdigest()

    def _evict_lru(self):
        """Evict least recently used items from memory cache."""
        if len(self._memory_cache) >= self.max_memory_cache_size:
            # Find the least recently used item
            lru_key = min(
                self._access_times.items(), key=lambda x: x[1]
            )[0]

            # Save to persistent cache before evicting
            if self.enable_persistent_cache:
                self._save_agent_to_disk(
                    lru_key, self._memory_cache[lru_key]
                )

            # Remove from memory
            del self._memory_cache[lru_key]
            del self._access_times[lru_key]

            logger.debug(f"Evicted agent {lru_key} from memory cache")

    def _save_agent_to_disk(self, cache_key: str, agent: Agent):
        """Save agent to persistent cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(agent.to_dict(), f)
            logger.debug(f"Saved agent {cache_key} to disk cache")
        except Exception as e:
            logger.error(f"Error saving agent to disk: {e}")

    def _load_agent_from_disk(
        self, cache_key: str
    ) -> Optional[Agent]:
        """Load agent from persistent cache."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    agent_dict = pickle.load(f)

                # Reconstruct agent from dictionary
                agent = Agent(**agent_dict)
                logger.debug(
                    f"Loaded agent {cache_key} from disk cache"
                )
                return agent
        except Exception as e:
            logger.error(f"Error loading agent from disk: {e}")
        return None

    def get_agent(
        self, agent_config: Dict[str, Any]
    ) -> Optional[Agent]:
        """
        Get an agent from cache, loading if necessary.

        Args:
            agent_config: Configuration dictionary for the agent

        Returns:
            Cached or newly loaded Agent instance
        """
        cache_key = self._generate_cache_key(agent_config)

        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                self._access_times[cache_key] = time.time()
                self._hits += 1
                logger.debug(
                    f"Cache hit (memory) for agent {cache_key}"
                )
                return self._memory_cache[cache_key]

            # Check weak reference cache
            if (
                self.enable_weak_refs
                and cache_key in self._weak_cache
            ):
                agent = self._weak_cache[cache_key]
                if agent is not None:
                    # Move back to memory cache
                    self._memory_cache[cache_key] = agent
                    self._access_times[cache_key] = time.time()
                    self._hits += 1
                    logger.debug(
                        f"Cache hit (weak ref) for agent {cache_key}"
                    )
                    return agent

            # Check persistent cache
            if self.enable_persistent_cache:
                agent = self._load_agent_from_disk(cache_key)
                if agent is not None:
                    self._evict_lru()
                    self._memory_cache[cache_key] = agent
                    self._access_times[cache_key] = time.time()
                    if self.enable_weak_refs:
                        self._weak_cache[cache_key] = agent
                    self._hits += 1
                    logger.debug(
                        f"Cache hit (disk) for agent {cache_key}"
                    )
                    return agent

            # Cache miss - need to create new agent
            self._misses += 1
            logger.debug(f"Cache miss for agent {cache_key}")
            return None

    def put_agent(self, agent_config: Dict[str, Any], agent: Agent):
        """
        Put an agent into the cache.

        Args:
            agent_config: Configuration dictionary for the agent
            agent: The Agent instance to cache
        """
        cache_key = self._generate_cache_key(agent_config)

        with self._lock:
            self._evict_lru()
            self._memory_cache[cache_key] = agent
            self._access_times[cache_key] = time.time()

            if self.enable_weak_refs:
                self._weak_cache[cache_key] = agent

            logger.debug(f"Added agent {cache_key} to cache")

    def preload_agents(self, agent_configs: List[Dict[str, Any]]):
        """
        Preload agents in the background for faster access.

        Args:
            agent_configs: List of agent configurations to preload
        """

        def _preload_worker(config):
            try:
                cache_key = self._generate_cache_key(config)
                if cache_key not in self._memory_cache:
                    start_time = time.time()
                    agent = Agent(**config)
                    load_time = time.time() - start_time

                    self.put_agent(config, agent)
                    self._load_times[cache_key] = load_time
                    logger.debug(
                        f"Preloaded agent {cache_key} in {load_time:.3f}s"
                    )
            except Exception as e:
                logger.error(f"Error preloading agent: {e}")

        # Use thread pool for concurrent preloading
        max_workers = min(len(agent_configs), os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(_preload_worker, agent_configs)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._hits + self._misses
        hit_rate = (
            (self._hits / total_requests * 100)
            if total_requests > 0
            else 0
        )

        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "memory_cache_size": len(self._memory_cache),
            "weak_cache_size": len(self._weak_cache),
            "average_load_time": (
                sum(self._load_times.values()) / len(self._load_times)
                if self._load_times
                else 0
            ),
            "total_agents_loaded": len(self._load_times),
        }

    def clear_cache(self):
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            self._weak_cache.clear()
            self._access_times.clear()
            logger.info("Cleared all caches")

    def save_cache_to_disk(self):
        """Save current memory cache to disk."""
        if not self.enable_persistent_cache:
            return

        with self._lock:
            saved_count = 0
            for cache_key, agent in self._memory_cache.items():
                try:
                    self._save_agent_to_disk(cache_key, agent)
                    saved_count += 1
                except Exception as e:
                    logger.error(
                        f"Error saving agent {cache_key}: {e}"
                    )

            logger.info(f"Saved {saved_count} agents to disk cache")

    def shutdown(self):
        """Shutdown the cache system gracefully."""
        self._shutdown_event.set()
        if self._auto_save_thread:
            self._auto_save_thread.join(timeout=5)

        # Final save
        if self.enable_persistent_cache:
            self.save_cache_to_disk()

        logger.info("AgentCache shutdown complete")


# Global cache instance
_global_cache: Optional[AgentCache] = None


def get_global_cache() -> AgentCache:
    """Get or create the global agent cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AgentCache()
    return _global_cache


def cached_agent_loader(
    agents: List[Agent],
    cache_instance: Optional[AgentCache] = None,
    preload: bool = True,
    parallel_loading: bool = True,
) -> List[Agent]:
    """
    Load a list of agents with caching for super fast performance.

    Args:
        agents: List of Agent instances to cache/load
        cache_instance: Optional cache instance (uses global cache if None)
        preload: Whether to preload agents in background
        parallel_loading: Whether to load agents in parallel

    Returns:
        List of Agent instances (cached versions if available)

    Examples:
        # Basic usage
        agents = [Agent(agent_name="Agent1", model_name="gpt-4"), ...]
        cached_agents = cached_agent_loader(agents)

        # With custom cache
        cache = AgentCache(max_memory_cache_size=256)
        cached_agents = cached_agent_loader(agents, cache_instance=cache)

        # Preload for even faster subsequent access
        cached_agent_loader(agents, preload=True)
        cached_agents = cached_agent_loader(agents)  # Super fast!
    """
    cache = cache_instance or get_global_cache()

    start_time = time.time()

    # Extract configurations from agents for caching
    agent_configs = []
    for agent in agents:
        config = _extract_agent_config(agent)
        agent_configs.append(config)

    if preload:
        # Preload agents in background
        cache.preload_agents(agent_configs)

    def _load_single_agent(agent: Agent) -> Agent:
        """Load a single agent with caching."""
        config = _extract_agent_config(agent)

        # Try to get from cache first
        cached_agent = cache.get_agent(config)

        if cached_agent is None:
            # Cache miss - use the provided agent and cache it
            load_start = time.time()

            # Add to cache for future use
            cache.put_agent(config, agent)
            load_time = time.time() - load_start

            logger.debug(
                f"Cached new agent {agent.agent_name} in {load_time:.3f}s"
            )
            return agent
        else:
            logger.debug(
                f"Retrieved cached agent {cached_agent.agent_name}"
            )
            return cached_agent

    # Load agents (parallel or sequential)
    if parallel_loading and len(agents) > 1:
        max_workers = min(len(agents), os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            cached_agents = list(
                executor.map(_load_single_agent, agents)
            )
    else:
        cached_agents = [
            _load_single_agent(agent) for agent in agents
        ]

    total_time = time.time() - start_time

    # Log performance stats
    stats = cache.get_cache_stats()
    logger.info(
        f"Processed {len(cached_agents)} agents in {total_time:.3f}s "
        f"(Hit rate: {stats['hit_rate_percent']}%)"
    )

    return cached_agents


def _extract_agent_config(agent: Agent) -> Dict[str, Any]:
    """
    Extract a configuration dictionary from an Agent instance for caching.

    Args:
        agent: Agent instance to extract config from

    Returns:
        Configuration dictionary suitable for cache key generation
    """
    # Extract key attributes that define an agent's identity
    config = {
        "agent_name": getattr(agent, "agent_name", None),
        "model_name": getattr(agent, "model_name", None),
        "system_prompt": getattr(agent, "system_prompt", None),
        "max_loops": getattr(agent, "max_loops", None),
        "temperature": getattr(agent, "temperature", None),
        "max_tokens": getattr(agent, "max_tokens", None),
        "agent_description": getattr(
            agent, "agent_description", None
        ),
        # Add other key identifying attributes
        "tools": str(
            getattr(agent, "tools", [])
        ),  # Convert to string for hashing, default to empty list
        "context_length": getattr(agent, "context_length", None),
    }

    # Remove None values to create a clean config
    config = {k: v for k, v in config.items() if v is not None}

    return config


def cached_agent_loader_from_configs(
    agent_configs: List[Dict[str, Any]],
    cache_instance: Optional[AgentCache] = None,
    preload: bool = True,
    parallel_loading: bool = True,
) -> List[Agent]:
    """
    Load a list of agents from configuration dictionaries with caching.

    Args:
        agent_configs: List of agent configuration dictionaries
        cache_instance: Optional cache instance (uses global cache if None)
        preload: Whether to preload agents in background
        parallel_loading: Whether to load agents in parallel

    Returns:
        List of Agent instances

    Examples:
        # Basic usage
        configs = [{"agent_name": "Agent1", "model_name": "gpt-4"}, ...]
        agents = cached_agent_loader_from_configs(configs)

        # With custom cache
        cache = AgentCache(max_memory_cache_size=256)
        agents = cached_agent_loader_from_configs(configs, cache_instance=cache)
    """
    cache = cache_instance or get_global_cache()

    start_time = time.time()

    if preload:
        # Preload agents in background
        cache.preload_agents(agent_configs)

    def _load_single_agent(config: Dict[str, Any]) -> Agent:
        """Load a single agent with caching."""
        # Try to get from cache first
        agent = cache.get_agent(config)

        if agent is None:
            # Cache miss - create new agent
            load_start = time.time()
            agent = Agent(**config)
            load_time = time.time() - load_start

            # Add to cache for future use
            cache.put_agent(config, agent)

            logger.debug(
                f"Created new agent {agent.agent_name} in {load_time:.3f}s"
            )

        return agent

    # Load agents (parallel or sequential)
    if parallel_loading and len(agent_configs) > 1:
        max_workers = min(len(agent_configs), os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            agents = list(
                executor.map(_load_single_agent, agent_configs)
            )
    else:
        agents = [
            _load_single_agent(config) for config in agent_configs
        ]

    total_time = time.time() - start_time

    # Log performance stats
    stats = cache.get_cache_stats()
    logger.info(
        f"Loaded {len(agents)} agents in {total_time:.3f}s "
        f"(Hit rate: {stats['hit_rate_percent']}%)"
    )

    return agents


# Decorator for caching individual agent creation
def cache_agent_creation(cache_instance: Optional[AgentCache] = None):
    """
    Decorator to cache agent creation based on initialization parameters.

    Args:
        cache_instance: Optional cache instance (uses global cache if None)

    Returns:
        Decorator function

    Example:
        @cache_agent_creation()
        def create_trading_agent(symbol: str, model: str):
            return Agent(
                agent_name=f"Trading-{symbol}",
                model_name=model,
                system_prompt=f"You are a trading agent for {symbol}"
            )

        agent1 = create_trading_agent("AAPL", "gpt-4")  # Creates new agent
        agent2 = create_trading_agent("AAPL", "gpt-4")  # Returns cached agent
    """

    def decorator(func: Callable[..., Agent]) -> Callable[..., Agent]:
        cache = cache_instance or get_global_cache()

        @wraps(func)
        def wrapper(*args, **kwargs) -> Agent:
            # Create a config dict from function arguments
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            config = dict(bound_args.arguments)

            # Try to get from cache
            agent = cache.get_agent(config)

            if agent is None:
                # Cache miss - call original function
                agent = func(*args, **kwargs)
                cache.put_agent(config, agent)

            return agent

        return wrapper

    return decorator


# LRU Cache-based simple approach
@lru_cache(maxsize=128)
def _cached_agent_by_hash(
    config_hash: str, config_json: str
) -> Agent:
    """Internal LRU cached agent creation by config hash."""
    config = json.loads(config_json)
    return Agent(**config)


def simple_lru_agent_loader(
    agents: List[Agent],
) -> List[Agent]:
    """
    Simple LRU cache-based agent loader using functools.lru_cache.

    Args:
        agents: List of Agent instances

    Returns:
        List of Agent instances (cached versions if available)

    Note:
        This is a simpler approach but less flexible than the full AgentCache.
    """
    cached_agents = []

    for agent in agents:
        # Extract config from agent
        config = _extract_agent_config(agent)

        # Create stable hash and JSON string
        config_json = json.dumps(config, sort_keys=True, default=str)
        config_hash = hashlib.md5(config_json.encode()).hexdigest()

        # Use LRU cached function
        cached_agent = _cached_agent_by_hash_from_agent(
            config_hash, agent
        )
        cached_agents.append(cached_agent)

    return cached_agents


@lru_cache(maxsize=128)
def _cached_agent_by_hash_from_agent(
    config_hash: str, agent: Agent
) -> Agent:
    """Internal LRU cached agent storage by config hash."""
    # Return the same agent instance (this creates the caching effect)
    return agent


# Utility functions for cache management
def clear_agent_cache():
    """Clear the global agent cache."""
    cache = get_global_cache()
    cache.clear_cache()


def get_agent_cache_stats() -> Dict[str, Any]:
    """Get statistics from the global agent cache."""
    cache = get_global_cache()
    return cache.get_cache_stats()


def shutdown_agent_cache():
    """Shutdown the global agent cache gracefully."""
    global _global_cache
    if _global_cache:
        _global_cache.shutdown()
        _global_cache = None
