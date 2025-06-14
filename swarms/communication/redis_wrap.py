import datetime
import hashlib
import json
import threading
import subprocess
import tempfile
import os
import atexit
import time
from typing import Any, Dict, List, Optional, Union

import yaml

from loguru import logger

from swarms.structs.base_structure import BaseStructure
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
from swarms.utils.litellm_tokenizer import count_tokens

# Module-level variable to track Redis availability
REDIS_AVAILABLE = False

# Try to import Redis and set availability flag
try:
    import redis
    from redis.exceptions import (
        AuthenticationError,
        BusyLoadingError,
        ConnectionError,
        RedisError,
        TimeoutError,
    )

    REDIS_AVAILABLE = True
except ImportError:
    # Auto-install Redis at import time
    print("📦 Redis not found. Installing automatically...")
    try:
        import subprocess
        import sys

        # Install redis
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "redis"]
        )
        print("✅ Redis installed successfully!")

        # Try importing again
        import redis
        from redis.exceptions import (
            AuthenticationError,
            BusyLoadingError,
            ConnectionError,
            RedisError,
            TimeoutError,
        )

        REDIS_AVAILABLE = True
        print("✅ Redis loaded successfully!")

    except Exception as e:
        REDIS_AVAILABLE = False
        print(
            f"❌ Failed to auto-install Redis. Please install manually with 'pip install redis': {e}"
        )


class RedisConnectionError(Exception):
    """Custom exception for Redis connection errors."""

    pass


class RedisOperationError(Exception):
    """Custom exception for Redis operation errors."""

    pass


class EmbeddedRedisServer:
    """Embedded Redis server manager"""

    def __init__(
        self,
        port: int = 6379,
        data_dir: str = None,
        persist: bool = True,
        auto_persist: bool = True,
    ):
        self.port = port
        self.process = None
        self.data_dir = data_dir or os.path.expanduser(
            "~/.swarms/redis"
        )
        self.persist = persist
        self.auto_persist = auto_persist

        # Only create data directory if persistence is enabled
        if self.persist and self.auto_persist:
            os.makedirs(self.data_dir, exist_ok=True)
            # Create Redis configuration file
            self._create_redis_config()

        atexit.register(self.stop)

    def _create_redis_config(self):
        """Create Redis configuration file with persistence settings"""
        config_path = os.path.join(self.data_dir, "redis.conf")
        config_content = f"""
port {self.port}
dir {self.data_dir}
dbfilename dump.rdb
appendonly yes
appendfilename appendonly.aof
appendfsync everysec
save 1 1
rdbcompression yes
rdbchecksum yes
"""
        with open(config_path, "w") as f:
            f.write(config_content)
        logger.info(f"Created Redis configuration at {config_path}")

    def start(self) -> bool:
        """Start the Redis server

        Returns:
            bool: True if server started successfully, False otherwise
        """
        try:
            # Check if Redis is available
            if not REDIS_AVAILABLE:
                logger.error("Redis package is not installed")
                return False

            # Use data directory if persistence is enabled and auto_persist is True
            if not (self.persist and self.auto_persist):
                self.data_dir = tempfile.mkdtemp()
                self._create_redis_config()  # Create config even for temporary dir

            config_path = os.path.join(self.data_dir, "redis.conf")

            # Start Redis server with config file
            redis_args = [
                "redis-server",
                config_path,
                "--daemonize",
                "no",
            ]

            # Start Redis server
            self.process = subprocess.Popen(
                redis_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for Redis to start
            time.sleep(1)
            if self.process.poll() is not None:
                stderr = self.process.stderr.read().decode()
                raise Exception(f"Redis failed to start: {stderr}")

            # Test connection
            try:
                r = redis.Redis(host="localhost", port=self.port)
                r.ping()
                r.close()
            except redis.ConnectionError as e:
                raise Exception(
                    f"Could not connect to Redis: {str(e)}"
                )

            logger.info(
                f"Started {'persistent' if (self.persist and self.auto_persist) else 'temporary'} Redis server on port {self.port}"
            )
            if self.persist and self.auto_persist:
                logger.info(f"Redis data directory: {self.data_dir}")
            return True
        except Exception as e:
            logger.error(
                f"Failed to start embedded Redis server: {str(e)}"
            )
            self.stop()
            return False

    def stop(self):
        """Stop the Redis server and cleanup resources"""
        try:
            if self.process:
                # Send SAVE and BGSAVE commands before stopping if persistence is enabled
                if (
                    self.persist
                    and self.auto_persist
                    and REDIS_AVAILABLE
                ):
                    try:
                        r = redis.Redis(
                            host="localhost", port=self.port
                        )
                        r.save()  # Synchronous save
                        r.bgsave()  # Asynchronous save
                        time.sleep(
                            1
                        )  # Give time for background save to complete
                        r.close()
                    except Exception as e:
                        logger.warning(
                            f"Error during Redis save: {str(e)}"
                        )

                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
                logger.info("Stopped Redis server")

            # Only remove directory if not persisting or auto_persist is False
            if (
                (not self.persist or not self.auto_persist)
                and self.data_dir
                and os.path.exists(self.data_dir)
            ):
                import shutil

                shutil.rmtree(self.data_dir)
                self.data_dir = None
        except Exception as e:
            logger.error(f"Error stopping Redis server: {str(e)}")


class RedisConversation(BaseStructure):
    """
    A Redis-based implementation of the Conversation class for managing conversation history.
    This class provides the same interface as the memory-based Conversation class but uses
    Redis as the storage backend.

    Attributes:
        system_prompt (Optional[str]): The system prompt for the conversation.
        time_enabled (bool): Flag to enable time tracking for messages.
        autosave (bool): Flag to enable automatic saving of conversation history.
        save_filepath (str): File path for saving the conversation history.
        tokenizer (Any): Tokenizer for counting tokens in messages.
        context_length (int): Maximum number of tokens allowed in the conversation history.
        rules (str): Rules for the conversation.
        custom_rules_prompt (str): Custom prompt for rules.
        user (str): The user identifier for messages.
        auto_save (bool): Flag to enable auto-saving of conversation history.
        save_as_yaml (bool): Flag to save conversation history as YAML.
        save_as_json_bool (bool): Flag to save conversation history as JSON.
        token_count (bool): Flag to enable token counting for messages.
        cache_enabled (bool): Flag to enable prompt caching.
        cache_stats (dict): Statistics about cache usage.
        cache_lock (threading.Lock): Lock for thread-safe cache operations.
        redis_client (redis.Redis): Redis client instance.
        conversation_id (str): Unique identifier for the current conversation.
    """

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        time_enabled: bool = False,
        autosave: bool = False,
        save_filepath: str = None,
        tokenizer: Any = None,
        context_length: int = 8192,
        rules: str = None,
        custom_rules_prompt: str = None,
        user: str = "User:",
        auto_save: bool = True,
        save_as_yaml: bool = True,
        save_as_json_bool: bool = False,
        token_count: bool = True,
        cache_enabled: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        redis_ssl: bool = False,
        redis_retry_attempts: int = 3,
        redis_retry_delay: float = 1.0,
        use_embedded_redis: bool = True,
        persist_redis: bool = True,
        auto_persist: bool = True,
        redis_data_dir: Optional[str] = None,
        conversation_id: Optional[str] = None,
        name: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the RedisConversation with Redis backend.

        Args:
            system_prompt (Optional[str]): The system prompt for the conversation.
            time_enabled (bool): Flag to enable time tracking for messages.
            autosave (bool): Flag to enable automatic saving of conversation history.
            save_filepath (str): File path for saving the conversation history.
            tokenizer (Any): Tokenizer for counting tokens in messages.
            context_length (int): Maximum number of tokens allowed in the conversation history.
            rules (str): Rules for the conversation.
            custom_rules_prompt (str): Custom prompt for rules.
            user (str): The user identifier for messages.
            auto_save (bool): Flag to enable auto-saving of conversation history.
            save_as_yaml (bool): Flag to save conversation history as YAML.
            save_as_json_bool (bool): Flag to save conversation history as JSON.
            token_count (bool): Flag to enable token counting for messages.
            cache_enabled (bool): Flag to enable prompt caching.
            redis_host (str): Redis server host.
            redis_port (int): Redis server port.
            redis_db (int): Redis database number.
            redis_password (Optional[str]): Redis password for authentication.
            redis_ssl (bool): Whether to use SSL for Redis connection.
            redis_retry_attempts (int): Number of connection retry attempts.
            redis_retry_delay (float): Delay between retry attempts in seconds.
            use_embedded_redis (bool): Whether to start an embedded Redis server.
                If True, redis_host and redis_port will be used for the embedded server.
            persist_redis (bool): Whether to enable Redis persistence.
            auto_persist (bool): Whether to automatically handle persistence.
                If True, persistence will be managed automatically.
                If False, persistence will be manual even if persist_redis is True.
            redis_data_dir (Optional[str]): Directory for Redis data persistence.
            conversation_id (Optional[str]): Specific conversation ID to use/restore.
                If None, a new ID will be generated.
            name (Optional[str]): A friendly name for the conversation.
                If provided, this will be used to look up or create a conversation.
                Takes precedence over conversation_id if both are provided.

        Raises:
            ImportError: If Redis package is not installed.
            RedisConnectionError: If connection to Redis fails.
            RedisOperationError: If Redis operations fail.
        """
        global REDIS_AVAILABLE

        # Check if Redis is available (should be True after module import auto-installation)
        if not REDIS_AVAILABLE:
            raise ImportError(
                "Redis is not available. Module-level auto-installation failed. "
                "Please install manually with 'pip install redis'"
            )

        self.redis_available = True

        super().__init__()
        self.system_prompt = system_prompt
        self.time_enabled = time_enabled
        self.autosave = autosave
        self.save_filepath = save_filepath
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.rules = rules
        self.custom_rules_prompt = custom_rules_prompt
        self.user = user
        self.auto_save = auto_save
        self.save_as_yaml = save_as_yaml
        self.save_as_json_bool = save_as_json_bool
        self.token_count = token_count
        self.cache_enabled = cache_enabled
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        self.cache_lock = threading.Lock()

        # Initialize Redis server (embedded or external)
        self.embedded_server = None
        if use_embedded_redis:
            self.embedded_server = EmbeddedRedisServer(
                port=redis_port,
                data_dir=redis_data_dir,
                persist=persist_redis,
                auto_persist=auto_persist,
            )
            if not self.embedded_server.start():
                raise RedisConnectionError(
                    "Failed to start embedded Redis server"
                )

        # Initialize Redis client with retries
        self.redis_client = None
        self._initialize_redis_connection(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            ssl=redis_ssl,
            retry_attempts=redis_retry_attempts,
            retry_delay=redis_retry_delay,
        )

        # Handle conversation name and ID
        self.name = name
        if name:
            # Try to find existing conversation by name
            existing_id = self._get_conversation_id_by_name(name)
            if existing_id:
                self.conversation_id = existing_id
                logger.info(
                    f"Found existing conversation '{name}' with ID: {self.conversation_id}"
                )
            else:
                # Create new conversation with name
                self.conversation_id = f"conversation:{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                self._save_conversation_name(name)
                logger.info(
                    f"Created new conversation '{name}' with ID: {self.conversation_id}"
                )
        else:
            # Use provided ID or generate new one
            self.conversation_id = (
                conversation_id
                or f"conversation:{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            )
            logger.info(
                f"Using conversation ID: {self.conversation_id}"
            )

        # Check if we have existing data
        has_existing_data = self._load_existing_data()

        if has_existing_data:
            logger.info(
                f"Restored conversation data for: {self.name or self.conversation_id}"
            )
        else:
            logger.info(
                f"Initialized new conversation: {self.name or self.conversation_id}"
            )
            # Initialize with prompts only for new conversations
            try:
                if self.system_prompt is not None:
                    self.add("System", self.system_prompt)

                if self.rules is not None:
                    self.add("User", rules)

                if custom_rules_prompt is not None:
                    self.add(user or "User", custom_rules_prompt)
            except RedisError as e:
                logger.error(
                    f"Failed to initialize conversation: {str(e)}"
                )
                raise RedisOperationError(
                    f"Failed to initialize conversation: {str(e)}"
                )

    def _initialize_redis_connection(
        self,
        host: str,
        port: int,
        db: int,
        password: Optional[str],
        ssl: bool,
        retry_attempts: int,
        retry_delay: float,
    ):
        """Initialize Redis connection with retry mechanism.

        Args:
            host (str): Redis host.
            port (int): Redis port.
            db (int): Redis database number.
            password (Optional[str]): Redis password.
            ssl (bool): Whether to use SSL.
            retry_attempts (int): Number of retry attempts.
            retry_delay (float): Delay between retries in seconds.

        Raises:
            RedisConnectionError: If connection fails after all retries.
        """
        import time

        for attempt in range(retry_attempts):
            try:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    ssl=ssl,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                )
                # Test connection and load data
                self.redis_client.ping()

                # Try to load the RDB file if it exists
                try:
                    self.redis_client.config_set(
                        "dbfilename", "dump.rdb"
                    )
                    self.redis_client.config_set(
                        "dir", os.path.expanduser("~/.swarms/redis")
                    )
                except redis.ResponseError:
                    pass  # Ignore if config set fails

                logger.info(
                    f"Successfully connected to Redis at {host}:{port}"
                )
                return
            except (
                ConnectionError,
                TimeoutError,
                AuthenticationError,
                BusyLoadingError,
            ) as e:
                if attempt < retry_attempts - 1:
                    logger.warning(
                        f"Redis connection attempt {attempt + 1} failed: {str(e)}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f"Failed to connect to Redis after {retry_attempts} attempts"
                    )
                    raise RedisConnectionError(
                        f"Failed to connect to Redis: {str(e)}"
                    )

    def _load_existing_data(self):
        """Load existing data for a conversation ID if it exists"""
        try:
            # Check if conversation exists
            message_ids = self.redis_client.lrange(
                f"{self.conversation_id}:message_ids", 0, -1
            )
            if message_ids:
                logger.info(
                    f"Found existing data for conversation {self.conversation_id}"
                )
                return True
            return False
        except Exception as e:
            logger.warning(
                f"Error checking for existing data: {str(e)}"
            )
            return False

    def _safe_redis_operation(
        self,
        operation_name: str,
        operation_func: callable,
        *args,
        **kwargs,
    ):
        """Execute Redis operation safely with error handling and logging.

        Args:
            operation_name (str): Name of the operation for logging.
            operation_func (callable): Function to execute.
            *args: Arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: Result of the operation.

        Raises:
            RedisOperationError: If the operation fails.
        """
        try:
            return operation_func(*args, **kwargs)
        except RedisError as e:
            error_msg = (
                f"Redis operation '{operation_name}' failed: {str(e)}"
            )
            logger.error(error_msg)
            raise RedisOperationError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error during Redis operation '{operation_name}': {str(e)}"
            logger.error(error_msg)
            raise

    def _generate_cache_key(
        self, content: Union[str, dict, list]
    ) -> str:
        """Generate a cache key for the given content.

        Args:
            content (Union[str, dict, list]): The content to generate a cache key for.

        Returns:
            str: The cache key.
        """
        try:
            if isinstance(content, (dict, list)):
                content = json.dumps(content, sort_keys=True)
            return hashlib.md5(str(content).encode()).hexdigest()
        except Exception as e:
            logger.error(f"Failed to generate cache key: {str(e)}")
            return hashlib.md5(
                str(datetime.datetime.now()).encode()
            ).hexdigest()

    def _get_cached_tokens(
        self, content: Union[str, dict, list]
    ) -> Optional[int]:
        """Get the number of cached tokens for the given content.

        Args:
            content (Union[str, dict, list]): The content to check.

        Returns:
            Optional[int]: The number of cached tokens, or None if not cached.
        """
        if not self.cache_enabled:
            return None

        with self.cache_lock:
            try:
                cache_key = self._generate_cache_key(content)
                cached_value = self._safe_redis_operation(
                    "get_cached_tokens",
                    self.redis_client.hget,
                    f"{self.conversation_id}:cache",
                    cache_key,
                )
                if cached_value:
                    self.cache_stats["hits"] += 1
                    return int(cached_value)
                self.cache_stats["misses"] += 1
                return None
            except Exception as e:
                logger.warning(
                    f"Failed to get cached tokens: {str(e)}"
                )
                return None

    def _update_cache_stats(
        self, content: Union[str, dict, list], token_count: int
    ):
        """Update cache statistics for the given content.

        Args:
            content (Union[str, dict, list]): The content to update stats for.
            token_count (int): The number of tokens in the content.
        """
        if not self.cache_enabled:
            return

        with self.cache_lock:
            try:
                cache_key = self._generate_cache_key(content)
                self._safe_redis_operation(
                    "update_cache",
                    self.redis_client.hset,
                    f"{self.conversation_id}:cache",
                    cache_key,
                    token_count,
                )
                self.cache_stats["cached_tokens"] += token_count
                self.cache_stats["total_tokens"] += token_count
            except Exception as e:
                logger.warning(
                    f"Failed to update cache stats: {str(e)}"
                )

    def add(
        self,
        role: str,
        content: Union[str, dict, list],
        *args,
        **kwargs,
    ):
        """Add a message to the conversation history.

        Args:
            role (str): The role of the speaker (e.g., 'User', 'System').
            content (Union[str, dict, list]): The content of the message.

        Raises:
            RedisOperationError: If the operation fails.
        """
        try:
            message = {
                "role": role,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            if isinstance(content, (dict, list)):
                message["content"] = json.dumps(content)
            elif self.time_enabled:
                message["content"] = (
                    f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \n {content}"
                )
            else:
                message["content"] = str(content)

            # Check cache for token count
            cached_tokens = self._get_cached_tokens(content)
            if cached_tokens is not None:
                message["token_count"] = cached_tokens
                message["cached"] = "true"
            else:
                message["cached"] = "false"

            # Add message to Redis
            message_id = self._safe_redis_operation(
                "increment_counter",
                self.redis_client.incr,
                f"{self.conversation_id}:message_counter",
            )

            self._safe_redis_operation(
                "store_message",
                self.redis_client.hset,
                f"{self.conversation_id}:message:{message_id}",
                mapping=message,
            )

            self._safe_redis_operation(
                "append_message_id",
                self.redis_client.rpush,
                f"{self.conversation_id}:message_ids",
                message_id,
            )

            if (
                self.token_count is True
                and message["cached"] == "false"
            ):
                self._count_tokens(content, message, message_id)

            logger.debug(
                f"Added message with ID {message_id} to conversation {self.conversation_id}"
            )
        except Exception as e:
            error_msg = f"Failed to add message: {str(e)}"
            logger.error(error_msg)
            raise RedisOperationError(error_msg)

    def _count_tokens(
        self, content: str, message: dict, message_id: int
    ):
        """Count tokens for a message in a separate thread.

        Args:
            content (str): The content to count tokens for.
            message (dict): The message dictionary.
            message_id (int): The ID of the message in Redis.
        """

        def count_tokens_thread():
            try:
                tokens = count_tokens(any_to_str(content))
                message["token_count"] = int(tokens)

                # Update the message in Redis
                self._safe_redis_operation(
                    "update_token_count",
                    self.redis_client.hset,
                    f"{self.conversation_id}:message:{message_id}",
                    "token_count",
                    int(tokens),
                )

                # Update cache stats
                self._update_cache_stats(content, int(tokens))

                if self.autosave and self.save_filepath:
                    self.save_as_json(self.save_filepath)

                logger.debug(
                    f"Updated token count for message {message_id}: {tokens} tokens"
                )
            except Exception as e:
                logger.error(
                    f"Failed to count tokens for message {message_id}: {str(e)}"
                )

        token_thread = threading.Thread(target=count_tokens_thread)
        token_thread.daemon = True
        token_thread.start()

    def delete(self, index: int):
        """Delete a message from the conversation history.

        Args:
            index (int): Index of the message to delete.

        Raises:
            RedisOperationError: If the operation fails.
            ValueError: If the index is invalid.
        """
        try:
            message_ids = self._safe_redis_operation(
                "get_message_ids",
                self.redis_client.lrange,
                f"{self.conversation_id}:message_ids",
                0,
                -1,
            )

            if not (0 <= index < len(message_ids)):
                raise ValueError(f"Invalid message index: {index}")

            message_id = message_ids[index]
            self._safe_redis_operation(
                "delete_message",
                self.redis_client.delete,
                f"{self.conversation_id}:message:{message_id}",
            )
            self._safe_redis_operation(
                "remove_message_id",
                self.redis_client.lrem,
                f"{self.conversation_id}:message_ids",
                1,
                message_id,
            )
            logger.info(
                f"Deleted message {message_id} from conversation {self.conversation_id}"
            )
        except Exception as e:
            error_msg = (
                f"Failed to delete message at index {index}: {str(e)}"
            )
            logger.error(error_msg)
            raise RedisOperationError(error_msg)

    def update(
        self, index: int, role: str, content: Union[str, dict]
    ):
        """Update a message in the conversation history.

        Args:
            index (int): Index of the message to update.
            role (str): Role of the speaker.
            content (Union[str, dict]): New content of the message.

        Raises:
            RedisOperationError: If the operation fails.
            ValueError: If the index is invalid.
        """
        try:
            message_ids = self._safe_redis_operation(
                "get_message_ids",
                self.redis_client.lrange,
                f"{self.conversation_id}:message_ids",
                0,
                -1,
            )

            if not message_ids or not (0 <= index < len(message_ids)):
                raise ValueError(f"Invalid message index: {index}")

            message_id = message_ids[index]
            message = {
                "role": role,
                "content": (
                    json.dumps(content)
                    if isinstance(content, (dict, list))
                    else str(content)
                ),
                "timestamp": datetime.datetime.now().isoformat(),
                "cached": "false",
            }

            # Update the message in Redis
            self._safe_redis_operation(
                "update_message",
                self.redis_client.hset,
                f"{self.conversation_id}:message:{message_id}",
                mapping=message,
            )

            # Update token count if needed
            if self.token_count:
                self._count_tokens(content, message, message_id)

            logger.debug(
                f"Updated message {message_id} in conversation {self.conversation_id}"
            )
        except Exception as e:
            error_msg = (
                f"Failed to update message at index {index}: {str(e)}"
            )
            logger.error(error_msg)
            raise RedisOperationError(error_msg)

    def query(self, index: int) -> dict:
        """Query a message in the conversation history.

        Args:
            index (int): Index of the message to query.

        Returns:
            dict: The message with its role and content.
        """
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        if 0 <= index < len(message_ids):
            message_id = message_ids[index]
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            if "content" in message and message["content"].startswith(
                "{"
            ):
                try:
                    message["content"] = json.loads(
                        message["content"]
                    )
                except json.JSONDecodeError:
                    pass
            return message
        return {}

    def search(self, keyword: str) -> List[dict]:
        """Search for messages containing a keyword.

        Args:
            keyword (str): Keyword to search for.

        Returns:
            List[dict]: List of messages containing the keyword.
        """
        results = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )

        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            if keyword in message.get("content", ""):
                if message["content"].startswith("{"):
                    try:
                        message["content"] = json.loads(
                            message["content"]
                        )
                    except json.JSONDecodeError:
                        pass
                results.append(message)

        return results

    def display_conversation(self, detailed: bool = False):
        """Display the conversation history.

        Args:
            detailed (bool): Whether to show detailed information.
        """
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            if message["content"].startswith("{"):
                try:
                    message["content"] = json.loads(
                        message["content"]
                    )
                except json.JSONDecodeError:
                    pass
            formatter.print_panel(
                f"{message['role']}: {message['content']}\n\n"
            )

    def export_conversation(self, filename: str):
        """Export the conversation history to a file.

        Args:
            filename (str): Filename to export to.
        """
        with open(filename, "w") as f:
            message_ids = self.redis_client.lrange(
                f"{self.conversation_id}:message_ids", 0, -1
            )
            for message_id in message_ids:
                message = self.redis_client.hgetall(
                    f"{self.conversation_id}:message:{message_id}"
                )
                f.write(f"{message['role']}: {message['content']}\n")

    def import_conversation(self, filename: str):
        """Import a conversation history from a file.

        Args:
            filename (str): Filename to import from.
        """
        with open(filename) as f:
            for line in f:
                role, content = line.split(": ", 1)
                self.add(role, content.strip())

    def count_messages_by_role(self) -> Dict[str, int]:
        """Count messages by role.

        Returns:
            Dict[str, int]: Count of messages by role.
        """
        counts = {
            "system": 0,
            "user": 0,
            "assistant": 0,
            "function": 0,
        }
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            role = message["role"].lower()
            if role in counts:
                counts[role] += 1
        return counts

    def return_history_as_string(self) -> str:
        """Return the conversation history as a string.

        Returns:
            str: The conversation history formatted as a string.
        """
        messages = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            messages.append(
                f"{message['role']}: {message['content']}\n\n"
            )
        return "".join(messages)

    def get_str(self) -> str:
        """Get the conversation history as a string.

        Returns:
            str: The conversation history.
        """
        messages = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            msg_str = f"{message['role']}: {message['content']}"
            if "token_count" in message:
                msg_str += f" (tokens: {message['token_count']})"
            if message.get("cached", "false") == "true":
                msg_str += " [cached]"
            messages.append(msg_str)
        return "\n".join(messages)

    def save_as_json(self, filename: str = None):
        """Save the conversation history as a JSON file.

        Args:
            filename (str): Filename to save to.
        """
        if filename:
            data = []
            message_ids = self.redis_client.lrange(
                f"{self.conversation_id}:message_ids", 0, -1
            )
            for message_id in message_ids:
                message = self.redis_client.hgetall(
                    f"{self.conversation_id}:message:{message_id}"
                )
                if message["content"].startswith("{"):
                    try:
                        message["content"] = json.loads(
                            message["content"]
                        )
                    except json.JSONDecodeError:
                        pass
                data.append(message)

            with open(filename, "w") as f:
                json.dump(data, f, indent=2)

    def load_from_json(self, filename: str):
        """Load the conversation history from a JSON file.

        Args:
            filename (str): Filename to load from.
        """
        with open(filename) as f:
            data = json.load(f)
            self.clear()  # Clear existing conversation
            for message in data:
                self.add(message["role"], message["content"])

    def clear(self):
        """Clear the conversation history."""
        # Get all message IDs
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )

        # Delete all messages
        for message_id in message_ids:
            self.redis_client.delete(
                f"{self.conversation_id}:message:{message_id}"
            )

        # Clear message IDs list
        self.redis_client.delete(
            f"{self.conversation_id}:message_ids"
        )

        # Clear cache
        self.redis_client.delete(f"{self.conversation_id}:cache")

        # Reset message counter
        self.redis_client.delete(
            f"{self.conversation_id}:message_counter"
        )

    def to_dict(self) -> List[Dict]:
        """Convert the conversation history to a dictionary.

        Returns:
            List[Dict]: The conversation history as a list of dictionaries.
        """
        data = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            if message["content"].startswith("{"):
                try:
                    message["content"] = json.loads(
                        message["content"]
                    )
                except json.JSONDecodeError:
                    pass
            data.append(message)
        return data

    def to_json(self) -> str:
        """Convert the conversation history to a JSON string.

        Returns:
            str: The conversation history as a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Convert the conversation history to a YAML string.

        Returns:
            str: The conversation history as a YAML string.
        """
        return yaml.dump(self.to_dict())

    def get_last_message_as_string(self) -> str:
        """Get the last message as a formatted string.

        Returns:
            str: The last message formatted as 'role: content'.
        """
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", -1, -1
        )
        if message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_ids[0]}"
            )
            return f"{message['role']}: {message['content']}"
        return ""

    def return_messages_as_list(self) -> List[str]:
        """Return the conversation messages as a list of formatted strings.

        Returns:
            List[str]: List of messages formatted as 'role: content'.
        """
        messages = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            messages.append(
                f"{message['role']}: {message['content']}"
            )
        return messages

    def return_messages_as_dictionary(self) -> List[Dict]:
        """Return the conversation messages as a list of dictionaries.

        Returns:
            List[Dict]: List of dictionaries containing role and content of each message.
        """
        messages = []
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            if message["content"].startswith("{"):
                try:
                    message["content"] = json.loads(
                        message["content"]
                    )
                except json.JSONDecodeError:
                    pass
            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )
        return messages

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about cache usage.

        Returns:
            Dict[str, Union[int, float]]: Statistics about cache usage.
        """
        with self.cache_lock:
            total = (
                self.cache_stats["hits"] + self.cache_stats["misses"]
            )
            hit_rate = (
                self.cache_stats["hits"] / total if total > 0 else 0
            )
            return {
                "hits": self.cache_stats["hits"],
                "misses": self.cache_stats["misses"],
                "cached_tokens": self.cache_stats["cached_tokens"],
                "total_tokens": self.cache_stats["total_tokens"],
                "hit_rate": hit_rate,
            }

    def truncate_memory_with_tokenizer(self):
        """Truncate the conversation history based on token count."""
        if not self.tokenizer:
            return

        total_tokens = 0
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", 0, -1
        )
        keep_message_ids = []

        for message_id in message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_id}"
            )
            tokens = int(
                message.get("token_count", 0)
            ) or count_tokens(message["content"])

            if total_tokens + tokens <= self.context_length:
                total_tokens += tokens
                keep_message_ids.append(message_id)
            else:
                # Delete messages that exceed the context length
                self.redis_client.delete(
                    f"{self.conversation_id}:message:{message_id}"
                )

        # Update the message IDs list
        self.redis_client.delete(
            f"{self.conversation_id}:message_ids"
        )
        if keep_message_ids:
            self.redis_client.rpush(
                f"{self.conversation_id}:message_ids",
                *keep_message_ids,
            )

    def get_final_message(self) -> str:
        """Return the final message from the conversation history.

        Returns:
            str: The final message formatted as 'role: content'.
        """
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", -1, -1
        )
        if message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_ids[0]}"
            )
            return f"{message['role']}: {message['content']}"
        return ""

    def get_final_message_content(self) -> str:
        """Return the content of the final message from the conversation history.

        Returns:
            str: The content of the final message.
        """
        message_ids = self.redis_client.lrange(
            f"{self.conversation_id}:message_ids", -1, -1
        )
        if message_ids:
            message = self.redis_client.hgetall(
                f"{self.conversation_id}:message:{message_ids[0]}"
            )
            return message["content"]
        return ""

    def __del__(self):
        """Cleanup method to close Redis connection and stop embedded server if running."""
        try:
            if hasattr(self, "redis_client") and self.redis_client:
                self.redis_client.close()
                logger.debug(
                    f"Closed Redis connection for conversation {self.conversation_id}"
                )

            if (
                hasattr(self, "embedded_server")
                and self.embedded_server
            ):
                self.embedded_server.stop()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

    def _get_conversation_id_by_name(
        self, name: str
    ) -> Optional[str]:
        """Get conversation ID for a given name.

        Args:
            name (str): The conversation name to look up.

        Returns:
            Optional[str]: The conversation ID if found, None otherwise.
        """
        try:
            return self.redis_client.get(f"conversation_name:{name}")
        except Exception as e:
            logger.warning(
                f"Error looking up conversation name: {str(e)}"
            )
            return None

    def _save_conversation_name(self, name: str):
        """Save the mapping between conversation name and ID.

        Args:
            name (str): The name to save.
        """
        try:
            # Save name -> ID mapping
            self.redis_client.set(
                f"conversation_name:{name}", self.conversation_id
            )
            # Save ID -> name mapping
            self.redis_client.set(
                f"conversation_id:{self.conversation_id}:name", name
            )
        except Exception as e:
            logger.warning(
                f"Error saving conversation name: {str(e)}"
            )

    def get_name(self) -> Optional[str]:
        """Get the friendly name of the conversation.

        Returns:
            Optional[str]: The conversation name if set, None otherwise.
        """
        if hasattr(self, "name") and self.name:
            return self.name
        try:
            return self.redis_client.get(
                f"conversation_id:{self.conversation_id}:name"
            )
        except Exception:
            return None

    def set_name(self, name: str):
        """Set a new name for the conversation.

        Args:
            name (str): The new name to set.
        """
        old_name = self.get_name()
        if old_name:
            # Remove old name mapping
            self.redis_client.delete(f"conversation_name:{old_name}")

        self.name = name
        self._save_conversation_name(name)
        logger.info(f"Set conversation name to: {name}")
