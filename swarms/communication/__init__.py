from swarms.communication.base_communication import (
    BaseCommunication,
    Message,
    MessageType,
)
from swarms.communication.sqlite_wrap import SQLiteConversation
from swarms.communication.duckdb_wrap import DuckDBConversation

# Optional dependencies with graceful fallbacks
try:
    from swarms.communication.supabase_wrap import (
        SupabaseConversation,
        SupabaseConnectionError,
        SupabaseOperationError,
    )
except ImportError:
    SupabaseConversation = None
    SupabaseConnectionError = None
    SupabaseOperationError = None

try:
    from swarms.communication.redis_wrap import RedisConversation
except ImportError:
    RedisConversation = None

try:
    from swarms.communication.pulsar_struct import PulsarConversation
except ImportError:
    PulsarConversation = None

__all__ = [
    "BaseCommunication",
    "Message",
    "MessageType",
    "SQLiteConversation",
    "DuckDBConversation",
    "SupabaseConversation",
    "SupabaseConnectionError",
    "SupabaseOperationError",
    "RedisConversation",
    "PulsarConversation",
]
