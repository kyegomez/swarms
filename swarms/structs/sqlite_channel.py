"""A CommsChannel stored as rows in a SQLite database."""

import threading
from typing import Any, Dict, List, Optional

from swarms.structs.comms_channel import CommsChannel


class SqliteChannel(CommsChannel):
    """A conversation kept in a SQLite table, shared across threads and processes.

    One database file can hold many conversations, told apart by name.
    Writes are serialised by SQLite and by a lock on this connection, so a
    fan-out step can post from several threads at once.

    Args:
        path: The database file, created if missing. ":memory:" keeps the
            conversation in this process only.
        name: Which conversation in the file this channel is. Defaults to
            "default".
    """

    def __init__(self, path: str, name: str = "default"):
        import sqlite3

        self.path = str(path)
        self.name = name
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(
            self.path, check_same_thread=False
        )
        with self.conn:
            # Readers never block writers in WAL mode; the pragma is ignored for :memory:.
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY, "
                "channel TEXT NOT NULL, "
                "agent_name TEXT NOT NULL, "
                "content TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS messages_channel "
                "ON messages (channel, id)"
            )

    def post(self, agent_name: str, content: str) -> None:
        """
        Post a message to the channel from a specific agent.

        Args:
            agent_name (str): The name of the agent posting the message.
            content (str): The content of the message to post.

        Returns:
            None
        """
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO messages (channel, agent_name, content) "
                "VALUES (?, ?, ?)",
                (self.name, agent_name, str(content)),
            )

    def messages(self) -> List[Dict[str, Any]]:
        """
        Get all messages in the current channel in order.

        Returns:
            List[Dict[str, Any]]: List of dictionaries with 'agent_name' and 'content' for each message.
        """
        with self._lock:
            rows = self.conn.execute(
                "SELECT agent_name, content FROM messages "
                "WHERE channel = ? ORDER BY id",
                (self.name,),
            ).fetchall()
        return [
            {"agent_name": agent_name, "content": content}
            for agent_name, content in rows
        ]

    def fetch_messages(
        self,
        agent_name: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve messages from the channel with ids and timestamps, oldest first.

        Args:
            agent_name (Optional[str], optional): If provided, filters for only this speaker's rows. Defaults to None.
            limit (Optional[int], optional): If provided, limits to at most this many of the latest rows (but still in order). Defaults to None.

        Returns:
            List[Dict[str, Any]]: Rows with 'id', 'agent_name', 'content', and 'created_at'.
        """
        sql = "SELECT id, agent_name, content, created_at FROM messages WHERE channel = ?"
        params: List[Any] = [self.name]
        if agent_name is not None:
            sql += " AND agent_name = ?"
            params.append(agent_name)
        sql += " ORDER BY id DESC" if limit else " ORDER BY id"
        if limit:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self._lock:
            found = self.conn.execute(sql, params).fetchall()
        if limit:
            found.reverse()
        return [
            {"id": i, "agent_name": r, "content": c, "created_at": t}
            for i, r, c, t in found
        ]

    @classmethod
    def list_conversations(cls, path: str) -> List[str]:
        """
        List the conversation names stored in a specific SQLite database.

        Args:
            path (str): The database file.

        Returns:
            List[str]: Names of all conversations (channels) in the order their first message arrived.
        """
        import sqlite3

        conn = sqlite3.connect(str(path))
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'messages'"
            ).fetchone()
            if not exists:
                return []
            return [
                name
                for (name,) in conn.execute(
                    "SELECT channel FROM messages GROUP BY channel ORDER BY MIN(id)"
                )
            ]
        finally:
            conn.close()

    def clear(self) -> None:
        """
        Delete all messages from the current channel.
        """
        with self._lock, self.conn:
            self.conn.execute(
                "DELETE FROM messages WHERE channel = ?", (self.name,)
            )

    def close(self) -> None:
        """
        Close the SQLite connection.
        The file persists and keeps every message.
        """
        with self._lock:
            self.conn.close()
