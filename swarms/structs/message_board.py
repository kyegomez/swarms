"""A board of threads that agents start and post to, stored in SQLite."""

import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.conversation import Conversation

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS message_boards ("
    "id TEXT PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "description TEXT NOT NULL, "
    "created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS board_threads ("
    "id TEXT PRIMARY KEY, "
    "board_id TEXT NOT NULL REFERENCES message_boards (id), "
    "title TEXT NOT NULL, "
    "author TEXT NOT NULL, "
    "labels TEXT NOT NULL, "
    "created_at TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS board_messages ("
    "id INTEGER PRIMARY KEY, "
    "thread_id TEXT NOT NULL REFERENCES board_threads (id), "
    "author TEXT NOT NULL, "
    "content TEXT NOT NULL, "
    "created_at TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS board_threads_board "
    "ON board_threads (board_id)",
    "CREATE INDEX IF NOT EXISTS board_messages_thread "
    "ON board_messages (thread_id, id)",
)


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_labels(text: str) -> Dict[str, Optional[str]]:
    """Parse "kind=bug, status=open" into a dict; "key=" maps to None.

    Args:
        text: Comma-separated key=value pairs.

    Returns:
        Dict[str, Optional[str]]: The labels, with None for an empty value.
    """
    labels: Dict[str, Optional[str]] = {}

    for pair in (text or "").split(","):
        key, _, value = pair.partition("=")
        if key.strip():
            labels[key.strip()] = value.strip() or None

    return labels


def _format_labels(labels: Dict[str, str]) -> str:
    """Render labels as "kind=bug, status=open", or "none"."""
    return (
        ", ".join(f"{key}={value}" for key, value in labels.items())
        or "none"
    )


def _connect(db_path: str) -> sqlite3.Connection:
    """Open the database and create the board tables if missing.

    Args:
        db_path: The database file, or ":memory:".

    Returns:
        sqlite3.Connection: A connection usable from any thread.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)

    with conn:
        # Readers never block writers in WAL mode; ignored for :memory:.
        conn.execute("PRAGMA journal_mode=WAL")
        for statement in SCHEMA:
            conn.execute(statement)

    return conn


class BoardThreadChannel(CommsChannel):
    """One thread of a MessageBoard, used as an agent's or swarm's channel.

    Args:
        board: The board that holds the thread.
        thread_id: The thread to read and post to.

    Raises:
        KeyError: If the board has no thread with this id.
    """

    def __init__(self, board: "MessageBoard", thread_id: str):
        board._thread_row(thread_id)
        self.board = board
        self.thread_id = thread_id

    def post(self, agent_name: str, content: str) -> None:
        """Reply to the thread.

        Args:
            agent_name: Who is speaking.
            content: The message text.
        """
        self.board.reply(self.thread_id, agent_name, content)

    def messages(self) -> List[Dict[str, Any]]:
        """Every message in the thread, oldest first.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """
        return [
            {
                "agent_name": message["author"],
                "content": message["content"],
            }
            for message in self.board._messages(self.thread_id)
        ]


class MessageBoard:
    """Many threads of messages, like a forum, kept in a SQLite database.

    Only agents start threads, through the tools in agent_tools.
    People and code can list, read, reply to and label threads, and can hand
    one thread to an agent or swarm as its channel. Every board, thread and
    message is a row in the database, and one file can hold many boards.

    Args:
        name: The board's name.
        description: What the board is for.
        db_path: The SQLite file, created if missing. ":memory:" keeps the
            board in this process only.

    Example:
        >>> board = MessageBoard(name="eng", description="Bug triage")
        >>> triager = Agent(
        ...     agent_name="Triager",
        ...     tools=board.agent_tools,
        ...     max_loops=2,
        ... )
        >>> triager.run("Open a thread for each bug in this report: ...")
        >>> board.threads({"kind": "bug"})
    """

    def __init__(
        self,
        name: str = "board",
        description: str = "",
        db_path: str = ":memory:",
    ):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.db_path = str(db_path)
        self.created_at = _now()
        self._lock = threading.RLock()
        self._conn = _connect(self.db_path)

        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO message_boards "
                "(id, name, description, created_at) "
                "VALUES (?, ?, ?, ?)",
                (self.id, name, description, self.created_at),
            )

    @classmethod
    def load(cls, db_path: str, board_id: str) -> "MessageBoard":
        """Open a board saved in a database file.

        Args:
            db_path: The SQLite file that holds the board.
            board_id: The board's id.

        Returns:
            MessageBoard: The board, with every thread and message.

        Raises:
            KeyError: If the file has no board with this id.
        """
        conn = _connect(db_path)
        row = conn.execute(
            "SELECT name, description, created_at "
            "FROM message_boards WHERE id = ?",
            (board_id,),
        ).fetchone()

        if row is None:
            conn.close()
            raise KeyError(f"No board {board_id!r} in {db_path!r}")

        board = cls.__new__(cls)
        board.id = board_id
        board.name, board.description, board.created_at = row
        board.db_path = str(db_path)
        board._lock = threading.RLock()
        board._conn = conn
        return board

    @staticmethod
    def list_boards(db_path: str) -> List[Dict[str, str]]:
        """List the boards saved in a database file, oldest first.

        Args:
            db_path: The SQLite file.

        Returns:
            List[Dict[str, str]]: Each board's id, name, description and created_at.
        """
        conn = _connect(db_path)

        try:
            rows = conn.execute(
                "SELECT id, name, description, created_at "
                "FROM message_boards ORDER BY rowid"
            ).fetchall()
        finally:
            conn.close()

        return [
            {
                "id": board_id,
                "name": name,
                "description": description,
                "created_at": created_at,
            }
            for board_id, name, description, created_at in rows
        ]

    def reply(
        self, thread_id: str, author: str, content: str
    ) -> None:
        """Add a message to a thread.

        Args:
            thread_id: The thread to reply to.
            author: Who is speaking.
            content: The message.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        with self._lock, self._conn:
            self._thread_row(thread_id)
            self._insert_message(thread_id, author, content)

    def thread(self, thread_id: str) -> Conversation:
        """Return a thread's messages as a Conversation, read from the database.

        The Conversation is a copy; post to the thread with reply().

        Args:
            thread_id: The thread's id.

        Returns:
            Conversation: The thread's history, oldest first.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        conversation = Conversation(id=thread_id, time_enabled=True)
        conversation.conversation_history = [
            {
                "role": message["author"],
                "content": message["content"],
                "timestamp": message["created_at"],
            }
            for message in self._messages(thread_id)
        ]
        return conversation

    def threads(
        self, labels: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """List threads, most recent activity first.

        Args:
            labels: Keep only threads that carry every one of these labels.

        Returns:
            List[Dict[str, Any]]: Each thread's id, title, author, created_at,
                updated_at, labels and number of messages.
        """
        wanted = labels or {}

        with self._lock:
            rows = self._conn.execute(
                "SELECT t.id, t.title, t.author, t.created_at, "
                "MAX(m.created_at), t.labels, COUNT(m.id) "
                "FROM board_threads t "
                "JOIN board_messages m ON m.thread_id = t.id "
                "WHERE t.board_id = ? "
                "GROUP BY t.id ORDER BY MAX(m.id) DESC",
                (self.id,),
            ).fetchall()

        summaries = [
            {
                "id": thread_id,
                "title": title,
                "author": author,
                "created_at": created_at,
                "updated_at": updated_at,
                "labels": json.loads(thread_labels),
                "messages": count,
            }
            for (
                thread_id,
                title,
                author,
                created_at,
                updated_at,
                thread_labels,
                count,
            ) in rows
        ]

        return [
            summary
            for summary in summaries
            if all(
                summary["labels"].get(key) == value
                for key, value in wanted.items()
            )
        ]

    def set_labels(
        self, thread_id: str, labels: Dict[str, Optional[str]]
    ) -> None:
        """Set or remove labels on a thread.

        Args:
            thread_id: The thread to label.
            labels: Labels to set. A value of None removes that label.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        with self._lock, self._conn:
            current = self._thread_row(thread_id)["labels"]

            for key, value in labels.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value

            self._conn.execute(
                "UPDATE board_threads SET labels = ? WHERE id = ?",
                (json.dumps(current), thread_id),
            )

    def channel(self, thread_id: str) -> BoardThreadChannel:
        """Return one thread as a channel for an agent or swarm.

        Args:
            thread_id: The thread to read and post to.

        Returns:
            BoardThreadChannel: A channel bound to that thread.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        return BoardThreadChannel(self, thread_id)

    def close(self) -> None:
        """Close the database connection; the file keeps every message."""
        with self._lock:
            self._conn.close()

    @property
    def agent_tools(self) -> List[Callable[..., str]]:
        """The tools agents use to start, find, read and answer threads.

        Pass them as Agent(tools=board.agent_tools). Each agent that gets them
        signs its threads and replies with its own agent_name. The tools are
        create_thread, list_threads, read_thread, reply_to_thread and
        label_thread, and each returns text for the model.

        Returns:
            List[Callable[..., str]]: The tools, ready for Agent(tools=...).
        """
        return self._tools_for(None)

    def _tools_for(
        self, agent_name: Optional[str]
    ) -> List[Callable[..., str]]:
        """Build the board's tools, signed with agent_name.

        Args:
            agent_name: The agent the tools post as. None until an Agent binds them.

        Returns:
            List[Callable[..., str]]: The tools, each with a bind_agent hook.
        """
        board = self
        unbound = (
            "These tools post as the agent that holds them; pass them "
            "with Agent(tools=board.agent_tools)."
        )

        def create_thread(
            title: str, content: str, labels: str = ""
        ) -> str:
            """Start a new thread on the message board with an opening post.

            Args:
                title: A short title that says what the thread is about.
                content: The opening post.
                labels: Optional comma-separated key=value labels, for example "kind=bug, status=open".

            Returns:
                str: The new thread's id.
            """
            if agent_name is None:
                return unbound

            thread_labels = {
                key: value
                for key, value in _parse_labels(labels).items()
                if value is not None
            }
            thread_id = board._create_thread(
                title, agent_name, content, thread_labels
            )
            return f"Created thread {thread_id}: {title}"

        def list_threads(labels: str = "") -> str:
            """List the threads on the message board, most recent activity first.

            Args:
                labels: Optional comma-separated key=value labels; only threads with all of them are listed.

            Returns:
                str: One line per thread with its id, title, author, labels and message count.
            """
            wanted = {
                key: value
                for key, value in _parse_labels(labels).items()
                if value is not None
            }
            found = board.threads(wanted)
            if not found:
                return "No threads match."

            return "\n".join(
                f"{t['id']} | {t['title']} | by {t['author']} | "
                f"labels: {_format_labels(t['labels'])} | "
                f"{t['messages']} messages"
                for t in found
            )

        def read_thread(thread_id: str) -> str:
            """Read every message in one thread of the message board.

            Args:
                thread_id: The id of the thread to read.

            Returns:
                str: The thread's title, labels and messages, oldest first.
            """
            try:
                meta = board._thread_row(thread_id)
            except KeyError as error:
                return str(error.args[0])

            header = (
                f"Thread {thread_id}: {meta['title']} "
                f"(started by {meta['author']}, labels: "
                f"{_format_labels(meta['labels'])})"
            )
            messages = [
                f"{message['author']}: {message['content']}"
                for message in board._messages(thread_id)
            ]
            return "\n\n".join([header, *messages])

        def reply_to_thread(thread_id: str, content: str) -> str:
            """Post a reply to one thread of the message board.

            Args:
                thread_id: The id of the thread to reply to.
                content: The reply.

            Returns:
                str: Whether the reply was posted.
            """
            if agent_name is None:
                return unbound

            try:
                board.reply(thread_id, agent_name, content)
            except KeyError as error:
                return str(error.args[0])
            return f"Replied to thread {thread_id}"

        def label_thread(thread_id: str, labels: str) -> str:
            """Set or remove labels on one thread of the message board.

            Args:
                thread_id: The id of the thread to label.
                labels: Comma-separated key=value labels, for example "status=fixed"; "key=" with no value removes that label.

            Returns:
                str: The thread's labels after the change.
            """
            try:
                board.set_labels(thread_id, _parse_labels(labels))
            except KeyError as error:
                return str(error.args[0])
            current = board._thread_row(thread_id)["labels"]
            return f"Thread {thread_id} labels: {_format_labels(current)}"

        tools = [
            create_thread,
            list_threads,
            read_thread,
            reply_to_thread,
            label_thread,
        ]

        # Agent calls bind_agent with its own name when it sets up its tools.
        for index, tool in enumerate(tools):
            tool.bind_agent = (
                lambda name, index=index: board._tools_for(name)[
                    index
                ]
            )

        return tools

    def _create_thread(
        self,
        title: str,
        author: str,
        content: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> str:
        """Start a thread; agents reach this only through their create_thread tool.

        Args:
            title: The thread's title.
            author: The agent that started it.
            content: The opening post.
            labels: Labels such as {"kind": "bug", "status": "open"}.

        Returns:
            str: The new thread's id.
        """
        with self._lock, self._conn:
            thread_id = secrets.token_hex(4)
            while self._conn.execute(
                "SELECT 1 FROM board_threads WHERE id = ?",
                (thread_id,),
            ).fetchone():
                thread_id = secrets.token_hex(4)

            self._conn.execute(
                "INSERT INTO board_threads "
                "(id, board_id, title, author, labels, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    thread_id,
                    self.id,
                    title,
                    author,
                    json.dumps(labels or {}),
                    _now(),
                ),
            )
            self._insert_message(thread_id, author, content)

        return thread_id

    def _insert_message(
        self, thread_id: str, author: str, content: str
    ) -> None:
        """Insert one message row; the caller holds the lock and transaction."""
        self._conn.execute(
            "INSERT INTO board_messages "
            "(thread_id, author, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (thread_id, author, str(content), _now()),
        )

    def _thread_row(self, thread_id: str) -> Dict[str, Any]:
        """Return a thread's title, author and labels.

        Args:
            thread_id: The thread's id.

        Returns:
            Dict[str, Any]: The thread's title, author and labels.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT title, author, labels FROM board_threads "
                "WHERE id = ? AND board_id = ?",
                (thread_id, self.id),
            ).fetchone()

        if row is None:
            raise KeyError(
                f"No thread {thread_id!r} on board {self.name!r}"
            )

        title, author, labels = row
        return {
            "title": title,
            "author": author,
            "labels": json.loads(labels),
        }

    def _messages(self, thread_id: str) -> List[Dict[str, str]]:
        """Return a thread's messages, oldest first.

        Args:
            thread_id: The thread's id.

        Returns:
            List[Dict[str, str]]: Each message's author, content and created_at.

        Raises:
            KeyError: If the board has no thread with this id.
        """
        with self._lock:
            self._thread_row(thread_id)
            rows = self._conn.execute(
                "SELECT author, content, created_at FROM board_messages "
                "WHERE thread_id = ? ORDER BY id",
                (thread_id,),
            ).fetchall()

        return [
            {"author": author, "content": content, "created_at": at}
            for author, content, at in rows
        ]
