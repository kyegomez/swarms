"""
One owner for a swarm's autosave directory and every write into it.

Each swarm used to carry its own ``_setup_autosave`` and
``_save_conversation_history`` pair. They were copies of each other, so a fix
to one silently left the others wrong. :class:`WorkspaceManager` holds that
logic once; a swarm builds one in ``__init__`` and calls it from ``run``.

Example:
    >>> self.workspace = WorkspaceManager(self, verbose=self.verbose)
    >>> self.swarm_workspace_dir = self.workspace.dir
    >>> self.workspace.save_conversation(self.conversation)

Nothing here raises. Autosave is a side effect of a run, never a reason for
one to fail, so every method logs and returns ``None`` on error.
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from swarms.utils.workspace_utils import get_workspace_dir

DEFAULT_WORKSPACE_DIRNAME = "agent_workspace"
CONVERSATION_FILENAME = "conversation_history.json"

_INVALID_PATH_CHARS = '<>:"/\\|?*'
_MAX_NAME_LEN = 100


def sanitize_name(name: str) -> str:
    """
    Make a name safe to use as a single path segment.

    Args:
        name (str): The raw swarm or class name.

    Returns:
        str: A filesystem-safe name, or ``"unnamed"`` when empty.
    """
    if not name:
        return "unnamed"

    sanitized = str(name)
    for char in _INVALID_PATH_CHARS:
        sanitized = sanitized.replace(char, "_")

    sanitized = sanitized.strip(". ").replace(" ", "-")
    return sanitized[:_MAX_NAME_LEN] or "unnamed"


def agent_dir_name(agent_name: str, agent_id: str) -> str:
    """
    Build an agent's directory name as ``{name-of-agent}-{uuid12}``.

    Args:
        agent_name (str): The agent's name; lowercased and hyphenated.
        agent_id (str): The agent id, with any ``agent-`` prefix dropped.

    Returns:
        str: The directory name, e.g. ``"my-agent-a1b2c3d4e5f6"``.
    """
    safe = (agent_name or "agent").lower()
    for char in _INVALID_PATH_CHARS + " _":
        safe = safe.replace(char, "-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    safe = safe.strip("-")[:_MAX_NAME_LEN] or "agent"

    uuid_part = str(agent_id or "")
    if uuid_part.startswith("agent-"):
        uuid_part = uuid_part[len("agent-") :]
    return f"{safe}-{uuid_part[-12:]}"


def ensure_workspace_env(verbose: bool = False) -> Optional[str]:
    """
    Resolve ``WORKSPACE_DIR``, defaulting it under the current directory.

    Args:
        verbose (bool): Log the default when one has to be invented.

    Returns:
        Optional[str]: The workspace path, or ``None`` if unresolvable.
    """
    if not os.getenv("WORKSPACE_DIR"):
        default = os.path.join(os.getcwd(), DEFAULT_WORKSPACE_DIRNAME)
        os.environ["WORKSPACE_DIR"] = default
        # get_workspace_dir is lru_cached, so a stale miss would stick.
        get_workspace_dir.cache_clear()
        if verbose:
            logger.info(
                f"WORKSPACE_DIR not set, using default: {default}"
            )

    return get_workspace_dir()


def conversation_to_data(conversation: Any) -> Any:
    """
    Pull serialisable history out of a Conversation-like object.

    Args:
        conversation (Any): A ``Conversation``, or anything exposing
            ``conversation_history`` or ``to_dict``.

    Returns:
        Any: The history, or ``[]`` when nothing can be read.
    """
    if conversation is None:
        return []

    history = getattr(conversation, "conversation_history", None)
    if history is not None:
        return history

    to_dict = getattr(conversation, "to_dict", None)
    if callable(to_dict):
        return to_dict()

    return []


class WorkspaceManager:
    """
    A swarm's autosave directory, created once and written to on demand.

    The directory is ``{WORKSPACE_DIR}/swarms/{ClassName}/{name}-{stamp}``,
    created eagerly so ``dir`` is usable straight after construction.

    Args:
        owner (Any): The swarm instance. Its class name and ``name``
            attribute pick the directory, and it is the default source
            for conversation and config data.
        name (Optional[str]): Overrides ``owner.name`` in the path.
        use_timestamp (bool): Timestamp in the directory name when
            ``True``, otherwise a short UUID.
        verbose (bool): Log the directory and each successful write.
        enabled (bool): When ``False`` nothing is created or written and
            ``dir`` stays ``None``.

    Attributes:
        dir (Optional[str]): The directory, or ``None`` when disabled or
            setup failed.
    """

    def __init__(
        self,
        owner: Any,
        name: Optional[str] = None,
        use_timestamp: bool = True,
        verbose: bool = False,
        enabled: bool = True,
        subpath: Optional[Sequence[str]] = None,
        metadata_base: Optional[Dict[str, Any]] = None,
    ):
        self.owner = owner
        self.class_name = owner.__class__.__name__
        self.name = name or getattr(owner, "name", None) or "unnamed"
        self.use_timestamp = use_timestamp
        self.verbose = verbose
        self.enabled = enabled
        self.subpath = subpath
        self.metadata_base = metadata_base
        self.dir: Optional[str] = self._setup() if enabled else None

    @classmethod
    def for_agent(
        cls,
        agent: Any,
        verbose: bool = False,
        enabled: bool = True,
    ) -> "WorkspaceManager":
        """
        Build a manager over an Agent's own ``agents/{name}-{uuid}`` dir.

        Agents predate the ``swarms/`` layout and their path is load-bearing:
        the autonomous loop's file tools resolve against it.

        Args:
            agent (Any): The agent, read for ``agent_name`` and ``id``.
            verbose (bool): Log the directory and each successful write.
            enabled (bool): When ``False`` nothing is created or written.

        Returns:
            WorkspaceManager: Rooted at the agent's workspace directory.
        """
        name = getattr(agent, "agent_name", None)
        return cls(
            agent,
            name=name,
            verbose=verbose,
            enabled=enabled,
            subpath=(
                "agents",
                agent_dir_name(name, getattr(agent, "id", "")),
            ),
            metadata_base={
                "agent_id": getattr(agent, "id", None),
                "agent_name": name,
            },
        )

    def __repr__(self) -> str:
        return (
            f"WorkspaceManager({self.class_name}, dir={self.dir!r})"
        )

    def __bool__(self) -> bool:
        """True when writes will actually land somewhere."""
        return bool(self.dir)

    def _setup(self) -> Optional[str]:
        """
        Create the directory for this swarm.

        Returns:
            Optional[str]: The created path, or ``None`` on failure.
        """
        try:
            workspace_dir = ensure_workspace_env(self.verbose)
            if not workspace_dir:
                logger.warning(
                    "WORKSPACE_DIR unresolved; autosave disabled for "
                    f"{self.class_name}"
                )
                return None

            stamp = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                if self.use_timestamp
                else uuid.uuid4().hex[:12]
            )
            if self.subpath:
                path = os.path.join(workspace_dir, *self.subpath)
            else:
                path = os.path.join(
                    workspace_dir,
                    "swarms",
                    sanitize_name(self.class_name),
                    f"{sanitize_name(self.name)}-{stamp}",
                )
            os.makedirs(path, exist_ok=True)

            if self.verbose:
                logger.info(f"Autosave enabled, writing to: {path}")
            return path
        except Exception as e:
            logger.warning(
                f"Failed to setup autosave for {self.class_name}: {e}"
            )
            return None

    def save_json(self, filename: str, data: Any) -> Optional[str]:
        """
        Write ``data`` as JSON into the workspace directory.

        Args:
            filename (str): File name to write inside the directory.
            data (Any): Anything ``json.dumps`` can take with
                ``default=str``.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if not self.dir:
            return None

        try:
            path = os.path.join(self.dir, filename)
            # Written via temp+replace: config.json is rewritten every loop,
            # so a crash mid-write would otherwise truncate it.
            temp_path = f"{path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    default=str,
                    ensure_ascii=False,
                )
            os.replace(temp_path, path)

            if self.verbose:
                logger.debug(f"Saved {filename} to {path}")
            return path
        except Exception as e:
            logger.warning(f"Failed to save {filename}: {e}")
            return None

    def save_text(self, filename: str, content: str) -> Optional[str]:
        """
        Write text into the workspace directory, atomically.

        Args:
            filename (str): File name to write inside the directory.
            content (str): The text to write.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if not self.dir:
            return None

        try:
            path = os.path.join(self.dir, filename)
            temp_path = f"{path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(temp_path, path)

            if self.verbose:
                logger.debug(f"Saved {filename} to {path}")
            return path
        except Exception as e:
            logger.warning(f"Failed to save {filename}: {e}")
            return None

    def save_conversation(
        self,
        conversation: Any = None,
        filename: str = CONVERSATION_FILENAME,
    ) -> Optional[str]:
        """
        Write conversation history to ``conversation_history.json``.

        Args:
            conversation (Any): The conversation to save. Defaults to
                ``owner.conversation`` when omitted, which is wrong for
                swarms that keep it elsewhere - pass it explicitly there.
            filename (str): Overrides the output file name.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if conversation is None:
            conversation = getattr(self.owner, "conversation", None)

        return self.save_json(
            filename, conversation_to_data(conversation)
        )

    def save_config(
        self, additional_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Write the owner's configuration to ``config.json``.

        Args:
            additional_metadata (Optional[Dict[str, Any]]): Merged into
                the ``_autosave_metadata`` block.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if not self.dir:
            return None

        try:
            to_dict = getattr(self.owner, "to_dict", None)
            if callable(to_dict):
                config = to_dict()
            else:
                config = {
                    k: v
                    for k, v in vars(self.owner).items()
                    if not k.startswith("_") and not callable(v)
                }

            base = self.metadata_base
            if base is None:
                base = {
                    "class_name": self.class_name,
                    "swarm_name": self.name,
                    "swarm_id": getattr(self.owner, "id", None),
                }
            config["_autosave_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                **base,
                **(additional_metadata or {}),
            }
            return self.save_json("config.json", config)
        except Exception as e:
            logger.warning(f"Failed to save swarm config: {e}")
            return None

    def save_state(
        self, state_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Write a state snapshot to ``state.json``.

        Args:
            state_data (Optional[Dict[str, Any]]): Extra keys merged over
                the defaults.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if not self.dir:
            return None

        state = {
            "timestamp": datetime.now().isoformat(),
            "swarm_name": self.name,
            "swarm_id": getattr(self.owner, "id", None),
            "swarm_type": getattr(self.owner, "swarm_type", None),
            "conversation": conversation_to_data(
                getattr(self.owner, "conversation", None)
            ),
        }

        logs = getattr(self.owner, "logs", None)
        if logs is not None:
            state["logs"] = [str(log) for log in logs]

        state.update(state_data or {})
        return self.save_json("state.json", state)

    def save_metadata(
        self,
        execution_result: Any = None,
        execution_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Write run metadata to ``metadata.json``.

        Args:
            execution_result (Any): Summarised, not stored whole.
            execution_metadata (Optional[Dict[str, Any]]): Extra keys
                merged over the defaults.

        Returns:
            Optional[str]: The written path, or ``None`` if skipped.
        """
        if not self.dir:
            return None

        agents = getattr(self.owner, "agents", None)
        metadata = {
            "execution_timestamp": datetime.now().isoformat(),
            "swarm_name": self.name,
            "swarm_id": getattr(self.owner, "id", None),
            "swarm_type": getattr(self.owner, "swarm_type", None),
            "max_loops": getattr(self.owner, "max_loops", None),
            "agents_count": len(agents) if agents else 0,
        }

        if execution_result is not None:
            metadata["execution_result_summary"] = _summarize(
                execution_result
            )

        metadata.update(execution_metadata or {})
        return self.save_json("metadata.json", metadata)

    def save_all(
        self,
        conversation: Any = None,
        execution_result: Any = None,
    ) -> Dict[str, Optional[str]]:
        """
        Write config, state, metadata and conversation in one call.

        Args:
            conversation (Any): Passed through to ``save_conversation``.
            execution_result (Any): Passed through to ``save_metadata``.

        Returns:
            Dict[str, Optional[str]]: Written path per file, ``None`` for
            any that was skipped.
        """
        return {
            "config": self.save_config(),
            "state": self.save_state(),
            "metadata": self.save_metadata(execution_result),
            "conversation": self.save_conversation(conversation),
        }


def _summarize(result: Any) -> Any:
    """
    Describe a run result without embedding the whole thing.

    Args:
        result (Any): The value returned by a swarm run.

    Returns:
        Any: A short scalar or a type/length pair.
    """
    if isinstance(result, (str, int, float, bool)):
        return str(result)
    if isinstance(result, (list, dict)):
        return {
            "type": type(result).__name__,
            "length": len(result),
        }
    return type(result).__name__
