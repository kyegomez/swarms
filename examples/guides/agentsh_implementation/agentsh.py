"""
AgentSH — a self-organized multi-agent organization harness on Swarms.

Implements *Agensh: Scaling Organizational Intelligence to 1,024 Agents*
(Zhan, Song, Dong, Huang, Lian, Xia, Wei — Microsoft Research,
arXiv:2609.26781).

There is no orchestrator. ``num_workers`` identical workers run concurrently
and asynchronously, and every one of them repeats the same multi-agent
cooperation loop (paper §2.1)::

    gather context → claim sub-task → take action → verify results → merge progress
          ↑                                                                │
          └────────────────────────────── repeat ──────────────────────────┘

The loop lives in the worker prompt, which is identical for every worker
except its ID (paper §2.3, Appendix A). The runtime only supplies the
*agentic organization infrastructure* (paper §2.2):

``SharedWorkspace``
    A versioned file store standing in for Git. ``main`` holds integrated
    work; each worker edits a private branch and merges into ``main``.
    Merges that touch files a peer changed since the last sync are refused,
    and ``sync`` writes conflict markers the worker must resolve.

``MessageInterface``
    A shared task channel for announcements (low priority, delivered at the
    start of each turn) and direct messages for urgent one-to-one traffic
    (high priority, delivered at the end of every infrastructure tool call).

``SharedContext``
    An append-only board of short typed entries — ``OBSERVED``, ``FACT``,
    ``FAIL``, ``CLAIM`` and ``PATCH_SUMMARY`` — with read, grep over the full
    history, and unfold for long details. New peer entries are forwarded
    mid-turn, appended to infrastructure tool results.

Around each worker's own agentic loop (a Swarms ``Agent``), the event-driven
runtime of Appendix B builds one prompt per *turn* from queued events,
nudges workers that publish nothing, and sends the wrap-up reminders of
Appendix C as the budget runs out.

Example:
    >>> from agentsh import AgentSH
    >>> org = AgentSH(num_workers=4, model_name="gpt-5.4-mini", max_turns=4)
    >>> report = org.run("Build a pure-Python Markdown-to-HTML converter with tests.")
    >>> print(report)
"""

import functools
import itertools
import threading
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from swarms import Agent

__all__ = [
    "AgentSH",
    "BoardEntry",
    "Commit",
    "EntryKind",
    "MergeResult",
    "Message",
    "MessageInterface",
    "SharedContext",
    "SharedWorkspace",
    "Worker",
]


class EntryKind(str, Enum):
    """The five typed entries a worker may publish to the shared context."""

    OBSERVED = "OBSERVED"
    FACT = "FACT"
    FAIL = "FAIL"
    CLAIM = "CLAIM"
    PATCH_SUMMARY = "PATCH_SUMMARY"


class BoardEntry(BaseModel):
    """One immutable note on the shared context board."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(
        description="Monotonic 1-based id; also the board position."
    )
    author: str = Field(description="Handle of the writing worker.")
    kind: EntryKind = Field(description="The entry type.")
    summary: str = Field(
        description="Short, capped one-liner shown to every worker."
    )
    detail: str = Field(
        default="",
        description="Optional long form, read with board_unfold.",
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Wall-clock creation time.",
    )

    def render(self) -> str:
        """Return the one-line form used in prompts and tool results."""
        more = " [+detail]" if self.detail else ""
        return f"#{self.id} {self.kind.value} @{self.author}: {self.summary}{more}"


class Message(BaseModel):
    """A message on the task channel or a direct message."""

    model_config = ConfigDict(frozen=True)

    id: int = Field(description="Monotonic id across all messages.")
    sender: str = Field(description="Handle of the sender.")
    recipient: Optional[str] = Field(
        default=None,
        description="Handle of the recipient; None for the channel.",
    )
    content: str = Field(description="The message text.")
    created_at: float = Field(
        default_factory=time.time,
        description="Wall-clock send time.",
    )

    def render(self) -> str:
        """Return the one-line form used in prompts and tool results."""
        where = "DM" if self.recipient else "#task"
        return f"[{where}] @{self.sender}: {self.content}"


class Commit(BaseModel):
    """A change set merged into ``main``."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Short hexadecimal commit hash.")
    revision: int = Field(
        description="Revision of main this commit produced (1-based)."
    )
    author: str = Field(description="Handle of the merging worker.")
    message: str = Field(description="Commit message.")
    changes: Dict[str, Optional[str]] = Field(
        description="Changed paths to new content; None for deletions."
    )
    created_at: float = Field(
        default_factory=time.time,
        description="Wall-clock merge time.",
    )


class Branch(BaseModel):
    """A worker's private checkout of the shared workspace."""

    owner: str = Field(description="Handle of the owning worker.")
    base_revision: int = Field(
        description="Revision of main the branch last synced with."
    )
    files: Dict[str, str] = Field(
        description="Full view: main at base plus local edits."
    )
    dirty: Dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="Unmerged local changes; None for deletions.",
    )


class MergeResult(BaseModel):
    """Outcome of merging a branch into ``main``."""

    model_config = ConfigDict(frozen=True)

    merged: bool = Field(description="Whether the branch landed.")
    commit: Optional[Commit] = Field(
        default=None, description="The new commit when merged."
    )
    conflicts: Tuple[str, ...] = Field(
        default=(), description="Paths that blocked the merge."
    )
    reason: str = Field(
        default="", description="Why the merge was refused."
    )


class SharedContext:
    """Append-only board of typed findings shared by the whole organization.

    Thread-safe. Entry ``n`` always sits at position ``n - 1``, so a worker's
    read cursor is simply the id of the last entry it has seen.
    """

    DEFAULT_SUMMARY_LIMIT = 100
    SUMMARY_LIMITS: Dict[EntryKind, int] = {
        EntryKind.PATCH_SUMMARY: 300
    }

    def __init__(self) -> None:
        self._entries: List[BoardEntry] = []
        self._lock = threading.Lock()

    @property
    def latest_id(self) -> int:
        """Id of the newest entry, or ``0`` for an empty board."""
        return len(self._entries)

    def write(
        self,
        author: str,
        kind: EntryKind,
        summary: str,
        detail: str = "",
    ) -> BoardEntry:
        """Append an entry, truncating the summary to its kind's cap.

        Args:
            author: Handle of the writing worker.
            kind: Entry type.
            summary: One-line summary; truncated beyond the cap.
            detail: Optional long form.

        Returns:
            The stored entry.
        """
        limit = self.SUMMARY_LIMITS.get(
            kind, self.DEFAULT_SUMMARY_LIMIT
        )
        summary = " ".join(summary.split())
        with self._lock:
            entry = BoardEntry(
                id=len(self._entries) + 1,
                author=author,
                kind=kind,
                summary=summary[:limit],
                detail=detail.strip(),
            )
            self._entries.append(entry)
            return entry

    def recent(self, limit: int) -> List[BoardEntry]:
        """Return the newest ``limit`` entries, oldest first."""
        with self._lock:
            return self._entries[-limit:] if limit > 0 else []

    def since(self, cursor: int) -> List[BoardEntry]:
        """Return every entry newer than id ``cursor``, oldest first."""
        with self._lock:
            return self._entries[cursor:]

    def get(self, entry_id: int) -> Optional[BoardEntry]:
        """Return the entry with ``entry_id``, or ``None`` if it does not exist."""
        with self._lock:
            if 1 <= entry_id <= len(self._entries):
                return self._entries[entry_id - 1]
        return None

    def grep(self, query: str) -> List[BoardEntry]:
        """Search the complete history, case-insensitively.

        ``','`` separates OR-clauses and ``'&'`` joins AND-terms, so
        ``"a&b,c&d"`` matches entries containing (a AND b) OR (c AND d).

        Args:
            query: The keyword expression.

        Returns:
            Matching entries, oldest first.
        """
        clauses = [
            [
                term.strip().lower()
                for term in clause.split("&")
                if term.strip()
            ]
            for clause in query.split(",")
        ]
        clauses = [clause for clause in clauses if clause]
        if not clauses:
            return []

        def matches(entry: BoardEntry) -> bool:
            text = f"{entry.kind.value} {entry.author} {entry.summary} {entry.detail}".lower()
            return any(
                all(term in text for term in clause)
                for clause in clauses
            )

        with self._lock:
            return [
                entry for entry in self._entries if matches(entry)
            ]

    def entries(self) -> List[BoardEntry]:
        """Return a snapshot of every entry, oldest first."""
        with self._lock:
            return list(self._entries)


class MessageInterface:
    """Task channel plus per-worker direct messages, with retained history.

    Thread-safe. Readers keep their own integer cursors into the channel and
    into their direct-message inbox, so delivery is asynchronous and nothing
    is ever lost or delivered twice.
    """

    def __init__(self) -> None:
        self._channel: List[Message] = []
        self._direct: Dict[str, List[Message]] = defaultdict(list)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def post(self, sender: str, content: str) -> Message:
        """Post ``content`` to the shared task channel."""
        with self._lock:
            message = Message(
                id=next(self._ids), sender=sender, content=content
            )
            self._channel.append(message)
            return message

    def send(
        self, sender: str, recipient: str, content: str
    ) -> Message:
        """Send a direct message from ``sender`` to ``recipient``."""
        with self._lock:
            message = Message(
                id=next(self._ids),
                sender=sender,
                recipient=recipient,
                content=content,
            )
            self._direct[recipient].append(message)
            return message

    def channel_since(self, cursor: int) -> List[Message]:
        """Return channel messages after position ``cursor``."""
        with self._lock:
            return self._channel[cursor:]

    def direct_since(
        self, recipient: str, cursor: int
    ) -> List[Message]:
        """Return direct messages to ``recipient`` after position ``cursor``."""
        with self._lock:
            return self._direct[recipient][cursor:]

    def history(self, limit: int) -> List[Message]:
        """Return the newest ``limit`` channel messages, oldest first."""
        with self._lock:
            return self._channel[-limit:] if limit > 0 else []

    def channel(self) -> List[Message]:
        """Return a snapshot of the full channel history."""
        with self._lock:
            return list(self._channel)


class SharedWorkspace:
    """A small Git-like file store: one ``main`` branch plus private branches.

    Merges are file-granular: a merge is refused when any file it changes
    was also changed on ``main`` after the branch's last sync and the two
    versions differ. ``sync`` then brings ``main`` into the branch and
    wraps each such file in conflict markers, which must be removed before
    the branch can merge. Thread-safe.
    """

    CONFLICT_MARKER = "<<<<<<< "

    def __init__(self) -> None:
        self._main: Dict[str, str] = {}
        self._touched: Dict[str, int] = {}
        self._history: List[Commit] = []
        self._branches: Dict[str, Branch] = {}
        self._lock = threading.RLock()

    @property
    def revision(self) -> int:
        """Current revision of ``main`` (the number of merged commits)."""
        return len(self._history)

    @staticmethod
    def normalize(path: str) -> str:
        """Return ``path`` as a clean relative POSIX path.

        Raises:
            ValueError: If the path is empty or escapes the workspace root.
        """
        parts = [
            p
            for p in path.strip().replace("\\", "/").split("/")
            if p not in ("", ".")
        ]
        if not parts or ".." in parts:
            raise ValueError(f"invalid workspace path: {path!r}")
        return "/".join(parts)

    def checkout(self, owner: str) -> Branch:
        """Return ``owner``'s branch, creating it from ``main`` on first use."""
        with self._lock:
            if owner not in self._branches:
                self._branches[owner] = Branch(
                    owner=owner,
                    base_revision=self.revision,
                    files=dict(self._main),
                )
            return self._branches[owner]

    def list(self, owner: str, on_main: bool = False) -> List[str]:
        """Return the sorted paths on ``main`` or on ``owner``'s branch."""
        with self._lock:
            files = (
                self._main if on_main else self.checkout(owner).files
            )
            return sorted(files)

    def read(
        self, owner: str, path: str, on_main: bool = False
    ) -> Optional[str]:
        """Return a file's content from ``main`` or ``owner``'s branch."""
        path = self.normalize(path)
        with self._lock:
            files = (
                self._main if on_main else self.checkout(owner).files
            )
            return files.get(path)

    def write(self, owner: str, path: str, content: str) -> str:
        """Write ``content`` to ``path`` on ``owner``'s branch and return the path."""
        path = self.normalize(path)
        with self._lock:
            branch = self.checkout(owner)
            branch.files[path] = content
            branch.dirty[path] = content
            return path

    def delete(self, owner: str, path: str) -> bool:
        """Delete ``path`` on ``owner``'s branch; return whether it existed."""
        path = self.normalize(path)
        with self._lock:
            branch = self.checkout(owner)
            existed = branch.files.pop(path, None) is not None
            if existed or path in self._main:
                branch.dirty[path] = None
            return existed

    def changes(self, owner: str) -> Dict[str, Optional[str]]:
        """Return a copy of ``owner``'s unmerged changes."""
        with self._lock:
            return dict(self.checkout(owner).dirty)

    def sync(self, owner: str) -> List[str]:
        """Bring the latest ``main`` into ``owner``'s branch.

        Untouched files fast-forward; local edits are kept. A file edited
        both locally and on ``main`` since the last sync is rewritten with
        conflict markers.

        Returns:
            The paths left in conflict.
        """
        with self._lock:
            branch = self.checkout(owner)
            files = dict(self._main)
            conflicts: List[str] = []
            for path, mine in list(branch.dirty.items()):
                theirs = self._main.get(path)
                if mine == theirs:
                    del branch.dirty[path]
                    continue
                if self._touched.get(path, 0) > branch.base_revision:
                    mine = self._conflict_text(mine, theirs)
                    branch.dirty[path] = mine
                    conflicts.append(path)
                if mine is None:
                    files.pop(path, None)
                else:
                    files[path] = mine
            branch.files = files
            branch.base_revision = self.revision
            return conflicts

    def merge(self, owner: str, message: str) -> MergeResult:
        """Merge ``owner``'s branch into ``main``.

        Args:
            owner: Handle of the merging worker.
            message: Commit message.

        Returns:
            A ``MergeResult``; on success the branch is fast-forwarded to
            the new ``main``.
        """
        with self._lock:
            branch = self.checkout(owner)
            if not branch.dirty:
                return MergeResult(
                    merged=False,
                    reason="nothing to merge: your branch has no changes",
                )

            unresolved = tuple(
                path
                for path, content in branch.dirty.items()
                if content is not None
                and self.CONFLICT_MARKER in content
            )
            if unresolved:
                return MergeResult(
                    merged=False,
                    conflicts=unresolved,
                    reason="unresolved conflict markers; edit these files, re-verify, then merge",
                )

            stale = tuple(
                path
                for path, content in branch.dirty.items()
                if self._touched.get(path, 0) > branch.base_revision
                and self._main.get(path) != content
            )
            if stale:
                return MergeResult(
                    merged=False,
                    conflicts=stale,
                    reason="peers changed these files on main since your last sync; "
                    "call sync_main, resolve, re-verify, and merge again",
                )

            commit = Commit(
                id=uuid.uuid4().hex[:7],
                revision=self.revision + 1,
                author=owner,
                message=message.strip() or "(no message)",
                changes=dict(branch.dirty),
            )
            for path, content in commit.changes.items():
                if content is None:
                    self._main.pop(path, None)
                else:
                    self._main[path] = content
                self._touched[path] = commit.revision
            self._history.append(commit)

            branch.files = dict(self._main)
            branch.dirty.clear()
            branch.base_revision = commit.revision
            return MergeResult(merged=True, commit=commit)

    def snapshot(self) -> Dict[str, str]:
        """Return a copy of every file on ``main``."""
        with self._lock:
            return dict(self._main)

    def log(self) -> List[Commit]:
        """Return every commit on ``main``, oldest first."""
        with self._lock:
            return list(self._history)

    def export(self, directory: str) -> Path:
        """Write ``main`` to ``directory`` and return its resolved path."""
        root = Path(directory).expanduser().resolve()
        for path, content in self.snapshot().items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    @classmethod
    def _conflict_text(
        cls, mine: Optional[str], theirs: Optional[str]
    ) -> str:
        """Wrap both sides of a file conflict in Git-style markers."""
        return (
            f"{cls.CONFLICT_MARKER}yours\n{mine or ''}\n"
            f"=======\n{theirs or ''}\n>>>>>>> main\n"
        )


WORKER_PROMPT = """\
# Hello

You are {name}.

## Where you are

You work with {peers} peers. There is no manager and no orchestrator: the
organization moves forward only through what each of you claims, builds,
verifies and merges. Work is tracked in the **shared workspace**; you talk on
the **message interface**; you share findings on the **shared context board**.

- **shared workspace** -- a versioned file store. `main` holds the
  organization's integrated work and is the only thing that counts at the end.
  You edit your own private branch with `list_files`, `read_file`, `write_file`
  and `delete_file` (pass `scope="main"` to read what is already integrated),
  and publish with `merge_main`. Every merge is announced to the team.
- **message interface** -- the `#task` channel (`channel_post`) is where the
  team talks; be brief and concrete. Peer handles are `{handles}`; use the full
  handle. A **direct message** (`direct_message`) reaches a peer **inside their
  current turn**, so it is the fast way to settle a collision -- and it
  interrupts what they are doing. Send a direct message when two of you are
  about to edit the same thing; use the channel for everything else.
  `channel_history` shows recent channel traffic.
- **shared context board** -- an append-only log of short, verified notes from
  every worker. It is rendered into your prompt each turn under
  `==== SHARED CONTEXT ====`, and anything a peer writes after that reaches you
  mid-turn, appended to whatever infrastructure tool result you were waiting
  on. Publish with `board_write` the moment you establish something --
  `OBSERVED` for noticeable behaviour of the problem, `FACT` for what you
  confirmed, `FAIL` for a hypothesis or approach you falsified (the
  highest-value entry: it stops peers spending budget on it), `PATCH_SUMMARY`
  as `files= | idea= | evidence=` when you merge something, and `CLAIM` while
  you are working so peers pick a different angle. Summaries are capped (100
  chars, 300 for `PATCH_SUMMARY`); put the long version in `detail` and peers
  can `board_unfold` it. Do not re-derive a peer's `FACT`, and do not retry
  something recorded as `FAIL`. `board_read` returns the recent board while
  `board_grep` searches the complete history: ',' is OR and '&' is AND
  (case-insensitive), so `a&b,c&d` means (a AND b) OR (c AND d).

## The team

{size} workers, all equal. You claim what to build, you build it, you merge
it, and you choose the next thing. Your peers are working simultaneously: sync
with the latest `main` before merging, and sync up with peers to avoid
duplicated work. Always focus on the goal; reach for the coordination tools
only when your information can greatly impact others' work.

# Goal

{goal}

## Iron rules

1. **Fully autonomous**: the team defines requirements itself, breaks down work
   itself, decides itself, and finishes itself. There is no external input; do
   not wait for instructions and do not ask the channel "what to do next" --
   the answer is always: **push the goal forward**.
2. **Only `main` is delivered**: work that is not merged does not exist.
3. **Be truthful**: publish `FACT` and `PATCH_SUMMARY` only for what you have
   actually verified.

## Working with the tools

**Turns.** You are event-driven. Each turn starts with one message listing what
happened since your last turn (channel posts, merges, direct messages) and the
recent shared context. It reports facts; deciding what to do about them is
yours. A reply with **no tool call ends your turn**, so end every turn with a
short plain-text status: what you did and what you will do next.

A board entry arriving mid-turn provides information:

- `FAIL` -- stop if you are doing that thing.
- `FACT` / `OBSERVED` -- use it.
- `CLAIM` on what you are building -- a collision; settle it by direct message.
- any other `CLAIM` -- keep building what you are on.

**The work cycle.** Each step is finished by the worker doing it.

1. Gather context: read the board, the channel and `main` to see what is done
   and what remains.
2. Announce the area you are taking as a `CLAIM` on the board.
3. Build it on your branch, based on current `main`. You do not need to sync
   with others during implementation.
4. Verify it against the goal's acceptance criteria{verify_hint}. If it falls
   short, revise until it passes.
5. `merge_main`. If the merge is refused, `sync_main`, resolve any conflict
   markers, re-verify, and merge again.
6. Say what you finished as a `PATCH_SUMMARY` (`files= | idea= | evidence=`)
   on the board, then go back to 1.

When you are confident the goal is fully achieved on `main` and nothing
valuable remains to do, call `declare_goal_achieved` with your evidence.
"""

WRAP_UP_REMINDER = (
    "Stop dispatching new features. Ensure `main` is coherent and existing work "
    "is merged. Land what is ready; if something cannot be merged, say so on the "
    "channel and move on."
)

FINAL_REMINDER = (
    "Merge anything ready, then stop. Confirm `main` holds a complete, working "
    "deliverable and post the final state on the channel."
)

IDLE_NUDGE = (
    "There is always new work -- do not be bounded by the previous CLAIMs' "
    "scope: look at the goal from an untried angle and close the gap between "
    "the goal and the team's work on `main`."
)

PUBLISH_NUDGE = (
    "Your last turn published nothing to the shared context board. If you "
    "established anything peers can reuse -- an observation, a confirmed fact, "
    "a failed approach, a merged patch -- write it now."
)


class Worker:
    """One self-organized worker: a Swarms ``Agent`` plus its organization state.

    The worker's own agentic loop belongs to the ``Agent``; this class adds
    what AgentSH layers around it — identity, read cursors into the shared
    infrastructure, the infrastructure tools, and per-turn prompt assembly.

    Attributes:
        name: The worker's handle, e.g. ``"agentsh-worker3"``.
        org: The organization the worker belongs to.
        agent: The underlying single-agent harness.
        turns: Number of turns completed.
        finished: Whether the worker declared the goal achieved.
    """

    def __init__(self, name: str, org: "AgentSH") -> None:
        self.name = name
        self.org = org
        self.turns = 0
        self.finished = False
        self._board_cursor = 0
        self._channel_cursor = 0
        self._direct_cursor = 0
        self._tool_calls = 0
        self._board_writes = 0
        self._status = ""
        self._reminders_sent: Set[str] = set()
        settings: Dict[str, Any] = {
            "agent_description": f"Self-organized AgentSH worker {name}.",
            "model_name": org.model_name,
            "output_type": "final",
            "persistent_memory": False,
            "tool_call_summary": False,
            "print_on": org.print_on,
            **org.agent_kwargs,
        }
        self.agent = Agent(
            agent_name=name,
            system_prompt=org.worker_prompt(name),
            max_loops=org.loops_per_turn,
            tools=self._infrastructure_tools() + list(org.tools),
            stopping_func=self._ends_turn,
            **settings,
        )

    def run_turn(self) -> str:
        """Run one turn of the cooperation loop and return the worker's status."""
        prompt = self._turn_prompt()
        self._tool_calls = 0
        self._board_writes = 0
        self._status = (
            "(turn reached the step limit without a status)"
        )
        self.agent.run(prompt)
        self.turns += 1
        return self._status

    def _ends_turn(self, response: Any) -> bool:
        """End the turn on a plain-text reply with no tool call, keeping it as the status."""
        if isinstance(response, str) and response.strip():
            self._status = response.strip()
            return True
        return False

    def _turn_prompt(self) -> str:
        """Assemble the queued events and recent shared context into one prompt."""
        org = self.org
        channel = org.messages.channel_since(self._channel_cursor)
        direct = org.messages.direct_since(
            self.name, self._direct_cursor
        )
        self._channel_cursor += len(channel)
        self._direct_cursor += len(direct)
        peer_channel = [m for m in channel if m.sender != self.name]

        board = org.context.recent(org.context_window)
        if board:
            self._board_cursor = board[-1].id

        sections = [
            f"[turn {self.turns + 1}/{org.max_turns}] {self.name} | {org.clock()}"
        ]
        shown = peer_channel[-org.channel_window :]
        if len(peer_channel) > len(shown):
            sections.append(
                f"({len(peer_channel) - len(shown)} older channel messages omitted; "
                "use channel_history)"
            )
        if shown:
            sections.append(
                "== EVENTS ==\n"
                + "\n".join(m.render() for m in shown)
            )
        if direct:
            sections.append(
                "== DIRECT MESSAGES ==\n"
                + "\n".join(m.render() for m in direct)
            )
        rendered = "\n".join(e.render() for e in board) or "(empty)"
        sections.append(
            f"==== SHARED CONTEXT ====\n{rendered}\n==== END SHARED CONTEXT ===="
        )

        if self.turns > 0 and self._tool_calls == 0:
            sections.append(IDLE_NUDGE)
        elif self.turns > 0 and self._board_writes == 0:
            sections.append(PUBLISH_NUDGE)
        reminder = self._due_reminder()
        if reminder:
            sections.append(reminder)
        if self.turns == 0:
            sections.append("Begin with step 1 of the work cycle.")
        return "\n\n".join(sections)

    def _due_reminder(self) -> Optional[str]:
        """Return the Appendix C wrap-up reminder due now, at most once each."""
        org = self.org
        remaining_turns = org.max_turns - self.turns
        progress = org.time_progress()
        if "final" not in self._reminders_sent and (
            remaining_turns <= 1 or progress >= 0.985
        ):
            self._reminders_sent.update({"final", "wrap_up"})
            return FINAL_REMINDER
        if "wrap_up" not in self._reminders_sent and (
            remaining_turns <= 2 or progress >= 0.875
        ):
            self._reminders_sent.add("wrap_up")
            return WRAP_UP_REMINDER
        return None

    def _urgent_updates(self) -> str:
        """Drain direct messages and new peer board entries since the last delivery."""
        org = self.org
        direct = org.messages.direct_since(
            self.name, self._direct_cursor
        )
        board = org.context.since(self._board_cursor)
        self._direct_cursor += len(direct)
        self._board_cursor += len(board)
        peers = [e for e in board if e.author != self.name]
        lines = [m.render() for m in direct] + [
            e.render() for e in peers
        ]
        if not lines:
            return ""
        return "\n\n---- arrived while you worked ----\n" + "\n".join(
            lines
        )

    def _infrastructure(
        self, tool: Callable[..., str]
    ) -> Callable[..., str]:
        """Wrap an infrastructure tool: count it, trap errors, append urgent updates."""

        @functools.wraps(tool)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            self._tool_calls += 1
            try:
                result = tool(*args, **kwargs)
            except Exception as error:
                result = f"error: {error}"
            return f"{result}{self._urgent_updates()}"

        return wrapper

    def _infrastructure_tools(self) -> List[Callable[..., str]]:
        """Build the worker's shared-workspace, messaging and board tools."""
        org = self.org
        me = self.name

        def board_write(
            kind: str, summary: str, detail: str = ""
        ) -> str:
            """Publish a short, verified note to the shared context board.

            Args:
                kind: One of OBSERVED, FACT, FAIL, CLAIM, PATCH_SUMMARY.
                summary: One line; capped at 100 chars (300 for PATCH_SUMMARY).
                    PATCH_SUMMARY must read "files= | idea= | evidence=".
                detail: Optional long form that peers can board_unfold.

            Returns:
                The stored entry.
            """
            try:
                entry_kind = EntryKind(kind.strip().upper())
            except ValueError:
                return f"error: kind must be one of {', '.join(k.value for k in EntryKind)}"
            entry = org.context.write(me, entry_kind, summary, detail)
            self._board_writes += 1
            return f"published {entry.render()}"

        def board_read(limit: int = 200) -> str:
            """Read the most recent shared context board entries.

            Args:
                limit: How many of the newest entries to return (max 2000).

            Returns:
                One entry per line, oldest first.
            """
            entries = org.context.recent(
                max(1, min(int(limit), 2000))
            )
            return (
                "\n".join(e.render() for e in entries)
                or "(the board is empty)"
            )

        def board_grep(query: str) -> str:
            """Search the complete board history by keywords, case-insensitively.

            Args:
                query: Keywords where ',' is OR and '&' is AND,
                    e.g. "parser&table,renderer" means (parser AND table) OR renderer.

            Returns:
                Matching entries, one per line.
            """
            entries = org.context.grep(query)
            return (
                "\n".join(e.render() for e in entries)
                or f"no entries match {query!r}"
            )

        def board_unfold(entry_id: int) -> str:
            """Show the full detail of one board entry.

            Args:
                entry_id: The number after '#' in a rendered entry.

            Returns:
                The entry with its detail.
            """
            entry = org.context.get(int(entry_id))
            if entry is None:
                return f"error: no entry #{entry_id}"
            return (
                f"{entry.render()}\n\n{entry.detail or '(no detail)'}"
            )

        def channel_post(message: str) -> str:
            """Post a brief, concrete announcement to the team's #task channel.

            Args:
                message: The announcement.

            Returns:
                Confirmation.
            """
            org.messages.post(me, message)
            return "posted to #task"

        def direct_message(recipient: str, message: str) -> str:
            """Send an urgent direct message that reaches a peer inside their current turn.

            Use it to settle collisions, e.g. two workers about to edit the same file.

            Args:
                recipient: The peer's full handle.
                message: The message.

            Returns:
                Confirmation.
            """
            recipient = recipient.strip().lstrip("@")
            if recipient not in org.handles:
                return f"error: unknown handle {recipient!r}; handles look like {org.handles[0]!r}"
            if recipient == me:
                return "error: you cannot message yourself"
            org.messages.send(me, recipient, message)
            return f"delivered to {recipient}"

        def channel_history(limit: int = 30) -> str:
            """Read recent #task channel messages, including merge announcements.

            Args:
                limit: How many of the newest messages to return.

            Returns:
                One message per line, oldest first.
            """
            messages = org.messages.history(max(1, int(limit)))
            return (
                "\n".join(m.render() for m in messages)
                or "(the channel is empty)"
            )

        def list_files(scope: str = "branch") -> str:
            """List files in your private branch or on main.

            Args:
                scope: "branch" (your working copy) or "main" (integrated work).

            Returns:
                One path per line, with your unmerged changes marked.
            """
            on_main = scope.strip().lower() == "main"
            paths = org.workspace.list(me, on_main=on_main)
            header = (
                f"main @ r{org.workspace.revision}"
                if on_main
                else f"branch of {me}"
            )
            if on_main:
                return (
                    "\n".join([header, *paths])
                    if paths
                    else f"{header}: (empty)"
                )
            changed = org.workspace.changes(me)
            lines = [
                f"{p}  [modified]" if p in changed else p
                for p in paths
            ]
            lines += [
                f"{p}  [deleted]"
                for p, c in changed.items()
                if c is None
            ]
            return (
                "\n".join([header, *lines])
                if lines
                else f"{header}: (empty)"
            )

        def read_file(path: str, scope: str = "branch") -> str:
            """Read a file from your private branch or from main.

            Args:
                path: Relative file path, e.g. "src/parser.py".
                scope: "branch" (your working copy) or "main".

            Returns:
                The file content.
            """
            content = org.workspace.read(
                me, path, on_main=scope.strip().lower() == "main"
            )
            return (
                content
                if content is not None
                else f"error: {path!r} does not exist in {scope}"
            )

        def write_file(path: str, content: str) -> str:
            """Create or overwrite a file on your private branch.

            Args:
                path: Relative file path, e.g. "src/parser.py".
                content: The complete new file content.

            Returns:
                Confirmation.
            """
            written = org.workspace.write(me, path, content)
            return f"wrote {written} ({len(content)} chars) to your branch"

        def delete_file(path: str) -> str:
            """Delete a file on your private branch.

            Args:
                path: Relative file path.

            Returns:
                Confirmation.
            """
            existed = org.workspace.delete(me, path)
            return (
                f"deleted {path}"
                if existed
                else f"error: {path!r} does not exist on your branch"
            )

        def sync_main() -> str:
            """Bring the latest main into your branch, keeping your local changes.

            Files that both you and a peer changed get Git-style conflict markers;
            rewrite them to resolve before merging.

            Returns:
                The new base revision and any conflicted paths.
            """
            conflicts = org.workspace.sync(me)
            base = f"synced to main @ r{org.workspace.revision}"
            if not conflicts:
                return f"{base}; no conflicts"
            return f"{base}; resolve conflict markers in: {', '.join(conflicts)}"

        def merge_main(message: str) -> str:
            """Merge your verified branch into main and announce it to the team.

            Args:
                message: What changed, the idea, and the verification evidence.

            Returns:
                The new commit, or why the merge was refused.
            """
            result = org.workspace.merge(me, message)
            if not result.merged:
                listed = (
                    f" ({', '.join(result.conflicts)})"
                    if result.conflicts
                    else ""
                )
                return f"merge refused: {result.reason}{listed}"
            commit = result.commit
            files = ", ".join(sorted(commit.changes))
            org.messages.post(
                "workspace",
                f"{me} merged {commit.id} into main @ r{commit.revision}: "
                f"{commit.message} [files: {files}]",
            )
            return f"merged {commit.id} into main @ r{commit.revision} ({files})"

        def declare_goal_achieved(evidence: str) -> str:
            """Declare that the goal is fully achieved on main and stop working.

            Call only when main holds a complete deliverable and nothing valuable remains.

            Args:
                evidence: Why the goal is achieved, with verification evidence.

            Returns:
                Confirmation.
            """
            self.finished = True
            org.messages.post(
                me, f"declares the goal achieved: {evidence}"
            )
            return "recorded; finish your turn with a final status"

        tools = [
            board_write,
            board_read,
            board_grep,
            board_unfold,
            channel_post,
            direct_message,
            channel_history,
            list_files,
            read_file,
            write_file,
            delete_file,
            sync_main,
            merge_main,
            declare_goal_achieved,
        ]
        return [self._infrastructure(tool) for tool in tools]


class AgentSH:
    """A self-organized multi-agent organization with no central orchestrator.

    ``num_workers`` identical workers share one ``SharedWorkspace``, one
    ``MessageInterface`` and one ``SharedContext``, and each runs the
    cooperation loop on its own thread until it declares the goal achieved,
    exhausts ``max_turns``, or the ``time_budget`` expires. The deliverable
    is the ``main`` branch of the shared workspace.

    Args:
        name: Organization name; workers are ``f"{name}-worker{i}"``.
        num_workers: Number of concurrent workers.
        model_name: Any LiteLLM model string, shared by every worker.
        max_turns: Turns per worker; one turn is one ``Agent.run`` call.
        loops_per_turn: Maximum model/tool steps inside one turn.
        time_budget: Optional wall-clock budget in seconds. Checked between
            turns, so a turn already running is allowed to finish.
        stagger_seconds: Delay between worker activations, to reduce early
            scope contention (paper Appendix C).
        context_window: Recent board entries rendered into every turn prompt.
        channel_window: Most recent unseen channel messages per turn prompt.
        tools: Extra environment tools given to every worker (for example a
            test runner); these are not infrastructure tools.
        output_dir: If set, ``main`` is exported here after each run.
        print_on: Print each worker's agent output.
        agent_kwargs: Extra keyword arguments for every worker ``Agent``
            (e.g. ``temperature``, ``reasoning_effort``).

    Attributes:
        workspace: The shared workspace of the latest run.
        messages: The message interface of the latest run.
        context: The shared context board of the latest run.
        workers: The workers of the latest run.
    """

    def __init__(
        self,
        name: str = "agentsh",
        num_workers: int = 4,
        model_name: str = "gpt-5.4",
        max_turns: int = 6,
        loops_per_turn: int = 12,
        time_budget: Optional[float] = None,
        stagger_seconds: float = 0.0,
        context_window: int = 40,
        channel_window: int = 30,
        tools: Optional[List[Callable[..., Any]]] = None,
        output_dir: Optional[str] = None,
        print_on: bool = False,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        if max_turns < 1 or loops_per_turn < 1:
            raise ValueError(
                "max_turns and loops_per_turn must be at least 1"
            )
        self.name = name
        self.num_workers = num_workers
        self.model_name = model_name
        self.max_turns = max_turns
        self.loops_per_turn = loops_per_turn
        self.time_budget = time_budget
        self.stagger_seconds = stagger_seconds
        self.context_window = context_window
        self.channel_window = channel_window
        self.tools: List[Callable[..., Any]] = list(tools or [])
        self.output_dir = output_dir
        self.print_on = print_on
        self.agent_kwargs: Dict[str, Any] = dict(agent_kwargs or {})
        self.handles: List[str] = [
            f"{name}-worker{i}" for i in range(1, num_workers + 1)
        ]

        self.goal = ""
        self.workspace = SharedWorkspace()
        self.messages = MessageInterface()
        self.context = SharedContext()
        self.workers: List[Worker] = []
        self._started_at = 0.0

    def run(self, task: str) -> str:
        """Let the organization pursue ``task`` and return the final report.

        Every call starts a fresh organization: new infrastructure and new
        workers with empty conversations.

        Args:
            task: The shared user goal every worker pursues.

        Returns:
            A Markdown report of ``main`` (files, commit log, patch summaries).
        """
        if not task or not task.strip():
            raise ValueError("task must be a non-empty string")
        self.goal = task.strip()
        self.workspace = SharedWorkspace()
        self.messages = MessageInterface()
        self.context = SharedContext()
        self.messages.post("user", f"Goal: {self.goal}")
        self.workers = [
            Worker(handle, self) for handle in self.handles
        ]

        self._started_at = time.monotonic()
        logger.info(
            f"[{self.name}] {self.num_workers} workers start on: {self.goal[:80]}"
        )
        with ThreadPoolExecutor(
            max_workers=self.num_workers, thread_name_prefix=self.name
        ) as pool:
            futures = []
            for index, worker in enumerate(self.workers):
                if index and self.stagger_seconds > 0:
                    time.sleep(self.stagger_seconds)
                futures.append(pool.submit(self._work, worker))
            for future in as_completed(futures):
                future.result()

        if self.output_dir:
            path = self.workspace.export(self.output_dir)
            logger.info(f"[{self.name}] exported main to {path}")
        logger.info(
            f"[{self.name}] finished in {self.clock()} at main @ r{self.workspace.revision}"
        )
        return self.report()

    def worker_prompt(self, name: str) -> str:
        """Return the worker prompt, identical for every worker except ``name``."""
        verify_hint = " with the tools you have" if self.tools else ""
        return WORKER_PROMPT.format(
            name=name,
            peers=self.num_workers - 1,
            size=self.num_workers,
            handles=f"{self.name}-worker1..{self.num_workers}",
            goal=self.goal,
            verify_hint=verify_hint,
        )

    def clock(self) -> str:
        """Return elapsed (and, with a budget, total) time as a short string."""
        elapsed = (
            time.monotonic() - self._started_at
            if self._started_at
            else 0.0
        )
        if self.time_budget:
            return f"t={elapsed:.0f}s of {self.time_budget:.0f}s"
        return f"t={elapsed:.0f}s"

    def time_progress(self) -> float:
        """Return the fraction of ``time_budget`` used, or ``0.0`` with no budget."""
        if not self.time_budget or not self._started_at:
            return 0.0
        return (
            time.monotonic() - self._started_at
        ) / self.time_budget

    def report(self) -> str:
        """Render the current state of ``main`` as a Markdown report."""
        files = self.workspace.snapshot()
        commits = self.workspace.log()
        patches = [
            e
            for e in self.context.entries()
            if e.kind is EntryKind.PATCH_SUMMARY
        ]
        turns = sum(w.turns for w in self.workers)
        done = sum(w.finished for w in self.workers)

        lines = [
            f"# {self.name} report",
            "",
            f"**Goal:** {self.goal}",
            "",
            f"{self.num_workers} workers · {turns} turns · {len(commits)} commits · "
            f"{len(self.context.entries())} board entries · {done} declared done · {self.clock()}",
            "",
            f"## main @ r{self.workspace.revision}",
        ]
        if not files:
            lines += ["", "_main is empty._"]
        for path in sorted(files):
            lines += [
                "",
                f"### `{path}`",
                "",
                "````",
                files[path].rstrip("\n"),
                "````",
            ]
        lines += ["", "## Commit log", ""]
        lines += [
            f"- `{c.id}` r{c.revision} @{c.author}: {c.message}"
            for c in commits
        ] or ["_none_"]
        lines += ["", "## Patch summaries", ""]
        lines += [f"- @{e.author}: {e.summary}" for e in patches] or [
            "_none_"
        ]
        return "\n".join(lines)

    def _work(self, worker: Worker) -> None:
        """Drive one worker's turns until it finishes or the budget runs out."""
        while not worker.finished and worker.turns < self.max_turns:
            if self.time_progress() >= 1.0:
                break
            try:
                status = worker.run_turn()
            except Exception as error:
                logger.exception(
                    f"[{worker.name}] turn {worker.turns + 1} failed: {error}"
                )
                worker.turns += 1
                continue
            logger.info(
                f"[{worker.name}] turn {worker.turns}: {status[:160]}"
            )


if __name__ == "__main__":
    org = AgentSH(
        num_workers=3,
        model_name="gpt-5.4-mini",
        max_turns=3,
        output_dir="agentsh_output",
    )
    print(
        org.run(
            "Build a small pure-Python library `textstats` with functions for word "
            "count, sentence count, average word length and Flesch reading ease, a "
            "README.md documenting the API, and a test file using only the standard library."
        )
    )
