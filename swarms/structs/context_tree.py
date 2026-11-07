"""Scoped, budgeted notes agents share instead of a conversation."""

import json
import math
import re
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

SCOPES = ("level", "tree", "global", "cohort")

# Cohort shares the depth but not the task, so it gets the least.
WEIGHTS = (3, 3, 3, 1)

HEADINGS = {
    "level": "Notes from your level of this task",
    "tree": "Notes from this task",
    "global": "Notes from every task",
    "cohort": "Notes from your level of any task",
}

MAX_NOTE_CHARS = 400

REASONING = re.compile(
    r"\b(?:because i|let me think|first i)\b", re.IGNORECASE
)


def _tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _split_budget(
    budget: int, demands: List[int], weights: List[int]
) -> List[int]:
    """Split a token budget across scopes by weighted water-filling.

    Scopes are filled from the least demand per weight up. Each gets
    its demand or its weighted share of what is left, whichever is
    smaller, so a scope that needs less passes the rest on. The last
    scope filled takes the tokens rounding leaves.
    """
    shares = [0] * len(demands)
    left, total = budget, sum(weights)
    order = sorted(
        range(len(demands)), key=lambda i: demands[i] / weights[i]
    )
    for i in order:
        shares[i] = min(demands[i], left * weights[i] // total)
        left -= shares[i]
        total -= weights[i]
    return shares


def _visible(note: dict, tree_id: str, depth: int) -> bool:
    same_tree = note["tree_id"] == tree_id
    same_depth = note["depth"] == depth
    return {
        "level": same_tree and same_depth,
        "tree": same_tree,
        "global": True,
        "cohort": same_depth,
    }[note["scope"]]


class ContextTree:
    """Short notes agents share, each read by the agents it concerns.

    A tree is one task, named by an id, and depth is how far an agent
    sits below the one that started it. A note's scope decides who
    reads it: level is the same depth of the same tree, tree is the
    whole tree, global is every tree, and cohort is the same depth of
    any tree.

    Args:
        spec: The task statement. It leads every render.
        path: A JSON lines file. Notes are loaded from it and appended
            to it, so global and cohort notes outlive a run.
        gate: Called as gate(author, text) before a note is saved. It
            returns (allowed, reason); a blocked note is rejected.
        token_budget: Tokens of notes that read and render return.

    Attributes:
        notes: Every note, oldest first, as dicts with scope, tree_id,
            depth, author and text.
    """

    def __init__(
        self,
        spec: str = "",
        path: Optional[str] = None,
        gate: Optional[Callable[[str, str], Tuple[bool, str]]] = None,
        token_budget: int = 1500,
    ):
        self.spec = spec
        self.path = Path(path) if path else None
        self.gate = gate
        self.token_budget = token_budget
        self.notes: List[dict] = []
        self._lock = threading.Lock()
        if self.path is not None and self.path.exists():
            lines = self.path.read_text(encoding="utf-8").splitlines()
            self.notes = [json.loads(line) for line in lines if line]

    def write(
        self,
        text: str,
        scope: str = "tree",
        tree_id: str = "default",
        depth: int = 0,
        author: str = "",
    ) -> None:
        """Save one note.

        Args:
            text: One observation, at most 400 characters.
            scope: Who reads it: level, tree, global or cohort.
            tree_id: The writer's task.
            depth: The writer's depth in that task.
            author: The writer's name, passed to the gate.

        Raises:
            ValueError: If the scope is unknown, the text is empty,
                long or reasoning, or the gate blocks it.
        """
        if scope not in SCOPES:
            raise ValueError(
                f"Unknown scope '{scope}'. Use one of: {', '.join(SCOPES)}."
            )
        text = text.strip() if isinstance(text, str) else ""
        if not text:
            raise ValueError("A note must be non-empty text.")
        if len(text) > MAX_NOTE_CHARS or REASONING.search(text):
            raise ValueError(
                f"A note is one finding of at most {MAX_NOTE_CHARS} "
                "characters, not the reasoning behind it."
            )
        if self.gate is not None:
            allowed, reason = self.gate(author, text)
            if not allowed:
                raise ValueError(f"Blocked by the gate: {reason}")
        note = {
            "scope": scope,
            "tree_id": tree_id,
            "depth": depth,
            "author": author,
            "text": text,
        }
        with self._lock:
            self.notes.append(note)
            if self.path is not None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(note) + "\n")

    def read(
        self,
        tree_id: str = "default",
        depth: int = 0,
        token_budget: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """The notes one reader sees, by scope, newest first.

        Level, tree and global get three parts of the budget each and
        cohort one, and a scope that needs less passes the rest on.
        Each scope keeps its newest whole notes that fit; none is cut.

        Args:
            tree_id: The reader's task.
            depth: The reader's depth in that task.
            token_budget: Tokens of notes. Defaults to the tree's.

        Returns:
            Dict[str, List[str]]: Note texts for each scope.
        """
        if token_budget is None:
            token_budget = self.token_budget
        seen = {scope: [] for scope in SCOPES}
        for note in reversed(self.notes):
            if _visible(note, tree_id, depth):
                seen[note["scope"]].append(note["text"])
        demands = [sum(map(_tokens, seen[s])) for s in SCOPES]
        shares = _split_budget(token_budget, demands, WEIGHTS)
        kept = {}
        for scope, share in zip(SCOPES, shares):
            kept[scope] = []
            for text in seen[scope]:
                share -= _tokens(text)
                # Stop, not skip, so no newer note is left out.
                if share < 0:
                    break
                kept[scope].append(text)
        return kept

    def render(
        self,
        tree_id: str = "default",
        depth: int = 0,
        token_budget: Optional[int] = None,
    ) -> str:
        """The spec, then one reader's notes, as a block for a prompt.

        The budget never trims the spec. An agent that can see the
        spec rarely follows a teammate's note that contradicts it; one
        that cannot often does.

        Args:
            tree_id: The reader's task.
            depth: The reader's depth in that task.
            token_budget: Tokens of notes. The spec is not counted.

        Returns:
            str: The spec and a headed list per scope that has notes.
        """
        sections = [f"Spec:\n{self.spec}"] if self.spec else []
        notes = self.read(tree_id, depth, token_budget)
        for scope, texts in notes.items():
            if texts:
                lines = "\n".join(f"- {text}" for text in texts)
                sections.append(f"{HEADINGS[scope]}:\n{lines}")
        return "\n\n".join(sections)

    def tools(
        self, author: str, tree_id: str = "default", depth: int = 0
    ) -> List[Callable]:
        """context_write and context_read for one agent's tools.

        Args:
            author: The agent's name, recorded on its notes.
            tree_id: The agent's task.
            depth: The agent's depth in that task.

        Returns:
            List[Callable]: The two tools, for Agent(tools=...).
        """

        def context_write(note: str, scope: str = "tree") -> str:
            """Save one finding for your teammates.

            A fact, a number or a decision in at most 400 characters,
            not your reasoning.

            Args:
                note: The finding.
                scope: Who reads it: "level" (your level of this
                    task), "tree" (this task), "global" (every task)
                    or "cohort" (your level of any task).

            Returns:
                str: "Saved." or "Rejected: " and the reason.
            """
            try:
                self.write(note, scope, tree_id, depth, author)
            except ValueError as error:
                return f"Rejected: {error}"
            return "Saved."

        def context_read() -> str:
            """Read the task spec and your teammates' latest notes.

            Returns:
                str: The spec, then the notes, newest first.
            """
            return self.render(tree_id, depth) or "No notes yet."

        return [context_write, context_read]

    def clear_tree(self, tree_id: str) -> None:
        """Drop a task's level and tree notes, keep global and cohort.

        Args:
            tree_id: The finished task.
        """
        with self._lock:
            self.notes = [
                note
                for note in self.notes
                if note["tree_id"] != tree_id
                or note["scope"] in ("global", "cohort")
            ]
            if self.path is not None and self.path.exists():
                self.path.write_text(
                    "".join(json.dumps(n) + "\n" for n in self.notes),
                    encoding="utf-8",
                )
