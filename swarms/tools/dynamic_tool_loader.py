"""
Load tools on demand instead of sending every schema on every request.

Tool definitions are part of the prompt. They are re-sent with each call and
they sit in the cached prefix, so a large tool set is paid for continuously:
the 16 built-in autonomous-loop tools alone are roughly 2,600 tokens per
request, and a single MCP server can add 40 more. Selection accuracy also
falls as the list grows - a model choosing among 80 tools chooses worse than
one choosing among 8.

:class:`DynamicToolLoader` keeps tools *deferred*: registered and executable,
but absent from the schema list sent to the model. Only one extra tool is
always present, ``tool_search``, which matches the catalog by name and
description and loads what it finds. Loaded tools stay loaded for the rest of
the run and become callable on the next request.

Example:
    >>> loader = DynamicToolLoader(tools=[get_weather, send_email, read_csv])
    >>> len(loader.schemas())          # only tool_search is exposed
    1
    >>> print(loader.run_search("weather"))
    get_weather: Get the current weather for a city.
    >>> len(loader.schemas())          # now tool_search + get_weather
    2

Wiring it to an agent takes two steps: pass ``loader.schemas()`` as the tool
list, and re-read it after each ``tool_search`` call so the newly loaded tools
are sent with the next request.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from loguru import logger

from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)

SEARCH_TOOL_NAME = "tool_search"

# Names listed back when a search finds nothing.
_MISS_LISTING_LIMIT = 30

SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Find and load tools you do not currently have. Most tools are "
            "deferred: their names and descriptions are searchable here, but "
            "you cannot call one until this loads it, and a loaded tool only "
            "becomes callable on your NEXT turn. So load everything you "
            "expect to need in a SINGLE call. Pass a plain query to search by "
            "keyword, or 'select:name_one,name_two' to load exact names. "
            "Before concluding that you cannot do something, search here."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Keywords to match against tool names and "
                        "descriptions, or 'select:name1,name2' for exact names."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum tools to load (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


DYNAMIC_TOOLS_NOTICE = """

## MOST TOOLS ARE NOT LOADED

Most of your tools are NOT currently loaded. Any tool list shown above
describes what EXISTS, not what you can call right now.

Right now you can call only the tools in your current tool list, plus
`tool_search`. Anything else - reading and writing files, running shell
commands, searching, delegating - must be loaded first:

1. Call `tool_search` with keywords describing what you need. Load everything
   you expect to need for the whole subtask in ONE call.
2. The loaded tools become callable on your NEXT turn.
3. Then do the work.

Never say a task cannot be done, and never mark a subtask complete by merely
describing what you would have done, without first searching for a tool.
If you are unsure what exists, call `tool_search` with a broad query - a miss
lists what is available.
"""


@dataclass
class DeferredTool:
    """One catalog entry: what it is, how to call it, and how to run it."""

    name: str
    description: str
    schema: Dict[str, Any]
    func: Optional[Callable] = None
    loaded: bool = False

    @property
    def summary(self) -> str:
        """One line, as shown in search results."""
        first = self.description.strip().split("\n")[0].strip()
        return f"{self.name}: {first}" if first else self.name

    @property
    def terms(self) -> List[str]:
        """Lowercased words this entry can be matched on."""
        params = (
            self.schema.get("function", {})
            .get("parameters", {})
            .get("properties", {})
        )
        text = f"{self.name} {self.description} {' '.join(params)}"
        return _tokenize(text)


# Filtering these matters more than it looks: without it a query like
# "weather in a city" matches every tool whose description contains "a",
# which loads the whole catalog and defeats the point of deferring.
_STOPWORDS = frozenset(
    """
    a an the and or of to in on for with from by at as is are be it its
    this that these those any some all please can could would should
    """.split()
)


def _tokenize(text: str) -> List[str]:
    """
    Split into lowercase word tokens, underscores treated as spaces.

    Single characters and common stopwords are dropped so that only terms
    carrying meaning can produce a match.
    """
    cleaned = "".join(
        c.lower() if c.isalnum() else " " for c in str(text)
    )
    return [
        t
        for t in cleaned.split()
        if len(t) > 1 and t not in _STOPWORDS
    ]


class DynamicToolLoader:
    """
    A catalog of deferred tools plus the ``tool_search`` tool that loads them.

    Args:
        tools: Callables to defer. Each is converted to an OpenAI function
            schema once, at registration.
        schemas: Pre-built schemas to defer, for tools that have no local
            callable (MCP tools, for instance).
        always_loaded: Schemas that are never deferred. Control-flow tools
            belong here - an agent that has to search for its own
            ``complete_task`` cannot finish.

    Note:
        Loading changes the tool list, which invalidates the provider's cached
        prompt prefix. Schemas are returned in a stable sorted order so that
        two runs loading the same tools produce an identical prefix, and the
        search tool's description pushes the model to load in one batch rather
        than one at a time.
    """

    def __init__(
        self,
        tools: Iterable[Callable] = (),
        schemas: Iterable[Dict[str, Any]] = (),
        always_loaded: Iterable[Dict[str, Any]] = (),
    ):
        self._catalog: Dict[str, DeferredTool] = {}
        self.always_loaded: List[Dict[str, Any]] = list(always_loaded)

        if tools:
            self.register(*tools)
        for schema in schemas:
            self.register_schema(schema)

    # ------------------------------------------------------------------
    # building the catalog
    # ------------------------------------------------------------------

    def register(self, *tools: Callable) -> "DynamicToolLoader":
        """Defer one or more Python callables."""
        tools = [t for t in tools if t is not None]
        if not tools:
            return self

        for func, schema in zip(
            tools,
            convert_multiple_functions_to_openai_function_schema(
                list(tools)
            ),
        ):
            self.register_schema(schema, func=func)
        return self

    def register_schema(
        self, schema: Dict[str, Any], func: Optional[Callable] = None
    ) -> "DynamicToolLoader":
        """Defer a pre-built OpenAI function schema."""
        function = schema.get("function", {})
        name = function.get("name")
        if not name:
            return self

        # A catalog entry with this name would collide with the search tool
        # itself: both would appear in the tool list and the model could not
        # tell which it was calling.
        if name == SEARCH_TOOL_NAME:
            logger.warning(
                f"Ignoring a tool named {SEARCH_TOOL_NAME!r}: that name is "
                "reserved for the dynamic tool search tool. Rename it to make "
                "it reachable."
            )
            return self

        self._catalog[name] = DeferredTool(
            name=name,
            description=function.get("description", ""),
            schema=schema,
            func=func,
        )
        return self

    # ------------------------------------------------------------------
    # searching and loading
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 5,
        min_score_ratio: float = 0.0,
    ) -> List[DeferredTool]:
        """
        Rank catalog entries against a query. Does not load anything.

        Matching is deliberately simple: token overlap, with a name match
        worth more than a description match. That is enough for the catalog
        sizes this is aimed at, has no dependencies, and is deterministic -
        so it can be tested. Swap in embeddings only when this measurably
        fails.

        Args:
            query: Keywords, or ``select:name1,name2`` for exact names.
            limit: Maximum results.
            min_score_ratio: Drop results scoring below this fraction of the
                best score. 0.0 keeps every match, which suits an explicit
                search where the model said what it wanted. Speculative
                callers should raise it: a long query contains enough common
                words to give weak matches a nonzero score.
        """
        query = (query or "").strip()

        if query.startswith("select:"):
            wanted = [
                n.strip() for n in query[len("select:") :].split(",")
            ]
            exact = [
                self._catalog[n] for n in wanted if n in self._catalog
            ]
            if exact:
                return exact
            # A miss usually means a guessed name, not a missing tool.
            query = " ".join(wanted)

        terms = _tokenize(query)
        if not terms:
            return []

        scored = []
        for tool in self._catalog.values():
            name_tokens = _tokenize(tool.name)
            haystack = tool.terms
            score = sum(
                3 if term in name_tokens else 1
                for term in terms
                if term in haystack
            )
            if score:
                scored.append((score, tool.name, tool))

        # Sort by score, then name, so results are stable run to run.
        scored.sort(key=lambda row: (-row[0], row[1]))

        if min_score_ratio > 0 and scored:
            cutoff = scored[0][0] * min_score_ratio
            scored = [row for row in scored if row[0] >= cutoff]

        return [tool for _, _, tool in scored[:limit]]

    def load(self, names: Iterable[str]) -> List[DeferredTool]:
        """Mark tools loaded. Returns only the ones newly loaded."""
        fresh = []
        for name in names:
            tool = self._catalog.get(name)
            if tool and not tool.loaded:
                tool.loaded = True
                fresh.append(tool)
        return fresh

    def run_search(
        self,
        query: str,
        max_results: int = 5,
        min_score_ratio: float = 0.0,
        **kwargs,
    ) -> str:
        """
        The ``tool_search`` handler: search, load, and report.

        The result is a compact listing rather than the full schemas, because
        the schemas themselves are already going out in the request's tool
        list - repeating them here would pay for them twice.
        """
        matches = self.search(
            query,
            limit=max_results or 5,
            min_score_ratio=min_score_ratio,
        )
        if not matches:
            # Listing what exists turns a miss into a usable next step: the
            # model can retry with 'select:'. Capped so a large catalog does
            # not dump hundreds of names into the conversation.
            names = sorted(self._catalog)
            shown = ", ".join(names[:_MISS_LISTING_LIMIT]) or "none"
            extra = (
                f" (+{len(names) - _MISS_LISTING_LIMIT} more)"
                if len(names) > _MISS_LISTING_LIMIT
                else ""
            )
            return (
                f"No tools matched {query!r}. Available tools: "
                f"{shown}{extra}. Retry with different keywords, or load by "
                f"exact name with 'select:name1,name2'."
            )

        fresh = self.load(tool.name for tool in matches)
        lines = [tool.summary for tool in matches]

        if fresh:
            lines.append("")
            lines.append(
                f"Loaded {len(fresh)}: "
                f"{', '.join(t.name for t in fresh)}. "
                "They are callable from your next turn."
            )
        else:
            lines.append("")
            lines.append("All already loaded - call them directly.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # what the model actually sees
    # ------------------------------------------------------------------

    def schemas(self) -> List[Dict[str, Any]]:
        """
        The tool list to send with the next request.

        ``always_loaded`` first, then ``tool_search``, then whatever has been
        loaded, name-sorted so the prefix is stable across runs.
        """
        loaded = sorted(
            (t for t in self._catalog.values() if t.loaded),
            key=lambda t: t.name,
        )
        return (
            list(self.always_loaded)
            + [SEARCH_TOOL_SCHEMA]
            + [t.schema for t in loaded]
        )

    def handlers(self) -> Dict[str, Callable]:
        """Name -> callable for every loaded tool that has one."""
        return {
            t.name: t.func
            for t in self._catalog.values()
            if t.loaded and t.func is not None
        }

    def catalog_listing(self) -> str:
        """Every deferred tool, one per line. Useful for prompts and debugging."""
        return "\n".join(
            tool.summary
            for tool in sorted(
                self._catalog.values(), key=lambda t: t.name
            )
        )

    @property
    def loaded_names(self) -> List[str]:
        return sorted(
            t.name for t in self._catalog.values() if t.loaded
        )

    @property
    def deferred_names(self) -> List[str]:
        return sorted(
            t.name for t in self._catalog.values() if not t.loaded
        )

    def __len__(self) -> int:
        return len(self._catalog)

    def __contains__(self, name: str) -> bool:
        return name in self._catalog

    def __repr__(self) -> str:
        return (
            f"DynamicToolLoader({len(self._catalog)} tools, "
            f"{len(self.loaded_names)} loaded)"
        )
