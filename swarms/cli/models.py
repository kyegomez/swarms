"""
Model discovery for the Swarms CLI.

Backs the ``swarms models`` command:

- ``swarms models``                  list every model, grouped by provider
- ``swarms models --provider X``     restrict the list to one provider
- ``swarms models --search opus``    fuzzy-search by name
- ``swarms models --info <name>``    show context window, capabilities, pricing

All metadata comes from LiteLLM (``litellm.model_list`` and
``litellm.get_model_info``), so this stays in sync as providers ship models.
"""

from __future__ import annotations

from collections import defaultdict
from difflib import get_close_matches
from typing import Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


def _provider_of(name: str) -> str:
    """Extract a provider key from a LiteLLM model string."""
    if "/" in name:
        return name.split("/", 1)[0]
    # Heuristics for the common no-prefix providers
    lowered = name.lower()
    if lowered.startswith(
        ("gpt-", "o1", "o3", "o4", "chatgpt", "text-")
    ):
        return "openai"
    if lowered.startswith(("claude", "anthropic")):
        return "anthropic"
    if lowered.startswith("gemini"):
        return "google"
    if lowered.startswith(("command", "cohere")):
        return "cohere"
    if lowered.startswith("mistral"):
        return "mistral"
    return "other"


def _load_model_list() -> List[str]:
    """Lazy import of LiteLLM to avoid slowing every CLI invocation."""
    from litellm import model_list

    return list(model_list)


def _group_by_provider(names: List[str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = defaultdict(list)
    for name in names:
        grouped[_provider_of(name)].append(name)
    for v in grouped.values():
        v.sort()
    return dict(sorted(grouped.items()))


def _format_cost(cost: Optional[float]) -> str:
    """Convert per-token cost (float) to a human-friendly per-1M-tokens string."""
    if cost is None:
        return "—"
    try:
        return f"${cost * 1_000_000:,.2f} / 1M"
    except (TypeError, ValueError):
        return "—"


def _format_int(value: Optional[int]) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def list_models(
    console: Console, provider: Optional[str] = None
) -> None:
    """Print every model, grouped by provider (or filtered to one)."""
    models = _load_model_list()
    grouped = _group_by_provider(models)

    if provider:
        key = provider.lower()
        grouped = {k: v for k, v in grouped.items() if k == key}
        if not grouped:
            console.print(
                f"[bold red]No models found for provider '{provider}'.[/bold red]\n"
                f"[dim white]Run [bold]swarms models[/bold] (no flag) to see every provider.[/dim white]"
            )
            return

    total = sum(len(v) for v in grouped.values())
    console.print(
        Panel(
            f"[bold white]{total:,}[/bold white] models across "
            f"[bold white]{len(grouped)}[/bold white] provider(s)\n"
            "[dim white]Use [bold]--search <pattern>[/bold] to filter, "
            "or [bold]--info <name>[/bold] for full details.[/dim white]",
            border_style="red",
            title="[bold red] swarms models [/bold red]",
            title_align="left",
            padding=(0, 2),
        )
    )

    for prov, names in grouped.items():
        table = Table(
            title=f"[bold red]{prov}[/bold red]  [dim white]({len(names)})[/dim white]",
            title_justify="left",
            border_style="dim red",
            show_header=False,
            padding=(0, 1),
        )
        # Lay out 3 columns of names so the output is dense
        cols = 3
        for _ in range(cols):
            table.add_column(no_wrap=False)
        for i in range(0, len(names), cols):
            row = names[i : i + cols]
            row += [""] * (cols - len(row))
            table.add_row(*row)
        console.print(table)


def search_models(
    console: Console, pattern: str, limit: int = 40
) -> None:
    """Fuzzy + substring search the model list."""
    if not pattern:
        console.print(
            "[bold red]--search needs a pattern, e.g. [bold]swarms models --search opus[/bold][/bold red]"
        )
        return

    models = _load_model_list()
    lowered = pattern.lower()

    # Exact substring first, then fuzzy fillers to round out short results
    substring = [m for m in models if lowered in m.lower()]
    fuzzy = [
        m
        for m in get_close_matches(
            pattern, models, n=limit, cutoff=0.5
        )
        if m not in substring
    ]
    matches = (substring + fuzzy)[:limit]

    if not matches:
        console.print(
            f"[bold red]No models match '[bold]{pattern}[/bold]'.[/bold red]\n"
            f"[dim white]Try [bold]swarms models[/bold] for the full list.[/dim white]"
        )
        return

    table = Table(
        title=f"[bold red]matches for '{pattern}'[/bold red]  "
        f"[dim white]({len(matches)} shown)[/dim white]",
        title_justify="left",
        border_style="dim red",
        show_header=True,
        header_style="bold white",
        padding=(0, 1),
    )
    table.add_column("model", no_wrap=False)
    table.add_column("provider", style="dim white")
    for m in matches:
        table.add_row(m, _provider_of(m))
    console.print(table)
    if len(matches) >= limit:
        console.print(
            f"[dim white]Showing first {limit} matches. Narrow with a more specific pattern.[/dim white]"
        )


def show_model_info(console: Console, name: str) -> None:
    """Detail view: context window, capabilities, pricing for one model."""
    if not name:
        console.print(
            "[bold red]--info needs a model name, e.g. [bold]swarms models --info gpt-4o[/bold][/bold red]"
        )
        return

    from litellm import get_model_info

    try:
        info = get_model_info(name)
    except Exception as e:
        suggestions = get_close_matches(
            name, _load_model_list(), n=5, cutoff=0.4
        )
        msg = f"[bold red]Could not load info for '[bold]{name}[/bold]'.[/bold red]\n[dim white]{e}[/dim white]"
        if suggestions:
            msg += "\n\n[bold white]Did you mean:[/bold white]\n"
            for s in suggestions:
                msg += f"  • [bold]{s}[/bold]\n"
        console.print(msg)
        return

    table = Table(
        title=f"[bold red]{name}[/bold red]",
        title_justify="left",
        border_style="dim red",
        show_header=False,
        padding=(0, 1),
    )
    table.add_column(style="dim white", no_wrap=True)
    table.add_column()

    rows = [
        ("Provider", str(info.get("litellm_provider") or "—")),
        ("Mode", str(info.get("mode") or "—")),
        (
            "Max input tokens",
            _format_int(info.get("max_input_tokens")),
        ),
        (
            "Max output tokens",
            _format_int(info.get("max_output_tokens")),
        ),
        (
            "Input cost",
            _format_cost(info.get("input_cost_per_token")),
        ),
        (
            "Output cost",
            _format_cost(info.get("output_cost_per_token")),
        ),
        (
            "Cache read cost",
            _format_cost(info.get("cache_read_input_token_cost")),
        ),
        (
            "Function calling",
            "✓" if info.get("supports_function_calling") else "—",
        ),
        (
            "Vision",
            "✓" if info.get("supports_vision") else "—",
        ),
        (
            "System messages",
            "✓" if info.get("supports_system_messages") else "—",
        ),
        (
            "Streaming",
            (
                "✓"
                if info.get("supports_response_schema") is not None
                else "—"
            ),
        ),
    ]
    for label, value in rows:
        table.add_row(label, value)
    console.print(table)
    console.print(
        f"\n[dim white]Use this model with [bold]--model-name {name}[/bold] on any swarms command.[/dim white]"
    )
