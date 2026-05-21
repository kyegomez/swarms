"""
Tips & tricks for the Swarms CLI.

Centralizes the rotating tip pool so the startup banner, the ``swarms tips``
subcommand, and any future "next step" prompts pull from the same source.

Every tip here is about using the **CLI** — commands, flags, env vars,
shortcuts. Framework / Python-API advice belongs in the docs, not here.

Tips are grouped by category. Each render randomizes both the prefix label
("Tip", "Pro tip", "Did you know?", etc.) and the tip body, giving the CLI a
Claude-Code-style varied feel.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Optional

from rich.text import Text


# ---------------------------------------------------------------------------
# Tip pools — CLI-focused only
# ---------------------------------------------------------------------------

COMMAND_TIPS: List[str] = [
    "Scaffold a new project with [bold]swarms init[/bold] — drops a .env and workspace dir for you",
    "Start chatting instantly with [bold]swarms chat[/bold]",
    "Verify your environment with [bold]swarms setup-check[/bold]",
    "Get a verbose diagnostic with [bold]swarms setup-check --verbose[/bold]",
    "See every command with [bold]swarms --help[/bold]",
    "Auto-build a swarm with [bold]swarms autoswarm --task '...'[/bold]",
    "Deep multi-agent analysis with [bold]swarms heavy-swarm --task '...'[/bold]",
    "Run a multi-model debate with [bold]swarms llm-council --task '...'[/bold]",
    "Load agents from YAML with [bold]swarms run-agents --yaml-file agents.yaml[/bold]",
    "Load agents from a folder of markdown files with [bold]swarms load-markdown --markdown-path ./agents/[/bold]",
    "Run a one-shot agent task with [bold]swarms agent --name '...' --task '...'[/bold]",
    "Upgrade the CLI with [bold]swarms upgrade[/bold]",
    "Get a fresh tip any time with [bold]swarms tips[/bold]",
    "See a bunch of tips at once with [bold]swarms tips --count 5[/bold]",
    "Filter tips by topic with [bold]swarms tips --category agents[/bold]",
    "Print every tip in a category with [bold]swarms tips --category pro --all[/bold]",
    "Re-run onboarding any time with [bold]swarms onboarding[/bold]",
    "Open the API-key portal with [bold]swarms get-api-key[/bold]",
    "Confirm you're logged in with [bold]swarms check-login[/bold]",
]

AGENT_FLAG_TIPS: List[str] = [
    "Let an agent decide when it's done with [bold]--max-loops auto[/bold]",
    "Give your agent a persona with [bold]--system-prompt '...'[/bold]",
    "Switch models per command with [bold]--model-name claude-opus-4-7[/bold]",
    "Use [bold]--temperature 0.1[/bold] for deterministic, factual responses",
    "Use [bold]--temperature 0.9[/bold] for creative, varied responses",
    "Stream tokens live with [bold]--streaming-on[/bold]",
    "See every step of an agent's run with [bold]--verbose[/bold]",
    "Combine [bold]--streaming-on --verbose[/bold] to watch the model think token by token",
    "Cap context with [bold]--context-length 32000[/bold] so long sessions stay snappy",
    "Persist agent state with [bold]--autosave --saved-state-path ./state.json[/bold]",
    "Resume an agent later by pointing [bold]--saved-state-path[/bold] at an existing file",
    "Use a marketplace prompt with [bold]--marketplace-prompt-id <id>[/bold]",
    "Tune retries on flaky models with [bold]--retry-attempts 3[/bold]",
    "Pick the response shape with [bold]--output-type json[/bold] (or [bold]str[/bold] / [bold]dict[/bold])",
    "Force non-interactive runs with [bold]--no-interactive[/bold] — handy for CI",
    "Connect an MCP tool server with [bold]--mcp-url http://localhost:8000/sse[/bold]",
    "Pop the live agent dashboard with [bold]--dashboard[/bold]",
    "Let the model self-tune temperature with [bold]--dynamic-temperature-enabled[/bold]",
    "Let the model self-extend context with [bold]--dynamic-context-window[/bold]",
    "Tag runs with [bold]--user-name '...'[/bold] for cleaner logs",
]

SWARM_FLAG_TIPS: List[str] = [
    "[bold]swarms heavy-swarm --loops-per-agent 5[/bold] gives each specialist room to think",
    "Make heavy-swarms non-deterministic with [bold]--random-loops-per-agent[/bold]",
    "Choose the heavy-swarm worker model with [bold]--worker-model-name claude-opus-4-7[/bold]",
    "Choose the heavy-swarm question agent with [bold]--question-agent-model-name gpt-4o[/bold]",
    "Generate a swarm spec without running it: [bold]swarms autoswarm --task '...' --no-run[/bold]",
    "Save the autoswarm output anywhere with [bold]swarms autoswarm -o ./my_swarm.py[/bold]",
    "Drop autoswarm output in a folder with [bold]swarms autoswarm -d ./swarms/[/bold]",
    "Tweak autoswarm's planner with [bold]swarms autoswarm --model gpt-4o --task '...'[/bold]",
    "Run a sequential YAML pipeline: [bold]swarms run-agents --yaml-file pipeline.yaml[/bold]",
    "Process many markdown agents in parallel — [bold]--concurrent[/bold] is on by default",
]

CLI_PRO_TRICKS: List[str] = [
    "Pipe a task from stdin: [bold]echo 'summarize this' | swarms agent --name X --task -[/bold]",
    "Run long autonomous loops inside [bold]tmux[/bold] so they survive a disconnect",
    "Combine [bold]--autosave[/bold] with a cron job for self-archiving daily agents",
    "Capture full output to a file: [bold]swarms agent ... > run.log 2>&1[/bold]",
    "Use [bold]direnv[/bold] to scope API keys per project — Swarms picks up .env automatically",
    "Diagnose any issue with [bold]swarms setup-check --verbose[/bold] before opening a bug report",
    "Hit Ctrl-C once to interrupt a chat — your session state is autosaved",
    "Re-launch the banner any time by running [bold]swarms[/bold] with no args",
    "Quick model A/B: run the same task with two different [bold]--model-name[/bold] values",
]

DID_YOU_KNOW: List[str] = [
    "The startup banner picks a fresh tip every time — just run [bold]swarms[/bold]",
    "[bold]swarms tips[/bold] is its own command — get hints whenever you want",
    "Eight tip categories ship out of the box: [bold]commands, agents, swarms, models, pro, trivia, env, community[/bold]",
    "[bold]swarms tips --category pro --all[/bold] dumps every power-user trick",
    "The banner shows which provider is active based on the API keys in your environment",
    "Every CLI command has [bold]--help[/bold] flags surfaced via [bold]swarms --help[/bold]",
    "[bold]swarms init[/bold] creates a [bold].env[/bold], [bold]agents.yaml[/bold], and a workspace dir in one shot",
]

ENV_TIPS: List[str] = [
    "Store your API keys in [bold].env[/bold] — Swarms loads it automatically from the cwd",
    "Set [bold]WORKSPACE_DIR[/bold] to control where agents read and write files",
    "Set [bold]SWARMS_VERBOSE=1[/bold] for detailed logging on every CLI run",
    "Run [bold]swarms setup-check --verbose[/bold] to spot missing env vars",
    "Set [bold]OPENAI_API_KEY[/bold] and [bold]ANTHROPIC_API_KEY[/bold] together — Swarms picks per-agent",
    "Override the workspace per command with [bold]WORKSPACE_DIR=/tmp/runX swarms agent ...[/bold]",
    "[bold]~/.swarms/[/bold] holds conversation logs and saved agent state",
]

MODEL_FLAG_TIPS: List[str] = [
    "Any LiteLLM-supported model works in [bold]--model-name[/bold] — OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere…",
    "[bold]--model-name claude-opus-4-7[/bold] picks the latest frontier Anthropic model",
    "[bold]--model-name groq/llama-3.3-70b-versatile[/bold] is the fastest open-weight option",
    "[bold]--model-name gemini/gemini-2.5-pro[/bold] gives you million-token context natively",
    "Local models work via Ollama or vLLM — point [bold]--model-name[/bold] at the endpoint",
    "Different commands have different defaults: [bold]swarms agent[/bold] uses gpt-4, [bold]swarms chat[/bold] uses gpt-5.4",
]

COMMUNITY_TIPS: List[str] = [
    "Star the repo at [bold]https://github.com/kyegomez/swarms[/bold]",
    "Join the Discord at [bold]https://discord.gg/EamjgSaEQf[/bold]",
    "Full CLI docs at [bold]https://docs.swarms.world/cli/overview[/bold]",
    "Report a CLI bug at [bold]github.com/kyegomez/swarms/issues[/bold]",
    "Browse the prompt marketplace at [bold]https://swarms.world/prompts[/bold] — use IDs with [bold]--marketplace-prompt-id[/bold]",
]


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

TIP_CATEGORIES: Dict[str, List[str]] = {
    "commands": COMMAND_TIPS,
    "agents": AGENT_FLAG_TIPS,
    "swarms": SWARM_FLAG_TIPS,
    "models": MODEL_FLAG_TIPS,
    "pro": CLI_PRO_TRICKS,
    "trivia": DID_YOU_KNOW,
    "env": ENV_TIPS,
    "community": COMMUNITY_TIPS,
}


# ---------------------------------------------------------------------------
# Random labels — Claude-Code-style variety
# ---------------------------------------------------------------------------

# (emoji, label, accent style). Sampling these gives the startup banner the
# "feels fresh every time" quality.
TIP_LABELS: List[tuple[str, str, str]] = [
    ("※", "Tip", "bold red"),
    ("⚡", "Pro tip", "bold yellow"),
    ("💡", "Did you know", "bold cyan"),
    ("🧠", "Trick", "bold magenta"),
    ("✨", "Insider", "bold white"),
    ("🎯", "Heads up", "bold green"),
    ("🪄", "Hint", "bold blue"),
    ("🔥", "Hot tip", "bold red"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _all_tips(category: Optional[str] = None) -> List[str]:
    """Return tips for a specific category, or the union of all categories."""
    if category is None:
        return [
            tip for tips in TIP_CATEGORIES.values() for tip in tips
        ]
    key = category.lower()
    if key not in TIP_CATEGORIES:
        raise ValueError(
            f"Unknown tip category '{category}'. "
            f"Valid: {', '.join(sorted(TIP_CATEGORIES))}"
        )
    return TIP_CATEGORIES[key]


def pick_random_label() -> tuple[str, str, str]:
    """Pick a random ``(emoji, label, style)`` triple for the next tip."""
    return random.choice(TIP_LABELS)


def pick_random_tip(category: Optional[str] = None) -> str:
    """Return one random tip string (already includes inline markup)."""
    return random.choice(_all_tips(category))


def render_tip(
    body: Optional[str] = None,
    category: Optional[str] = None,
) -> Text:
    """
    Build a Rich ``Text`` with a randomized prefix label and a random tip body.

    Args:
        body: Optional explicit tip string. If omitted, one is picked from
            the registry (filtered by ``category`` if provided).
        category: Restrict random selection to one of ``TIP_CATEGORIES``.
            Ignored when ``body`` is given.

    Returns:
        Rich ``Text`` ready to ``console.print(...)``.
    """
    emoji, label, style = pick_random_label()
    tip_body = body if body is not None else pick_random_tip(category)
    return Text.from_markup(
        f"[{style}] {emoji} {label}:[/{style}]  [white]{tip_body}[/white]"
    )


def render_random_tips(
    n: int = 1,
    category: Optional[str] = None,
) -> List[Text]:
    """Return ``n`` distinct random tips, each with its own random label."""
    pool = list(_all_tips(category))
    n = max(1, min(n, len(pool)))
    chosen: Iterable[str] = random.sample(pool, n)
    return [render_tip(body=tip) for tip in chosen]


def list_categories() -> List[str]:
    """Return the supported category keys, alphabetically."""
    return sorted(TIP_CATEGORIES)
