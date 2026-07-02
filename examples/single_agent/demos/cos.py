"""Chief of Staff Swarm (issue #1006). PlannerWorkerSwarm that
accepts a high-level task, decomposes via gpt-4o-mini planner, gates
on human approval, dispatches to 5 MCP-equipped gpt-4.1 workers.
Boots 6 npm MCP servers (filesystem, memory, git, reasoning, excel,
discovery) via `npx -y` + mcp-proxy stdio-to-HTTP bridge. Missing
servers are skipped; the swarm still runs.
"""
import atexit
import math
import os
import readline
import subprocess
import sys
import time

import httpx
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from dotenv import load_dotenv
from loguru import logger

from swarms import Agent
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm
from swarms.utils.litellm_wrapper import LiteLLM

_console = Console()

_orig_output_for_tools = LiteLLM.output_for_tools
_orig_output_for_reasoning = LiteLLM.output_for_reasoning

def _first_message_text(response):
    try:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return getattr(msg, "content", "") if msg else ""
    except Exception:
        return ""

def _has_tool_calls(response):
    try:
        for choice in getattr(response, "choices", []) or []:
            msg = getattr(choice, "message", None)
            if msg and getattr(msg, "tool_calls", None):
                return True
    except Exception:
        pass
    return False

def _patched_output_for_tools(self, response):
    """Workaround: LLM returning text crashes swarms' tool_calls handling."""
    if not _has_tool_calls(response):
        return []
    return _orig_output_for_tools(self, response)

def _patched_output_for_reasoning(self, response):
    """Workaround for IndexError when response.choices is empty."""
    try:
        if not getattr(response, "choices", None):
            return _first_message_text(response)
    except Exception:
        return _first_message_text(response)
    return _orig_output_for_reasoning(self, response)

LiteLLM.output_for_tools = _patched_output_for_tools
LiteLLM.output_for_reasoning = _patched_output_for_reasoning

load_dotenv()

logger.disable("swarms.tools.mcp_client_tools")
logger.disable("swarms.structs.agent")

_ENV = os.getenv
MCP_CONFIG = {
    "computer_use": _ENV("COS_MCP_COMPUTER_URL",  "http://localhost:8010/mcp"),
    "memory":       _ENV("COS_MCP_MEMORY_URL",    "http://localhost:8011/mcp"),
    "git":          _ENV("COS_MCP_GIT_URL",       "http://localhost:8012/mcp"),
    "reasoning":    _ENV("COS_MCP_REASONING_URL", "http://localhost:8013/mcp"),
    "excel":        _ENV("COS_MCP_EXCEL_URL",     "http://localhost:8014/mcp"),
    "discovery":    _ENV("COS_MCP_DISCOVERY_URL", "http://localhost:8015/mcp"),
}
PLANNER_MODEL = _ENV("COS_PLANNER_MODEL", "gpt-4o-mini")
WORKER_MODEL = _ENV("COS_WORKER_MODEL",  "gpt-4.1")
JUDGE_MODEL  = _ENV("COS_JUDGE_MODEL",   "gpt-4o-mini")

MODEL_RATES = {"gpt-4.1": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
VERBS = {
    "research", "find", "analyze", "compare", "summarize", "write",
    "build", "create", "deploy", "fix", "refactor", "test", "review",
    "search", "scrape", "list", "fetch", "download", "upload", "send",
    "open", "close", "delete", "update", "merge", "push", "commit",
    "investigate", "design", "plan", "schedule", "monitor", "check",
}
def mcp_urls_for(*categories):
    return [MCP_CONFIG[c] for c in categories if c in MCP_CONFIG]


def adaptive_max_loops(task, num_workers):
    n = sum(1 for w in task.lower().split() if w in VERBS)
    return max(1, min(5, math.ceil(math.sqrt(num_workers * n) / 2)))

def estimate_cost(task, num_workers, max_loops):
    in_tok = max(len(task.split()) * 2, 256)
    out_tok = 2048
    p_in, p_out = MODEL_RATES.get(PLANNER_MODEL, MODEL_RATES["gpt-4.1"])
    w_in, w_out = MODEL_RATES.get(WORKER_MODEL, MODEL_RATES["gpt-4o-mini"])
    wc = max_loops * num_workers * 3
    in_cost = in_tok * (max_loops * 2 + wc) * p_in / 1_000_000
    in_cost += in_tok * wc * w_in / 1_000_000
    out_cost = out_tok * (max_loops * 2 + wc) * p_out / 1_000_000
    out_cost += out_tok * wc * w_out / 1_000_000
    total = in_cost + out_cost
    est_min = max_loops * num_workers * 0.5
    return (
        f"Estimated cost: ${total:.2f}  |  "
        f"Estimated runtime: ~{est_min:.0f} min  |  Workers: {num_workers}"
    )

def compute_reliability(completed, total, avg_quality, edit_count):
    if total == 0:
        return 0.0
    return (
        (completed / total)
        * max(0.0, min(1.0, avg_quality / 10.0))
        / (1.0 + edit_count)
    )

_FILESYSTEM_CWD = os.getcwd()
MCP_SERVERS = [
    ("computer_use", ["npx", "-y", "@modelcontextprotocol/server-filesystem", _FILESYSTEM_CWD], 8010),
    ("memory",       ["npx", "-y", "@modelcontextprotocol/server-memory"], 8011),
    ("git",          ["npx", "-y", "@cyanheads/git-mcp-server"], 8012),
    ("reasoning",    ["npx", "-y", "@modelcontextprotocol/server-sequential-thinking"], 8013),
    ("excel",        ["npx", "-y", "@negokaz/excel-mcp-server"], 8014),
    ("discovery",    ["npx", "-y", "@modelcontextprotocol/server-everything"], 8015),
]

class StdioToHttpBridge:
    """Spawns a stdio MCP server through `npx mcp-proxy` on a port."""

    def __init__(self, cmd, port):
        self.cmd = cmd
        self.port = port
        self.proc = None

    def start(self):
        argv = [
            "npx", "-y", "mcp-proxy", "--port", str(self.port), "--",
        ] + self.cmd
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

def _probe(url, timeout=1.0):
    """HEAD-probe one URL; never raises."""
    try:
        return httpx.head(url, timeout=timeout).status_code < 500
    except Exception:
        return False

def _wait_for_health(url, timeout=10):
    """Poll URL until up or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _probe(url, timeout=0.5):
            return True
        time.sleep(0.5)
    return False

_started_bridges = []

@atexit.register
def _teardown_bridges():
    for b in _started_bridges:
        try:
            b.stop()
        except Exception:
            pass

def bootstrap_mcp_servers(timeout=10):
    """Idempotent: spawn each MCP server via npx + mcp-proxy; npx -y auto-installs."""
    results = {}
    for category, npx_cmd, port in MCP_SERVERS:
        url = f"http://localhost:{port}/mcp"
        if _probe(url):
            results[category] = True
            logger.info(f"{category}: already up at {url}")
            continue
        try:
            bridge = StdioToHttpBridge(cmd=npx_cmd, port=port)
            bridge.start()
            _started_bridges.append(bridge)
            if _wait_for_health(url, timeout=timeout):
                results[category] = True
                logger.info(f"{category}: ready at {url}")
            else:
                logger.warning(f"{category}: health check timed out")
                bridge.stop()
                results[category] = False
        except Exception as e:
            logger.warning(f"{category}: bridge failed: {e}")
            results[category] = False
    return results

CHIEF_OF_STAFF_RULES = """
[Chief of Staff rules]
0. CRITICAL: ALWAYS call a tool to perform an action. Never just
   describe. If the task says "create a file", you MUST call
   write_file/edit_file. If "run X", you MUST call the shell tool.
   A text-only response is a failure. Expand ~ in paths yourself.
1. Spawn sub-agents via create_sub_agent + assign_task when a task
   needs a specialist you do not cover.
2. Destructive actions (git push, docker rm, shell rm -rf, public PR)
   require human approval via respond_to_user before executing.
3. Don't redo work another worker already did. Check short_memory
   and conversation history before starting.
4. On unrecoverable error, fail fast with a clear message so the
   judge can decide whether to gap-fill or restart.
"""

RESEARCHER_PROMPT = """You are a research specialist. Gather, verify, and
synthesize information from the web and MCP discovery servers.
Prefer primary sources (docs, GitHub READMEs, arXiv). When sources
disagree, surface the disagreement. Return a structured finding:
## <claim> / Evidence / Sources / Confidence: low|medium|high"""

CODER_PROMPT = """You are a coding specialist. Write, run, refactor, and
test code; manage local files; perform git operations. Read existing
files before editing. Prefer small reversible changes. Commit often.
Use the git MCP for read operations; shell for local builds/tests.
Never push to main/master. Spawn sub-agents via create_sub_agent
when a task outgrows a single response."""

GITHUB_AGENT_PROMPT = """You are a GitHub specialist. Handle PRs, issues,
code review, releases, and repo metadata via the git/github MCP.
Read PR/issue body, linked code, and CI status before commenting.
Be specific in reviews: quote the line, suggest a concrete change.
Never merge, never push to protected branches. Use sub-agents for
specialised review passes."""

def make_worker_agent(
    name, description, system_prompt, mcp_categories=(),
):
    """Factory for the default workers; falls back to no-MCP if unreachable."""
    base = dict(
        agent_name=name,
        agent_description=description,
        system_prompt=system_prompt + "\n\n" + CHIEF_OF_STAFF_RULES,
        model_name=WORKER_MODEL,
        max_loops=1,
        max_tokens=16384,
        selected_tools="all",
        dynamic_temperature_enabled=True,
        dynamic_context_window=True,
        context_length=128000,
        context_compression=True,
        persistent_memory=True,
        autosave=True,
        verbose=True,
        print_on=False,
    )
    urls = mcp_urls_for(*mcp_categories)
    if not urls:
        return Agent(**base)
    try:
        return Agent(mcp_urls=urls, **base)
    except Exception as e:
        logger.warning(f"{name}: MCP unreachable ({type(e).__name__}); building without MCP tools")
        return Agent(**base)

_MEMORY_PROMPT = """You are the memory specialist in a
chief-of-staff swarm. Maintain a persistent knowledge graph of
entities, relations, and observations across sessions. Capture
project names, contacts, preferences, deadlines, conventions.
Prefer specific names over pronouns. Query the graph first on
recall, then return a concise summary. Never fabricate relations."""

_EXCEL_PROMPT = """You are an Excel specialist in a chief-of-staff
swarm. Read, write, and analyse spreadsheet data on behalf of the
principal. Read user intent before touching a file. Use formulas
(not hard-coded values) when the result depends on other cells.
Preserve existing formatting. After a write, report the new cell
range and the formula used. If a workbook is password-protected
or external-linked, stop and ask instead of guessing."""

_WORKER_SPECS = [
    ("Researcher", "Web research, fact gathering, synthesis.",
     RESEARCHER_PROMPT, ["computer_use", "memory"]),
    ("Coder",      "Write, run, refactor code; manage files and git.",
     CODER_PROMPT, ["computer_use", "git"]),
    ("MemoryAgent", "Persistent knowledge graph: store entities, "
     "relations, observations across sessions.",
     _MEMORY_PROMPT, ["memory"]),
    ("GitHubAgent", "GitHub-specific: PRs, issues, code review, releases.",
     GITHUB_AGENT_PROMPT, ["git"]),
    ("ExcelAgent",  "Read, write, and analyse Excel spreadsheets.",
     _EXCEL_PROMPT, ["excel"]),
]

def _build_workers(with_mcp=True):
    """Build the worker pool after bootstrap so workers see servers."""
    if not with_mcp:
        return [
            make_worker_agent(name, desc, prompt)
            for name, desc, prompt, _ in _WORKER_SPECS
        ]
    return [
        make_worker_agent(name, desc, prompt, list(mcp))
        for name, desc, prompt, mcp in _WORKER_SPECS
    ]


def _build_workers_no_mcp():
    """Fallback when bootstrap was interrupted (KeyboardInterrupt)."""
    return _build_workers(with_mcp=False)


class ChiefOfStaffSwarm(PlannerWorkerSwarm):
    """Gates on human approval, tracks edits, scores reliability, verifies files."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plan_approved = False
        self._edit_count = 0
        self._cycle_quality_scores = []

    def _run_planner(self, task, depth=0, parent_task_id=None):
        if task != self._original_task or self._plan_approved:
            return super()._run_planner(task, depth, parent_task_id)
        while True:
            sub_tasks = super()._run_planner(task, depth, parent_task_id)
            plan = self.task_queue.get_status()
            task_lines = "\n".join(
                f"  [white]{t['status']:<9}[/white] {t['title']}"
                for t in plan["tasks"]
            )
            _console.print(Panel(
                task_lines,
                title="[bold red]PLANNER PROPOSAL[/bold red]",
                border_style="red",
                padding=(0, 2),
            ))
            choice = input("Approve? (go / modify / quit): ").strip().lower()
            if choice in ("go", "g", ""):
                self._plan_approved = True
                return sub_tasks
            if choice in ("quit", "q", "exit"):
                logger.info("Human aborted the plan; ending swarm.")
                sys.exit(0)
            self._edit_count += 1
            task = (
                f"Original task: {self._original_task}\n\n"
                f"User edit: {input('What should change? ').strip()}\n\n"
                "Re-plan accordingly."
            )

    def _run_judge(self):
        verdict = super()._run_judge()
        self._cycle_quality_scores.append(verdict.overall_quality)
        return verdict

    def run(self, task=None, img=None, *args, **kwargs):
        up = down = 0
        for cat, _, port in MCP_SERVERS:
            ok = _probe(f"http://localhost:{port}/mcp")
            if ok:
                up += 1
            else:
                down += 1
        logger.info(f"MCP reachability: {up} up, {down} down")
        loops = adaptive_max_loops(task, len(self.agents))
        self.max_loops = loops
        logger.info(estimate_cost(task, len(self.agents), loops))

        raw = super().run(task=task, img=img, *args, **kwargs)

        self._print_task_results()
        completed = self.task_queue.get_completed_count()
        total = len(self.task_queue)
        avg_q = (
            sum(self._cycle_quality_scores) / len(self._cycle_quality_scores)
            if self._cycle_quality_scores else 0.0
        )
        reliability = compute_reliability(
            completed, total, avg_q, self._edit_count
        )
        _console.print(
            f"\n[bold red]Reliability:[/bold red] [white]{reliability:.2f}[/white]  "
            f"([white]completed[/white] [red]{completed}/{total}[/red] [white]tasks[/white], "
            f"[white]avg judge quality[/white] [red]{avg_q:.1f}/10[/red], "
            f"[white]{self._edit_count} human edit(s)[/white])\n"
        )
        return raw

    def _print_task_results(self):
        """Print each completed task's title and result. Verifies
        that any file paths mentioned in the result actually exist.
        """
        import re
        _console.print(
            "\n[bold red]" + "─" * 70 + "[/]\n"
            "[bold red]  TASK RESULTS[/]\n"
            "[bold red]" + "─" * 70 + "[/]"
        )
        for task in self.task_queue.get_all_tasks():
            status = task.status.value
            symbol = {
                "completed": "[bold white]✓[/]",
                "failed":    "[bold red]✗[/]",
            }.get(status, "[dim white]?[/]")
            _console.print(
                f"\n{symbol} [bold white]{task.title}[/] "
                f"[dim white]({status})[/]"
            )
            if status != "completed" or not task.result:
                continue
            for m in re.finditer(r"[/~][\w./\-]+\.\w{1,5}", task.result):
                p = os.path.expanduser(m.group(0))
                if not p.endswith(".") and not os.path.exists(p):
                    _console.print(
                        f"  [bold red]⚠[/] [dim white]claimed file "
                        f"`{p}` does not exist — worker likely described "
                        f"rather than called write_file.[/]"
                    )
            result = task.result
            if len(result) > 2000:
                result = result[:1500] + "\n\n[... truncated ...]\n\n" + result[-400:]
            _console.print(f"[dim white]{result}[/]")

def build_chief_of_staff(task=None):
    """Build the chief-of-staff swarm. Bootstrap MCP servers first
    so workers see them. Catches KeyboardInterrupt cleanly.
    """
    try:
        with Live(
            Spinner("dots", text="[red]Booting MCP servers[/red] "
                  "[white](npx -y auto-install + health check)...[/white]"),
            console=_console, transient=True,
        ):
            bootstrap_mcp_servers()
        workers = _build_workers()
    except KeyboardInterrupt:
        logger.info("Interrupted during bootstrap. Building workers without MCP.")
        workers = _build_workers_no_mcp()
    loops = adaptive_max_loops(task, len(workers)) if task else 3
    return ChiefOfStaffSwarm(
        name="ChiefOfStaff",
        description="Agentic chief-of-staff swarm with MCP integration.",
        agents=workers,
        max_loops=loops,
        planner_model_name=PLANNER_MODEL,
        judge_model_name=JUDGE_MODEL,
        max_planner_depth=1,
        task_timeout=600,
        worker_timeout=1800,
        max_workers=1,
        output_type="str-all-except-first",
        autosave=True,
        verbose=True,
    )

def _print_banner():
    from rich.table import Table
    cols = Table.grid(padding=(0, 2))
    cols.add_column(justify="left")
    cols.add_row(
        Text.from_markup(
            f"[bold white]CHIEF OF STAFF SWARM[/bold white]\n"
            f"[white]Orchestrator:[/white] [red]gpt-4o-mini[/]"
            f" [dim white](planner + judge)[/]\n"
            f"[white]Workers:[/white] [red]gpt-4.1[/]\n"
            f"  [dim white]Researcher · Coder · MemoryAgent · "
            f"GitHubAgent · ExcelAgent[/]\n"
            f"[white]MCP:[/white] filesystem · memory · git · "
            f"reasoning · excel · discovery\n"
            f"[white]cwd:[/white] [dim white]{os.getcwd()}[/]"
        ),
    )
    _console.print(Panel(
        cols,
        title="[bold white]v0.1 · CoS[/]",
        title_align="left",
        border_style="red",
        padding=(0, 2),
    ))
    _console.print(
        "[dim white]Multi-line input: end with a blank line. "
        "Up/Down arrows for history. Type "
        "[red]quit[/] to exit.[/dim white]"
    )

def _run_task(task):
    """Build the swarm, run it, return the result. Surfaces rate
    and network errors as one-line messages instead of stack traces.
    """
    try:
        return build_chief_of_staff(task).run(task)
    except Exception as e:
        err_str, err_type = str(e), type(e).__name__
        if "RateLimit" in err_type or "rate" in err_str.lower() or "TPM" in err_str:
            _console.print("\n[bold red]Rate limit hit.[/bold red] [white]Wait a minute, then run the same task again.[/white]")
            for line in err_str.splitlines():
                if "rate" in line.lower() or "TPM" in line:
                    _console.print(f"   [dim white]{line.strip()}[/]")
                    break
        elif "Connection" in err_type or "Timeout" in err_type:
            _console.print(f"\n[bold red]Network error[/bold red] [white]({err_type}): {err_str[:200]}[/white]")
        else:
            _console.print(f"\n[bold red]Error[/bold red] [white]({err_type}): {err_str[:300]}[/white]")
        return None

_HISTORY_PATH = os.path.expanduser("~/.cos_history")
try:
    readline.read_history_file(_HISTORY_PATH)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, _HISTORY_PATH)
readline.set_history_length(1000)


def _read_task():
    """Read a task with readline line editing and history.
    Multi-line input: end with a blank line or Ctrl-D.
    """
    lines = []
    try:
        while True:
            prompt = "CoS> " if not lines else "...> "
            line = input(prompt)
            if not line.strip() and lines:
                break
            lines.append(line)
    except EOFError:
        if not lines:
            return None
    return "\n".join(lines).strip()


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        _console.print("[bold red]OPENAI_API_KEY[/bold red] [white]not set. Add it to your .env and try again.[/white]")
        sys.exit(1)
    _print_banner()
    initial = sys.argv[1] if len(sys.argv) > 1 else None
    if initial:
        _run_task(initial)
    else:
        while True:
            try:
                task = _read_task()
                if task is None:
                    _console.print("\n[dim white]Goodbye.[/dim white]")
                    break
            except KeyboardInterrupt:
                _console.print("\n[dim white]Goodbye.[/dim white]")
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                _console.print("[dim white]Goodbye.[/dim white]")
                break
            try:
                _run_task(task)
            except KeyboardInterrupt:
                _console.print("\n[dim white]Task interrupted. Ready for the next task.[/dim white]")
            except Exception as e:
                _console.print(f"\n[bold red]Error:[/bold red] [white]{e}[/white]")
