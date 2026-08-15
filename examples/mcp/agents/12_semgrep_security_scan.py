"""
Example 12 — Semgrep MCP (free account, token required) for security review.

Semgrep runs static analysis over code you hand it and returns concrete
findings: rule id, severity, line number, and why the pattern is dangerous.
Pairing it with an LLM is a good division of labour — the scanner finds
occurrences deterministically, the model explains impact and drafts the fix.

    Server : https://mcp.semgrep.ai/mcp
    Auth   : Semgrep AppSec Platform token, Bearer  (free account)
    Tools  : semgrep_scan, security_check, get_abstract_syntax_tree, ...

Note: this endpoint accepted anonymous traffic historically and now returns
401 without a token. Verified 2026-08-12. Get a free token at
https://semgrep.dev/login -> Settings -> Tokens.

The point of the system prompt below is to stop the model doing what LLMs
love to do with security questions: inventing plausible-sounding
vulnerabilities. Findings come from the scanner; the model's job is triage,
not imagination.

Run:
    export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, etc.
    export SEMGREP_APP_TOKEN=...     # free at https://semgrep.dev
    python examples/mcp/agents/12_semgrep_security_scan.py
"""

import os
import sys

from swarms import Agent

MODEL = "gpt-5.4"

SECURITY_SYSTEM_PROMPT = (
    "You are a security reviewer. Report only vulnerabilities the scanner "
    "actually returned — never speculate about issues you did not find, and "
    "never pad a report to look thorough. For each finding give the rule id, "
    "severity, the exact line, why it is exploitable in this specific code, "
    "and a concrete patch. Rank by real-world exploitability rather than by "
    "the scanner's own severity label, and say plainly when a flagged line is "
    "a false positive in context. If the scan comes back clean, report that "
    "it is clean and note what the scan does not cover."
)

VULNERABLE_SAMPLE = """
import os
import sqlite3
import subprocess


def get_user(conn, user_id):
    # String-interpolated SQL
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = '%s'" % user_id)
    return cur.fetchone()


def run_report(report_name):
    # Shell invocation built from user input
    subprocess.call("generate_report " + report_name, shell=True)


def load_config(blob):
    # Deserializing untrusted input
    import pickle
    return pickle.loads(blob)


# Deliberately not shaped like any real provider's key: a realistic
# prefix here would trip secret scanners in every fork of this repo.
API_TOKEN = "hardcoded-credential-placeholder-do-not-use"
"""

agent = Agent(
    agent_name="Semgrep-Security-Agent",
    agent_description="Reviews code for vulnerabilities using Semgrep MCP.",
    system_prompt=SECURITY_SYSTEM_PROMPT,
    model_name=MODEL,
    mcp_url="https://mcp.semgrep.ai/mcp",
    mcp_api_key="env:SEMGREP_APP_TOKEN",
    max_loops=2,
)

if __name__ == "__main__":
    if not os.getenv("SEMGREP_APP_TOKEN"):
        sys.exit(
            "SEMGREP_APP_TOKEN is not set.\n"
            "Create a free token at https://semgrep.dev "
            "(Settings -> Tokens) and export it before running."
        )

    result = agent.run(
        "Scan this Python file with Semgrep and write up every finding: rule "
        "id, severity, line, why it is exploitable, and the fix.\n\n"
        f"```python\n{VULNERABLE_SAMPLE}\n```"
    )
    print(result)
