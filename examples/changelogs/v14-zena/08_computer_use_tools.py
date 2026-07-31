"""Zena: sandboxed computer-use tools.

File reads/writes, edits, patches, directory listing, grep, and shell
execution — each behind an explicit policy layer rather than trusting the
model. Paths are canonicalized against deny lists, symlink escapes are
rejected, NUL bytes refused, binaries allow-listed.
"""

from swarms import Agent
from swarms.tools.computer_use import (
    create_computer_use_tools,
    grep_files,
    list_directory,
    read_file,
)

# --- Full toolset -----------------------------------------------------
agent = Agent(
    agent_name="Coding-Agent",
    model_name="gpt-5.4",
    tools=create_computer_use_tools(),
    max_loops="auto",
)
print(
    agent.run(
        "Find every TODO in the src directory and summarize what they block."
    )
)

# --- Narrower surface: read-only, no write/patch/delete/shell ---------
auditor = Agent(
    agent_name="Auditor",
    model_name="gpt-5.4",
    tools=[read_file, grep_files, list_directory],
    max_loops="auto",
)
print(auditor.run("Summarize the public API of this package."))
