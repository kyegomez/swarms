"""
graph_workflow_checkpointing.py

Demonstrates the checkpoint fault-tolerance feature added to GraphWorkflow:

  GraphWorkflow(checkpoint_dir="./checkpoints/run_xyz")

Checkpoint behaviour
--------------------
* After each layer completes, outputs are written to
  {checkpoint_dir}/{task_hash}_layer_{idx}.json.
* On the next run() call with the same task string, completed layers are
  loaded from disk and skipped — saving LLM cost and time.
* workflow.clear_checkpoints(task) removes all checkpoint files for that task
  once you are satisfied the run succeeded.

Simulated crash scenario
-------------------------
1. Build a three-agent pipeline (Researcher -> Analyst -> Writer).
2. Run it once — all three agents execute and checkpoints are written.
3. Simulate a "crash" by deleting the Writer checkpoint only, then call run()
   again with the same task.
4. The Researcher and Analyst layers are restored from disk; only the Writer
   re-executes.
5. Clean up checkpoints after a confirmed successful run.
"""

import hashlib
import shutil
from pathlib import Path

from swarms.structs.agent import Agent
from swarms.structs.graph_workflow import GraphWorkflow

# ---------------------------------------------------------------------------
# 1. Build agents
# ---------------------------------------------------------------------------
researcher = Agent(
    agent_name="Researcher",
    model_name="claude-sonnet-4-5",
    agent_description="Gathers and summarises relevant information on the topic.",
    max_loops=1,
    temperature=1,
)

analyst = Agent(
    agent_name="Analyst",
    model_name="claude-sonnet-4-5",
    agent_description="Analyses the researcher's findings and draws key insights.",
    max_loops=1,
    temperature=1,
)

writer = Agent(
    agent_name="Writer",
    model_name="claude-sonnet-4-5",
    agent_description="Turns the analyst's insights into a clear, concise report.",
    max_loops=1,
    temperature=1,
)

# ---------------------------------------------------------------------------
# 2. Create workflow with checkpoint_dir
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = "./checkpoints/demo_run"

workflow = GraphWorkflow(
    name="Research Pipeline",
    description="Three-stage research, analysis, and writing pipeline.",
    max_loops=1,
    checkpoint_dir=CHECKPOINT_DIR,
    verbose=True,
)

workflow.add_nodes([researcher, analyst, writer])
workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.compile()

# ---------------------------------------------------------------------------
# 3. First run — all three agents execute, checkpoints written after each layer
# ---------------------------------------------------------------------------
TASK = "Summarise the main benefits of multi-agent AI systems."

print("\n" + "=" * 60)
print("FIRST RUN — all layers execute")
print("=" * 60)
results_first = workflow.run(TASK)
print("\nResults after first run:")
for agent_name, output in results_first.items():
    print(f"  [{agent_name}] {str(output)[:120]}")

# Confirm checkpoint files were created
cp_dir = Path(CHECKPOINT_DIR)
print(f"\nCheckpoint files written to {CHECKPOINT_DIR}:")
for f in sorted(cp_dir.glob("*.json")):
    print(f"  {f.name}")

# ---------------------------------------------------------------------------
# 4. Simulate a mid-run crash: delete the Writer checkpoint so only that
#    layer needs to re-execute on the next run.
# ---------------------------------------------------------------------------
task_key = hashlib.sha256(TASK.encode("utf-8")).hexdigest()[:16]
writer_layer_idx = 2  # layer 0=Researcher, 1=Analyst, 2=Writer
writer_cp = cp_dir / f"{task_key}_layer_{writer_layer_idx}.json"

if writer_cp.exists():
    writer_cp.unlink()
    print(f"\nSimulated crash: deleted {writer_cp.name}")

# ---------------------------------------------------------------------------
# 5. Second run — Researcher and Analyst are restored from checkpoints,
#    only Writer re-executes.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(
    "SECOND RUN — Researcher & Analyst restored, Writer re-executes"
)
print("=" * 60)
results_second = workflow.run(TASK)

print(
    "\nResults after second run (Writer output is freshly generated):"
)
for agent_name, output in results_second.items():
    print(f"  [{agent_name}] {str(output)[:120]}")

# ---------------------------------------------------------------------------
# 6. Clean up checkpoints after a confirmed successful run
# ---------------------------------------------------------------------------
deleted = workflow.clear_checkpoints(TASK)
print(f"\nCleared {deleted} checkpoint file(s).")

shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
print(f"Removed checkpoint directory: {CHECKPOINT_DIR}")
