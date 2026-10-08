# Auction Swarm Examples

This directory contains examples demonstrating the **`AuctionSwarm`** architecture — a market-based way to route a task to the right agent(s).

## Examples

- [basic_auction.py](basic_auction.py) — A pool of specialists bids on one task; the single best bidder (`top_k=1`) executes it.
- [custom_scoring_auction.py](custom_scoring_auction.py) — Multiple winners (`top_k=2`) plus a custom `scoring` function that rewards quality over cost.
- [batch_auction.py](batch_auction.py) — One specialist pool auctioned across several different tasks with `batch_run`; a different winner emerges per task.

## Overview

In most orchestrators a "boss" agent decides which worker should handle a task. That relies on the boss's model of each worker being accurate and up to date. `AuctionSwarm` inverts this: **every agent self-assesses its fitness for the specific task, and the best bid wins.**

### How it works

```
task
  │
  ├─▶ broadcast to every agent in the pool (concurrently)
  │
  ├─▶ each agent is forced to call bid(confidence, estimated_cost)
  │       • confidence    → 0..1, how likely it can do the task well
  │       • estimated_cost → relative cost (1.0 = average, lower = cheaper)
  │
  ├─▶ the auctioneer scores every bid  (default: confidence / estimated_cost)
  │       and sorts bidders best-first
  │
  ├─▶ the top_k highest-scoring agents execute the real task (concurrently)
  │
  └─▶ the response from the best-scoring winner that didn't error is kept
```

Each bid is collected through a **forced function call** (`BID_TOOL`), so every agent returns a structured, machine-readable `(confidence, estimated_cost)` pair instead of free-form text. The swarm handles the tool lifecycle automatically:

1. Before bidding, the `bid` tool is injected into any agent that doesn't already have it (`auto_equip=True`).
2. After bids are collected, the `bid` tool is removed so the winning agent runs the *real* task instead of bidding again.
3. Each bidder's short-term memory is reset between the bidding phase and execution (and between tasks in `batch_run`), so the bid prompt doesn't leak into the final answer.

### Key parameters

| Parameter | Default | Purpose |
|---|---|---|
| `agents` | — | The bidding pool (at least one agent required). |
| `top_k` | `1` | How many of the highest-scoring bidders execute the task. When `> 1`, all winners run and the best non-erroring result is kept. |
| `scoring` | `"confidence_per_cost"` | Either a built-in name or a callable `(confidence, estimated_cost) -> float` that ranks bids (highest wins). |
| `output_type` | `"dict"` | Format passed to `history_output_formatter`. |
| `auto_equip` | `True` | Automatically inject/remove the `bid` tool around each auction. |
| `print_on` | `True` | Print the bid table and the award panel. |

### Scoring

The default scoring function is `confidence_per_cost` (`confidence / estimated_cost`), which favors agents that are confident **and** cheap. To change the trade-off, pass your own callable — see [custom_scoring_auction.py](custom_scoring_auction.py) for a quality-first example that mostly ignores cost.

## Running

`AuctionSwarm` is imported from its module path:

```python
from swarms.structs.agent import Agent
from swarms.structs.auction_swarm import AuctionSwarm

agents = [
    Agent(agent_name="Translator", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Generalist", model_name="gpt-5.4", max_loops=1),
]

swarm = AuctionSwarm(agents=agents, top_k=1)
result = swarm.run("Translate this contract clause into plain English.")
print(result)
```

Set your model provider's API key first, then run any example:

```bash
export OPENAI_API_KEY="sk-..."
python basic_auction.py
```

## When to use it

- You have a **diverse pool of specialists** and want each task routed to whoever is genuinely best for it.
- You want routing decisions grounded in **each agent's own self-assessment**, not a central boss's (possibly stale) view.
- You want a **cost/quality trade-off** you can tune with a scoring function.
