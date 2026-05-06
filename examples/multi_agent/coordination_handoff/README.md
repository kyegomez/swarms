# Coordination + Handoff Protocol Example

Demonstrates standardized multi-agent orchestration patterns with Swarms.

## What This Shows

1. **Leader Election** — Deterministic coordinator selection from the swarm
2. **Work Distribution** — Round-robin task assignment across agents
3. **Signed Handoff** — Cryptographic task transfer with verification

## Run It

```bash
python examples/multi_agent/coordination_handoff/coordination_handoff_demo.py
```

## The Protocols

- **Coordination Protocol (L5):** Leader election, work distribution, conflict resolution
- **Handoff Protocol (L4):** Task transfer with context + verifiable acceptance

Both are CC BY 4.0 — free to adopt, adapt, or extend.

## Production SDK

```bash
pip install works-with-agents
```

Full specs: https://workswithagents.com/specs
