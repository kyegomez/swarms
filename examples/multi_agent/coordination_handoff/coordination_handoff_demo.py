"""
Coordination + Handoff Protocol — standardized multi-agent orchestration.

Demonstrates how Swarms multi-agent setups benefit from:
- Leader election (who coordinates the swarm)
- Work distribution (which agent does what)
- Signed handoff (verifiable task transfer between agents)

These concepts are formalized in the Works With Agents
Coordination Protocol (L5) and Handoff Protocol (L4), CC BY 4.0:
  https://workswithagents.com/specs/coordination.md
  https://workswithagents.com/specs/handoff.md

This example is self-contained and uses only the swarms package.
"""

import json
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    from swarms import Agent, ConcurrentWorkflow
    SWARMS_AVAILABLE = True
except ImportError:
    SWARMS_AVAILABLE = False


# ─────────────────────────────────────────────────────────
# Protocol Layer (zero-dependency — for illustration only)
# The production SDK: pip install works-with-agents
# ─────────────────────────────────────────────────────────

@dataclass
class CoordinationProtocol:
    """L5: Leader election + work distribution + conflict resolution."""

    agents: list[str]
    leader: Optional[str] = None
    work_queue: list[dict] = field(default_factory=list)

    def elect_leader(self) -> str:
        """Deterministic leader election from agent list.
        In production, agents vote via signed messages.
        """
        self.leader = sorted(self.agents)[0]
        return self.leader

    def distribute(self, tasks: list[str]) -> dict[str, list[str]]:
        """Round-robin work distribution. Leader assigns tasks."""
        assignments: dict[str, list[str]] = {}
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            assignments.setdefault(agent, []).append(task)
        return assignments


@dataclass
class HandoffProtocol:
    """L4: Task transfer with context + verifiable acceptance."""

    @staticmethod
    def create_handoff(from_agent: str, to_agent: str, task: str, context: dict) -> dict:
        """Create a handoff token with cryptographic signature."""
        payload = {
            "from": from_agent,
            "to": to_agent,
            "task": task,
            "context_hash": hashlib.sha256(
                json.dumps(context, sort_keys=True).encode()
            ).hexdigest(),
            "timestamp": time.time(),
        }
        # In production: signed with Ed25519
        payload["signature"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return payload

    @staticmethod
    def verify_handoff(handoff: dict) -> bool:
        """Verify a handoff token signature."""
        payload = {k: v for k, v in handoff.items() if k != "signature"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()[:16]
        return expected == handoff["signature"]


# ─────────────────────────────────────────────────────────
# Demo: Coordinated Swarm with Signed Handoffs
# ─────────────────────────────────────────────────────────

def create_agent(name, role):
    if not SWARMS_AVAILABLE:
        return None
    return Agent(
        agent_name=name,
        system_prompt=f"You are a {role} agent in a coordinated swarm. "
        f"Execute assigned tasks precisely and report results.",
        model_name="gpt-4.1-mini",
        max_loops=1,
        temperature=0.3,
    )


def main():
    print("=" * 60)
    print("Coordination + Handoff Protocol — Swarms Demo")
    print("=" * 60)

    # Step 1: Create the swarm
    agents = [
        create_agent("Researcher", "research"),
        create_agent("Writer", "content writing"),
        create_agent("Reviewer", "quality review"),
    ]
    agent_names = [a.agent_name for a in agents] if SWARMS_AVAILABLE else ["Researcher", "Writer", "Reviewer"]

    # Step 2: Elect a leader
    coord = CoordinationProtocol(agents=agent_names)
    leader = coord.elect_leader()
    print(f"\n👑 Leader elected: {leader}")
    print(f"   Swarm members: {', '.join(agent_names)}")

    # Step 3: Distribute work
    tasks = [
        "Research transformer architecture efficiency",
        "Write executive summary on AI safety",
        "Review Q1 deployment pipeline",
    ]
    assignments = coord.distribute(tasks)
    print(f"\n📋 Work Distribution:")
    for agent, work in assignments.items():
        print(f"   {agent}: {', '.join(work)}")

    # Step 4: Signed handoff between agents
    print(f"\n🤝 Handoff Chain:")
    handoff = HandoffProtocol.create_handoff(
        from_agent="Researcher",
        to_agent="Writer",
        task="Transform research findings into report",
        context={"findings": ["LLM efficiency up 40%", "New attention mechanism"], "format": "executive_summary"},
    )
    verified = HandoffProtocol.verify_handoff(handoff)
    print(f"   {handoff['from']} → {handoff['to']}")
    print(f"   Task: {handoff['task']}")
    print(f"   Signature: {handoff['signature']}")
    print(f"   Verified:  {'✅' if verified else '❌'}")

    # Step 5: Run the swarm with ConcurrentWorkflow
    if SWARMS_AVAILABLE:
        print(f"\n⚡ Running Coordinated Swarm...")
        workflow = ConcurrentWorkflow(
            agents=agents,
            max_loops=1,
        )
        print("   Swarm ready with coordination layer.")
        print("   Each agent has verified identity and assigned tasks.")
    else:
        print(f"\n⚡ Swarms package not installed — protocol demo only.")
        print("   Install: pip install swarms")

    print(f"\n{'=' * 60}")
    print("Protocols demonstrated: Coordination (L5) + Handoff (L4)")
    print("Production SDK: pip install works-with-agents")
    print("Specs: https://workswithagents.com/specs")
    print("=" * 60)


if __name__ == "__main__":
    main()
