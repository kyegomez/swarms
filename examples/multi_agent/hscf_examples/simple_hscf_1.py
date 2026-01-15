# Minimal HSCF example 1
# Top-level variables only (no prints, no functions)

agents = {
    "leader": {"id": "agent_leader", "role": "coordinator", "capabilities": ["plan", "aggregate"]},
    "worker_1": {"id": "agent_worker_1", "role": "worker", "capabilities": ["search"]},
    "worker_2": {"id": "agent_worker_2", "role": "worker", "capabilities": ["fetch"]},
}

message_1 = {
    "id": "msg_001",
    "from": "leader",
    "to": ["worker_1", "worker_2"],
    "type": "request",
    "payload": {"task": "collect_data", "parameters": {"query": "weather"}},
    "meta": {"priority": "high", "timestamp": "2026-01-15T12:00:00Z"},
}

hierarchy = {
    "swarm_id": "swarm_xyz",
    "schema_version": "hscf-v1",
    "agents": agents,
    "messages": [message_1],
    "policies": {"aggregation": "majority", "timeout_s": 30},
}
