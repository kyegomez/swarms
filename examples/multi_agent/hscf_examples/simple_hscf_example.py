# Minimal Hierarchical Structured Communication Framework (HSCF) example
# Very simple to run: no prints, no functions — just top-level data structures.

agents = [
    {"id": "leader", "role": "coordinator", "abilities": ["plan", "summarize"]},
    {"id": "worker1", "role": "executor", "abilities": ["fetch", "analyze"]},
    {"id": "worker2", "role": "executor", "abilities": ["fetch", "synthesize"]},
]

messages = [
    {"from": "leader", "to": "worker1", "content": "task:fetch_data", "priority": 1},
    {"from": "worker1", "to": "leader", "content": "result:partial_data", "priority": 2},
]

hscf_example = {
    "agents": agents,
    "messages": messages,
    "meta": {"framework": "hscf", "version": "0.1", "author": "ilum"},
}
