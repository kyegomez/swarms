# Minimal HSCF example 2
# Top-level variables only (no prints, no functions)

observation_1 = {"agent": "worker_1", "data": {"location": "locA", "value": 42}, "confidence": 0.9}
observation_2 = {"agent": "worker_2", "data": {"location": "locB", "value": 37}, "confidence": 0.8}

report = {
    "id": "report_001",
    "origin": "leader",
    "contents": {
        "observations": [observation_1, observation_2],
        "summary": {"avg_value": (observation_1["data"]["value"] + observation_2["data"]["value"]) / 2},
        "actions": [{"action": "merge", "targets": ["worker_1", "worker_2"]}],
    },
    "status": "ready",
}

hscf_payload = {
    "context": {"mission": "survey", "stage": "analysis"},
    "report": report,
    "routing": {"next": ["leader"], "history": ["worker_1", "worker_2"]},
}
