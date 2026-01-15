import os
import sys
import json

# ensure project root is on sys.path so examples/ and swarms/ packages are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from examples.auto_swarm_ui import load_eval_dataset, evaluate_agents, EVAL_DATASET, make_agents


def test_load_eval_dataset_tmp(tmp_path):
    p = tmp_path / "data.json"
    data = [
        {"id": 1, "input": "x", "expected": "y", "options": ["y","z"]}
    ]
    p.write_text(json.dumps(data))
    loaded = load_eval_dataset(str(p))
    assert isinstance(loaded, list)
    assert loaded[0]["expected"] == "y"


def test_evaluate_agents_simulated():
    agents = make_agents(3)
    scores = evaluate_agents(agents, EVAL_DATASET)
    assert isinstance(scores, dict)
    assert all("accuracy" in v and "avg_similarity" in v for v in scores.values())
    # accuracies between 0 and 1
    for v in scores.values():
        assert 0.0 <= v["accuracy"] <= 1.0


def test_evaluate_agents_deterministic():
    a1 = make_agents(1)[0]
    a2 = make_agents(1)[0]
    # same name -> same deterministic simulated score
    scores1 = evaluate_agents([a1], EVAL_DATASET)
    scores2 = evaluate_agents([a2], EVAL_DATASET)
    assert scores1[a1]["accuracy"] == scores2[a2]["accuracy"]
