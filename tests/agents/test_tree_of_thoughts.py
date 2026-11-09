"""Search behaviour of ``TreeOfThoughts`` against a scripted model."""

import json
import re

import pytest

from swarms.agents.tree_of_thoughts import (
    _TOOL_SCHEMAS,
    EvaluateThought,
    ProposeThoughts,
    TreeOfThoughts,
    parse_tool_call,
)
from swarms.structs.agent import Agent

# root -> A, B, C; A dead-ends; B -> B1 -> the right answer.
TREE = {
    "": [("A", False), ("B", False), ("C", False)],
    "A": [("A1", False), ("A2", False)],
    "B": [("B1", False), ("B2", False)],
    "B1": [("ANSWER 42", True)],
    "B2": [("ANSWER 7", True)],
}
SCORES = {
    "A": 0.9,
    "B": 0.6,
    "C": 0.2,
    "A1": 0.3,
    "A2": 0.4,
    "B1": 0.8,
    "B2": 0.7,
    "ANSWER 42": 0.95,
    "ANSWER 7": 0.1,
}


def _tool_call(name, arguments):
    return [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }
    ]


def _steps(prompt, header):
    """The numbered steps listed under ``header`` in a prompt."""
    section = prompt.split(header, 1)[1].split("\n\n", 1)[0]
    return re.findall(r"^\d+\. (.*)$", section, flags=re.M)


class ScriptedModel:
    """Answers each internal agent from TREE and SCORES, recording prompts."""

    def __init__(self, tree=None, scores=None, votes=None):
        self.tree = TREE if tree is None else tree
        self.scores = SCORES if scores is None else scores
        self.votes = list(votes or [])
        self.prompts = {
            "generator": [],
            "evaluator": [],
            "answerer": [],
        }

    def __call__(self, agent, prompt):
        role = agent.agent_name.rsplit("-", 1)[-1]
        self.prompts[role].append(prompt)
        agent._add_usage(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        if role == "generator":
            steps = _steps(prompt, "STEPS SO FAR:\n")
            last = steps[-1] if steps else ""
            thoughts = [
                {"content": content, "is_final": final}
                for content, final in self.tree.get(last, [])
            ]
            return _tool_call(
                "ProposeThoughts", {"thoughts": thoughts}
            )
        if role == "evaluator" and "CANDIDATE 1" in prompt:
            return _tool_call(
                "VoteForThought",
                {
                    "reasoning": "best",
                    "best_candidate": self.votes.pop(0),
                },
            )
        if role == "evaluator":
            last = _steps(prompt, "PARTIAL SOLUTION:\n")[-1]
            return _tool_call(
                "EvaluateThought",
                {
                    "reasoning": f"rated {last}",
                    "score": self.scores[last],
                },
            )
        steps = _steps(prompt, "REASONING:\n")
        return _tool_call(
            "FinalAnswer",
            {"answer": f"final: {steps[-1] if steps else 'none'}"},
        )


def _prompt(task, kwargs):
    """The prompt an agent sent: its task, or the last message when history is typed."""
    return task or kwargs["messages"][-1]["content"]


def _install(monkeypatch, scripted):
    monkeypatch.setattr(
        Agent,
        "call_llm",
        lambda self, task=None, *a, **k: scripted(
            self, _prompt(task, k)
        ),
    )


@pytest.fixture
def model(monkeypatch):
    scripted = ScriptedModel()
    _install(monkeypatch, scripted)
    return scripted


def _agent(**overrides):
    settings = {"max_depth": 3, "max_workers": 4, **overrides}
    return TreeOfThoughts(**settings)


def _solve(agent, task="task"):
    """Run the agent and return the search it kept on ``last_result``."""
    agent.run(task)
    return agent.last_result


def test_parse_tool_call_reads_every_output_shape():
    arguments = {"reasoning": "fine", "score": 0.7}
    call = _tool_call("EvaluateThought", arguments)

    shapes = [
        call,
        call[0],
        str(call),
        json.dumps(arguments),
        f"Here is my rating: {json.dumps(arguments)} thanks",
        [
            {
                "function": {
                    "name": "EvaluateThought",
                    "arguments": arguments,
                }
            }
        ],
    ]
    for shape in shapes:
        parsed = parse_tool_call(shape, EvaluateThought)
        assert parsed is not None, f"unparsed shape: {shape!r}"
        assert parsed.score == 0.7


def test_parse_tool_call_rejects_wrong_or_invalid_calls():
    assert (
        parse_tool_call(
            _tool_call("Other", {"score": 0.5}), EvaluateThought
        )
        is None
    )
    assert (
        parse_tool_call(
            _tool_call(
                "EvaluateThought", {"reasoning": "x", "score": 7}
            ),
            EvaluateThought,
        )
        is None
    ), "a score outside 0..1 must not be accepted"
    assert parse_tool_call("no json here", EvaluateThought) is None
    assert parse_tool_call(None, EvaluateThought) is None


def test_bare_string_thoughts_are_accepted():
    parsed = parse_tool_call(
        _tool_call(
            "ProposeThoughts", {"thoughts": ["step one", "step two"]}
        ),
        ProposeThoughts,
    )
    assert [t.content for t in parsed.thoughts] == [
        "step one",
        "step two",
    ]
    assert not any(t.is_final for t in parsed.thoughts)


def test_tool_schemas_carry_no_refs():
    """Models ignored $ref'd fields and invented their own thought keys."""
    for schema in _TOOL_SCHEMAS.values():
        assert "$ref" not in json.dumps(schema)
        assert "$defs" not in json.dumps(schema)
    thought = _TOOL_SCHEMAS[ProposeThoughts]["function"][
        "parameters"
    ]["properties"]["thoughts"]["items"]
    assert set(thought["properties"]) == {"content", "is_final"}
    assert thought["required"] == ["content"]


def test_bfs_keeps_the_beam_and_finds_the_solution(model):
    result = _solve(_agent(search_algorithm="bfs", breadth=2))

    assert result.solved
    assert result.steps == ["B", "B1", "ANSWER 42"]
    assert result.answer == "final: ANSWER 42"
    # root, then A and B, then B1 and B2: A's children were pruned.
    assert result.nodes_expanded == 5
    a = next(c for c in result.root.children if c.content == "A")
    assert all(child.pruned for child in a.children)
    assert all(not child.children for child in a.children)


def test_bfs_never_expands_a_pruned_node(model):
    result = _solve(_agent(search_algorithm="bfs", breadth=3))

    c = next(n for n in result.root.children if n.content == "C")
    assert c.pruned and c.score == 0.2
    assert (
        not c.children
    ), "C scored below the threshold but was expanded"


def test_dfs_backtracks_out_of_a_dead_end(model):
    result = _solve(_agent(search_algorithm="dfs"))

    assert result.solved
    assert result.steps == ["B", "B1", "ANSWER 42"]
    # root, A (dead end), B, B1; B2 is never needed.
    assert result.nodes_expanded == 4
    b2 = next(
        n
        for b in result.root.children
        for n in b.children
        if n.content == "B2"
    )
    assert not b2.children


def test_bfs_stops_once_a_solution_beats_every_open_path(monkeypatch):
    scripted = ScriptedModel(
        tree={"": [("DONE", True), ("MAYBE", False)]},
        scores={"DONE": 0.9, "MAYBE": 0.6},
    )
    _install(monkeypatch, scripted)

    result = _solve(_agent(search_algorithm="bfs"))

    assert result.steps == ["DONE"]
    assert result.nodes_expanded == 1


def test_bfs_keeps_searching_past_a_weaker_solution(monkeypatch):
    scripted = ScriptedModel(
        tree={
            "": [("WEAK", True), ("PROMISING", False)],
            "PROMISING": [("STRONG", True)],
        },
        scores={"WEAK": 0.6, "PROMISING": 0.9, "STRONG": 0.95},
    )
    _install(monkeypatch, scripted)

    result = _solve(_agent(search_algorithm="bfs"))

    assert result.steps == ["PROMISING", "STRONG"]


def test_the_last_step_is_final_even_if_unflagged(monkeypatch):
    scripted = ScriptedModel(
        tree={"": [("X", False)], "X": [("Y", False)]},
        scores={"X": 0.9, "Y": 0.9},
    )
    _install(monkeypatch, scripted)

    result = _solve(_agent(max_depth=2))

    assert result.solved and result.steps == ["X", "Y"]
    assert "last step allowed" in scripted.prompts["generator"][-1]
    assert "last step allowed" not in scripted.prompts["generator"][0]


def test_max_expansions_caps_generation(model):
    result = _solve(_agent(search_algorithm="dfs", max_expansions=2))

    assert result.nodes_expanded == 2
    assert len(model.prompts["generator"]) == 2
    assert not result.solved
    # Unsolved: answered from the best partial path, not a solution.
    assert result.best_node.content == "A"
    assert "did not finish" in model.prompts["answerer"][0]


def test_vote_scores_are_relative_to_the_leader(monkeypatch):
    scripted = ScriptedModel(
        tree={"": [("P", False), ("Q", False), ("R", False)]},
        votes=[1, 1, 2],
    )
    _install(monkeypatch, scripted)

    agent = _agent(
        evaluation_strategy="vote",
        n_evaluate_samples=3,
        max_depth=1,
        max_expansions=1,
    )
    result = _solve(agent)

    scores = {n.content: n.score for n in result.root.children}
    assert scores == {"P": 1.0, "Q": 0.5, "R": 0.0}
    pruned = {n.content for n in result.root.children if n.pruned}
    assert pruned == {"R"}


def test_sample_strategy_calls_once_per_thought_and_dedupes(
    monkeypatch,
):
    scripted = ScriptedModel(
        tree={"": [("same idea", False)]},
        scores={"same idea": 0.9},
    )
    _install(monkeypatch, scripted)

    agent = _agent(
        generation_strategy="sample",
        num_thoughts=3,
        max_expansions=1,
    )
    result = _solve(agent)

    assert len(scripted.prompts["generator"]) == 3
    assert (
        "one candidate next step" in scripted.prompts["generator"][0]
    )
    assert [n.content for n in result.root.children] == ["same idea"]


def test_a_failing_call_costs_one_candidate_not_the_search(
    monkeypatch,
):
    scripted = ScriptedModel()

    def flaky(self, task=None, *args, **kwargs):
        prompt = _prompt(task, kwargs)
        if "PARTIAL SOLUTION:" in prompt and _steps(
            prompt, "PARTIAL SOLUTION:\n"
        ) == ["A"]:
            raise ConnectionError("provider down")
        return scripted(self, prompt)

    monkeypatch.setattr(Agent, "call_llm", flaky)

    result = _solve(_agent(search_algorithm="dfs"))

    a = next(c for c in result.root.children if c.content == "A")
    assert a.score == 0.0 and a.pruned
    assert result.solved and result.steps == ["B", "B1", "ANSWER 42"]


def test_no_answer_at_all_raises(monkeypatch):
    monkeypatch.setattr(
        Agent,
        "call_llm",
        lambda self, task=None, *a, **k: "I refuse to call functions.",
    )

    with pytest.raises(RuntimeError, match="produced no answer"):
        _agent().run("task")


def test_run_returns_the_answer_and_records_the_path(model):
    agent = _agent(search_algorithm="dfs")

    assert agent.run("task") == "final: ANSWER 42"
    contents = [
        m["content"] for m in agent.conversation.conversation_history
    ]
    assert "task" in contents
    assert any("1. B\n2. B1\n3. ANSWER 42" in c for c in contents)


def test_name_labels_the_conversation_and_internal_agents(
    model, monkeypatch
):
    built = []
    original = Agent.__init__

    def spy(self, *args, **kwargs):
        built.append(kwargs["agent_name"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", spy)

    agent = _agent(name="Solver", search_algorithm="dfs")
    agent.run("task")

    roles = {
        m["role"] for m in agent.conversation.conversation_history
    }
    assert "Solver" in roles
    assert set(built) == {
        "Solver-generator",
        "Solver-evaluator",
        "Solver-answerer",
    }


def test_usage_and_call_counts_cover_every_call(model):
    agent = _agent(search_algorithm="dfs")
    result = _solve(agent)

    calls = sum(len(prompts) for prompts in model.prompts.values())
    assert result.llm_calls == calls
    assert result.usage["total_tokens"] == 15 * calls
    agent.run("task")
    assert agent.usage["total_tokens"] == 2 * 15 * calls


def test_agent_kwargs_cannot_replace_the_tool(model, monkeypatch):
    built = []
    original = Agent.__init__

    def spy(self, *args, **kwargs):
        built.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", spy)

    _agent(
        agent_kwargs={"tools_list_dictionary": [], "max_tokens": 321}
    ).run("task")

    assert built and all(k["max_tokens"] == 321 for k in built)
    assert all(len(k["tools_list_dictionary"]) == 1 for k in built)


def test_no_temperature_is_sent_unless_set(model, monkeypatch):
    """Claude Sonnet 5 rejects any temperature, so the default sends none."""
    built = []
    original = Agent.__init__

    def spy(self, *args, **kwargs):
        built.append(kwargs["temperature"])
        original(self, *args, **kwargs)

    monkeypatch.setattr(Agent, "__init__", spy)

    _agent(max_expansions=1).run("task")
    _agent(max_expansions=1, temperature=0.3).run("task")

    assert set(built) == {None, 0.3}


@pytest.mark.parametrize(
    "overrides",
    [
        {"search_algorithm": "mcts"},
        {"generation_strategy": "guess"},
        {"evaluation_strategy": "rank"},
        {"num_thoughts": 0},
        {"breadth": 0},
        {"max_depth": 0},
        {"n_evaluate_samples": 0},
        {"max_workers": 0},
        {"max_expansions": 0},
        {"value_threshold": 1.5},
    ],
)
def test_invalid_settings_are_rejected(overrides):
    with pytest.raises(ValueError):
        TreeOfThoughts(**overrides)


def test_empty_task_is_rejected():
    with pytest.raises(ValueError):
        TreeOfThoughts().run("  ")
