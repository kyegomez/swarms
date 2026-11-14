import pytest

from swarms.structs.context_tree import (
    SCOPES,
    WEIGHTS,
    ContextTree,
    _split_budget,
)
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)

READERS = [("a", 1), ("a", 2), ("b", 1), ("b", 2)]


def one_per_scope(tree):
    for scope in SCOPES:
        tree.write(f"{scope} finding", scope, tree_id="a", depth=1)


@pytest.mark.parametrize(
    "scope, sees",
    [
        ("level", [True, False, False, False]),
        ("tree", [True, True, False, False]),
        ("global", [True, True, True, True]),
        ("cohort", [True, False, True, False]),
    ],
)
def test_a_note_reaches_only_its_scope(scope, sees):
    tree = ContextTree()
    tree.write("api returns 200", scope, tree_id="a", depth=1)
    for (tree_id, depth), visible in zip(READERS, sees):
        expected = {s: [] for s in SCOPES}
        expected[scope] = ["api returns 200"] if visible else []
        assert tree.read(tree_id, depth) == expected


@pytest.mark.parametrize(
    "text, scope, error",
    [
        ("cache hit rate is 40%", "team", "Unknown scope"),
        ("   ", "tree", "empty"),
        ("x" * 401, "tree", "400"),
        ("Because I checked, auth is fine", "tree", "reasoning"),
        ("let me think about retries", "tree", "reasoning"),
        ("First I read config.py", "tree", "reasoning"),
        ("token is sk-123", "tree", "Coder sent a key"),
        ("first index is 0 because input is sorted", "tree", None),
        ("x" * 400, "global", None),
    ],
)
def test_write_accepts_only_short_observations(text, scope, error):
    def no_keys(author, text):
        return "sk-" not in text, f"{author} sent a key"

    tree = ContextTree(gate=no_keys)
    if error is None:
        tree.write(text, scope, author="Coder")
        assert [n["text"] for n in tree.notes] == [text]
        return
    with pytest.raises(ValueError, match=error):
        tree.write(text, scope, author="Coder")
    assert tree.notes == []


# Notes are (scope, characters), and four characters cost one token.
@pytest.mark.parametrize(
    "notes, budget, kept",
    [
        ([("tree", 40)] * 5, 35, [4, 3, 2]),
        (
            [("level", 40)] * 10 + [("cohort", 40)] * 10,
            40,
            [9, 8, 7, 19],
        ),
        (
            [("global", 40)] + [("tree", 40)] * 10,
            60,
            [10, 9, 8, 7, 6, 0],
        ),
        ([("tree", 40)] * 2, 9, []),
        ([("tree", 12), ("tree", 32)], 5, []),
        ([("tree", 41)] * 3, 30, [2, 1]),
    ],
    ids=["slack", "weights", "spare", "whole", "no-gaps", "ceil"],
)
def test_read_keeps_the_newest_whole_notes_within_budget(
    notes, budget, kept
):
    tree = ContextTree()
    texts = [
        str(i).ljust(size, ".") for i, (_, size) in enumerate(notes)
    ]
    for (scope, _), text in zip(notes, texts):
        tree.write(text, scope)
    got = tree.read(token_budget=budget)
    assert sum(got.values(), []) == [texts[i] for i in kept]


@pytest.mark.parametrize(
    "budget, demands, shares",
    [
        (10, [100, 100, 100, 0], [3, 3, 4, 0]),
        (7, [10, 10, 10, 10], [2, 2, 2, 1]),
        (60, [0, 100, 10, 0], [0, 50, 10, 0]),
        (100, [5, 5, 5, 5], [5, 5, 5, 5]),
    ],
)
def test_split_budget_uses_every_token_it_can(
    budget, demands, shares
):
    got = _split_budget(budget, demands, WEIGHTS)
    assert got == shares
    assert sum(got) == min(budget, sum(demands))


@pytest.mark.parametrize(
    "budget, shows_note", [(0, False), (7, True)]
)
def test_render_leads_with_the_whole_spec(budget, shows_note):
    spec = "Build a CSV parser that handles quoted fields. " * 10
    tree = ContextTree(spec=spec)
    tree.write("parser handles quoted commas")
    rendered = tree.render(token_budget=budget)
    assert rendered.startswith(f"Spec:\n{spec}")
    assert ("parser handles quoted commas" in rendered) is shows_note


def test_notes_survive_a_reload(tmp_path):
    path = tmp_path / "runs" / "notes.jsonl"
    first = ContextTree(path=path)
    one_per_scope(first)
    again = ContextTree(path=path)
    assert again.notes == first.notes
    assert again.read("a", 1) == first.read("a", 1)


@pytest.mark.parametrize("persist", [False, True])
def test_clear_tree_keeps_global_and_cohort(tmp_path, persist):
    path = tmp_path / "notes.jsonl" if persist else None
    tree = ContextTree(path=path)
    one_per_scope(tree)
    tree.write("b finding", "tree", tree_id="b", depth=1)
    tree.clear_tree("a")
    if persist:
        tree = ContextTree(path=path)
    assert sum(tree.read("a", 1).values(), []) == [
        "global finding",
        "cohort finding",
    ]
    assert tree.read("b", 1)["tree"] == ["b finding"]


@pytest.mark.parametrize(
    "text, scope, reply",
    [
        ("parser passes 12 of 12 cases", "level", "Saved."),
        ("Because I ran it, it works", "tree", "Rejected: A note is"),
        ("parser passes", "team", "Rejected: Unknown scope"),
        (42, "tree", "Rejected: A note must"),
    ],
)
def test_tools_work_as_agent_tools(text, scope, reply):
    tree = ContextTree(spec="Ship the parser.")
    write, read = tree.tools("Coder", tree_id="a", depth=1)
    schemas = convert_multiple_functions_to_openai_function_schema(
        [write, read]
    )
    assert len(schemas) == 2
    assert write(text, scope).startswith(reply)
    assert (str(text) in read()) is (reply == "Saved.")
    assert read().startswith("Spec:\nShip the parser.")
