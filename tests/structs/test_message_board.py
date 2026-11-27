import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from swarms import Agent, MessageBoard


def tools(board, agent_name):
    return {
        tool.__name__: tool.bind_agent(agent_name)
        for tool in board.agent_tools
    }


def start(board, agent_name, title, content, labels=""):
    created = tools(board, agent_name)["create_thread"](
        title, content, labels
    )
    return created.split()[2].rstrip(":")


def rows(board, thread_id):
    return [
        (message["agent_name"], message["content"])
        for message in board.channel(thread_id).messages()
    ]


def test_agents_start_read_reply_to_and_label_threads():
    board = MessageBoard()
    triager = tools(board, "Triager")
    fixer = tools(board, "Fixer")

    created = triager["create_thread"](
        "Flaky test", "Fails 1 in 20.", "kind=bug, status=open"
    )
    thread_id = created.split()[2].rstrip(":")
    other = start(board, "Triager", "Docs typo", "Typo on home.")

    assert created == f"Created thread {thread_id}: Flaky test"
    assert fixer["list_threads"]("kind=bug") == (
        f"{thread_id} | Flaky test | by Triager | "
        "labels: kind=bug, status=open | 1 messages"
    )
    assert fixer["reply_to_thread"](thread_id, "Fixed the race.") == (
        f"Replied to thread {thread_id}"
    )
    assert fixer["label_thread"](
        thread_id, "status=fixed, kind="
    ) == (f"Thread {thread_id} labels: status=fixed")
    assert fixer["read_thread"](thread_id) == (
        f"Thread {thread_id}: Flaky test (started by Triager, "
        "labels: status=fixed)\n\n"
        "Triager: Fails 1 in 20.\n\nFixer: Fixed the race."
    )
    assert rows(board, other) == [("Triager", "Typo on home.")]
    assert [t["id"] for t in board.threads()] == [thread_id, other]


@pytest.mark.parametrize(
    "wanted, titles",
    [
        ("", ["Docs typo", "Race in fan-out", "Crash on load"]),
        ("kind=bug", ["Race in fan-out", "Crash on load"]),
        ("kind=bug, status=open", ["Crash on load"]),
        ("kind=feature", []),
    ],
)
def test_list_threads_filters_on_labels(wanted, titles):
    board = MessageBoard()
    start(board, "A", "Crash on load", "x", "kind=bug, status=open")
    start(board, "A", "Race in fan-out", "x", "kind=bug")
    start(board, "A", "Docs typo", "x", "kind=docs")

    listed = tools(board, "B")["list_threads"](wanted)

    if not titles:
        assert listed == "No threads match."
    else:
        found = [line.split(" | ")[1] for line in listed.splitlines()]
        assert found == titles


def test_concurrent_replies_are_all_kept():
    board = MessageBoard()
    thread_id = start(board, "Lead", "Load test", "Go.")

    def post(worker):
        reply = tools(board, f"worker-{worker}")["reply_to_thread"]
        for n in range(25):
            reply(thread_id, f"{worker}-{n}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))

    assert len(rows(board, thread_id)) == 201


def test_concurrent_labels_from_two_loaded_boards_are_all_kept(
    tmp_path,
):
    db_path = str(tmp_path / "boards.db")
    board = MessageBoard(db_path=db_path)
    fixer = MessageBoard.load(db_path, board.id)
    reviewer = MessageBoard.load(db_path, board.id)
    thread_ids = [
        start(board, "Triager", f"Bug {n}", "Report.", "kind=bug")
        for n in range(50)
    ]

    for thread_id in thread_ids:
        with ThreadPoolExecutor(max_workers=2) as pool:
            pool.submit(
                fixer.set_labels, thread_id, {"status": "fixed"}
            )
            pool.submit(
                reviewer.set_labels, thread_id, {"reviewer": "Kye"}
            )

    assert [t["labels"] for t in board.threads()] == [
        {"kind": "bug", "status": "fixed", "reviewer": "Kye"}
    ] * 50


def test_a_board_is_stored_in_sqlite_and_loads_back(tmp_path):
    db_path = str(tmp_path / "boards.db")
    board = MessageBoard(
        name="eng", description="Bug triage", db_path=db_path
    )
    other = MessageBoard(name="ops", db_path=db_path)
    first = start(board, "Triager", "First", "Opening post.")
    second = start(board, "Triager", "Second", "Bug.", "kind=bug")
    start(other, "Pager", "Disk full", "On db-2.")
    board.reply(first, "Kye", "Looking into it.")
    board.set_labels(second, {"status": "open"})
    board.close()

    reloaded = MessageBoard.load(db_path, board.id)
    listed = MessageBoard.list_boards(db_path)

    assert board.id != other.id
    assert (reloaded.name, reloaded.description) == (
        "eng",
        "Bug triage",
    )
    assert [(b["id"], b["name"]) for b in listed] == [
        (board.id, "eng"),
        (other.id, "ops"),
    ]
    assert [t["title"] for t in reloaded.threads()] == [
        "First",
        "Second",
    ]
    assert reloaded.threads()[1]["labels"] == {
        "kind": "bug",
        "status": "open",
    }
    assert rows(reloaded, first) == [
        ("Triager", "Opening post."),
        ("Kye", "Looking into it."),
    ]
    with pytest.raises(KeyError, match="nope"):
        MessageBoard.load(db_path, "nope")


def agent_with_board_tools(board, name, call):
    agent = Agent(
        agent_name=name,
        model_name="gpt-5.4",
        max_loops=2,
        print_on=False,
        autosave=False,
        tools=board.agent_tools,
    )
    replies = [[call], "Done."]
    agent.call_llm = lambda *args, **kwargs: replies.pop(0)
    return agent


def tool_call(name, **arguments):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def test_agents_sharing_board_agent_tools_post_as_themselves():
    board = MessageBoard()
    triager = agent_with_board_tools(
        board,
        "Triager",
        tool_call(
            "create_thread",
            title="Login times out",
            content="Slow networks time out on /login.",
            labels="kind=bug",
        ),
    )

    triager.run("Open a thread for the bug in this report.")
    [thread] = board.threads()
    fixer = agent_with_board_tools(
        board,
        "Fixer",
        tool_call(
            "reply_to_thread",
            thread_id=thread["id"],
            content="Raised the timeout.",
        ),
    )
    fixer.run("Fix the bug.")

    assert (thread["title"], thread["author"], thread["labels"]) == (
        "Login times out",
        "Triager",
        {"kind": "bug"},
    )
    assert rows(board, thread["id"]) == [
        ("Triager", "Slow networks time out on /login."),
        ("Fixer", "Raised the timeout."),
    ]
    assert f"Created thread {thread['id']}" in (
        triager.short_memory.get_str()
    )
    assert board.agent_tools[0]("T", "x").startswith(
        "These tools post as the agent that holds them"
    )
    assert board.threads()[0]["messages"] == 2


@pytest.mark.parametrize(
    "tool, args",
    [
        ("read_thread", ("missing",)),
        ("reply_to_thread", ("missing", "hi")),
        ("label_thread", ("missing", "a=b")),
    ],
)
def test_a_tool_on_an_unknown_thread_says_so(tool, args):
    board = MessageBoard()

    assert tools(board, "A")[tool](*args) == (
        "No thread 'missing' on board 'board'"
    )
    with pytest.raises(KeyError, match="missing"):
        board.reply("missing", "Kye", "hi")
