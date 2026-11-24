# MessageBoard examples

A `MessageBoard` holds many threads, like a forum, in a SQLite database. Agents
start, read, reply to and label threads through the tools they get with
`Agent(tools=board.agent_tools)`; each post is signed with the agent's name.

Run any script from the repository root:

```bash
python examples/channel_examples/message_board/message_board_triage.py
```

| Script | What it shows | Stores the board in |
|---|---|---|
| [`message_board_agents.py`](message_board_agents.py) | A planner agent opens a thread per question; a researcher lists, answers and relabels them | memory |
| [`message_board_triage.py`](message_board_triage.py) | A bug queue run on labels: a triager opens a thread per bug, a fixer works each open one, a reviewer closes or reopens it | `runs/bugs.db` |
| [`message_board_swarms.py`](message_board_swarms.py) | A coordinator agent opens threads, then a `SequentialWorkflow` and a `ConcurrentWorkflow` each work in one at the same time | memory |
| [`message_board_resume.py`](message_board_resume.py) | A board stored in a SQLite file, loaded back by its id with `MessageBoard.load`, and continued by another agent | `runs/board.db` |

Docs: [MessageBoard](https://docs.swarms.world/api/channels/message-board)
