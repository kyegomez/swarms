# Comms channel examples

Each folder holds the examples for one kind of channel. Run any script from
the repository root; scripts that write files put them under `runs/`.

```bash
python examples/channel_examples/file_channel/sequential_workflow_channel.py
cat runs/sequential.md
```

| Folder | Channel | Where the conversation lives |
|---|---|---|
| [`file_channel/`](file_channel) | `FileChannel` | A markdown file you can read and edit |
| [`sqlite_channel/`](sqlite_channel) | `SqliteChannel` | A table in a SQLite database |
| [`queue_channel/`](queue_channel) | `QueueChannel` | A queue in this process |
| [`http_channel/`](http_channel) | `HttpChannel` | Another server, over HTTP |
| [`email/`](email) | `SmtpChannel`, `AgentMailChannel` | An email thread |
| [`message_board/`](message_board) | `MessageBoard` | Many threads in a SQLite database |

## file_channel

Every script writes the run to a markdown file under `runs/`. Open the file
while a script runs to watch the agents talk, or edit it between runs to steer
the next one.

| Script | What it shows |
|---|---|
| `agent_channel.py` | Two lone agents sharing one file through `Agent(channel=...)` |
| `sequential_workflow_channel.py` | A pipeline whose transcript lands in the file in step order |
| `concurrent_workflow_channel.py` | A fan-out whose answers are posted in agent order, never interleaved |
| `agent_rearrange_channel.py` | A flow with a parallel step, recorded by the structure |
| `graph_workflow_channel.py` | A DAG that posts each node's output as it completes |
| `mixture_of_agents_channel.py` | Workers plus an aggregator, with the synthesis as the last turn |
| `groupchat_channel.py` | A self-selecting chat whose accepted messages become the file |
| `hierarchical_swarm_channel.py` | A director's plan and each worker's report under their names |
| `planner_worker_swarm_channel.py` | A planner's steps and the workers that execute them |
| `majority_voting_channel.py` | Every vote and the consensus |
| `swarm_router_channel.py` | One channel forwarded to whichever swarm the router builds |
| `resume_channel.py` | A second run that continues an existing file; run `sequential_workflow_channel.py` first |

## sqlite_channel

| Script | What it shows |
|---|---|
| `sqlite_channel.py` | A fan-out posting into one SQLite table from several threads, then a SQL query over it |
| `sqlite_channel_example.py` | Three agents in one named SQLite conversation, read back with `fetch_messages` and `list_conversations` |

## queue_channel

| Script | What it shows |
|---|---|
| `queue_channel_example.py` | Three agents through a `QueueChannel` while the main thread prints each message as it lands |

## http_channel

| Script | What it shows |
|---|---|
| `http_channel_server.py` | A small server that keeps the conversation; start it first |
| `http_channel_example.py` | Two agents sharing the conversation on that server through an `HttpChannel` |

## email

| Script | What it shows |
|---|---|
| `smtp_channel_example.py` | Two agents sharing an email thread in your own mailbox through an `SmtpChannel`; people in `to` can reply from their mail client |
| `agentmail_channel_example.py` | Two agents sharing an AgentMail email thread through an `AgentMailChannel`; the people in `to` get each post by email and can reply |

## message_board

| Script | What it shows |
|---|---|
| `message_board_agents.py` | A planner agent opens `MessageBoard` threads with its tools; a researcher lists, answers and relabels them |
| `message_board_triage.py` | A bug queue run by agents: a triager opens a thread per bug, a fixer works each open one, a reviewer closes or reopens it |
| `message_board_swarms.py` | A coordinator agent opens threads, then a `SequentialWorkflow` and a `ConcurrentWorkflow` each work in one at the same time |
| `message_board_resume.py` | A board stored in a SQLite file, loaded back by its id with `MessageBoard.load`, and continued by another agent through its tools |
