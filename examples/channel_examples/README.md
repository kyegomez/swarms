# Comms channel examples

Every script here writes the run to a markdown file under `runs/` through a
`FileChannel`. Open the file while a script runs to watch the agents talk, or
edit it between runs to steer the next one.

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
| `resume_channel.py` | A second run that continues an existing file |
| `sqlite_channel.py` | A fan-out posting into one SQLite table from several threads, then a SQL query over it |
| `sqlite_channel_example.py` | Three agents in one named SQLite conversation, read back with `fetch_messages` and `list_conversations` |
| `queue_channel_example.py` | Three agents through a `QueueChannel` while the main thread prints each message as it lands |
| `http_channel_example.py` | Two agents sharing a conversation kept on another server through an `HttpChannel`; start `http_channel_server.py` first |
| `agentmail_channel_example.py` | Two agents sharing an AgentMail email thread through an `AgentMailChannel`; the people in `to` get each post by email and can reply |

Run any of them from the repository root, for example:

```bash
python examples/comms_channel/sequential_workflow_channel.py
cat runs/sequential.md
```
