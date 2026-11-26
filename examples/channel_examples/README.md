# Comms channel examples

A comms channel is where a conversation between agents lives. Pass one as
`channel=` to an `Agent` or a swarm, and every turn is posted there. Each
folder holds the examples for one kind of channel, with a README that lists
them.

| Folder | Channel | Where the conversation lives | Examples |
|---|---|---|---|
| [`file_channel/`](file_channel) | `FileChannel` | A markdown file you can read and edit | 12 |
| [`sqlite_channel/`](sqlite_channel) | `SqliteChannel` | A table in a SQLite database | 2 |
| [`queue_channel/`](queue_channel) | `QueueChannel` | A queue that another thread reads as messages land | 1 |
| [`http_channel/`](http_channel) | `HttpChannel` | Another server, reached over HTTP | 2 |
| [`email/`](email) | `SmtpChannel`, `AgentMailChannel` | An email thread people can reply to | 2 |
| [`message_board/`](message_board) | `MessageBoard` | Many threads in a SQLite database, worked by agents through tools | 4 |

Run any script from the repository root, for example:

```bash
python examples/channel_examples/file_channel/sequential_workflow_channel.py
cat runs/sequential.md
```

Docs: [Comms Channels overview](https://docs.swarms.world/api/channels/overview)
