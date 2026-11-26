# SqliteChannel examples

A `SqliteChannel` keeps the conversation as rows in a SQLite table. One
database file can hold many named conversations, and several threads or
processes can post to it at once.

Run from the repository root. Both scripts use `runs/channels.db`, and SQLite
does not create the folder, so make it first:

```bash
mkdir -p runs
python examples/channel_examples/sqlite_channel/sqlite_channel.py
```

| Script | What it shows |
|---|---|
| [`sqlite_channel.py`](sqlite_channel.py) | A fan-out posting into one SQLite table from several threads, then a SQL query over it |
| [`sqlite_channel_example.py`](sqlite_channel_example.py) | Three agents in one named SQLite conversation, read back with `fetch_messages` and `list_conversations` |

Docs: [SqliteChannel](https://docs.swarms.world/api/channels/sqlite-channel)
