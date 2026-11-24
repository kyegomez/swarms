# QueueChannel examples

A `QueueChannel` keeps the conversation in memory and hands each message to a
queue as it is posted, so another thread in the same process can react to it
straight away.

Run from the repository root:

```bash
python examples/channel_examples/queue_channel/queue_channel_example.py
```

| Script | What it shows |
|---|---|
| [`queue_channel_example.py`](queue_channel_example.py) | Three agents through a `QueueChannel` while the main thread prints each message as it lands |

Docs: [QueueChannel](https://docs.swarms.world/api/channels/queue-channel)
