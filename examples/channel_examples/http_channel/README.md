# HttpChannel examples

An `HttpChannel` keeps the conversation on another server and reaches it over
HTTP, so agents on different machines can share one conversation.

Start the server in one terminal, then run the example in another, both from
the repository root:

```bash
python examples/channel_examples/http_channel/http_channel_server.py
python examples/channel_examples/http_channel/http_channel_example.py
```

The server listens on `http://127.0.0.1:8765` and keeps each conversation in
memory, so stopping it clears them.

| Script | What it shows |
|---|---|
| [`http_channel_server.py`](http_channel_server.py) | A minimal server an `HttpChannel` can talk to, one conversation per URL path |
| [`http_channel_example.py`](http_channel_example.py) | Two agents, each with its own client, sharing the conversation at `/channels/ev-report` |

Docs: [HttpChannel](https://docs.swarms.world/api/channels/http-channel)
