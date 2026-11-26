# Email channel examples

These channels keep the conversation in an email thread. Each post is sent as
an email, and the people in `to` can reply from their mail client; their
replies become turns in the conversation.

Edit the addresses at the top of each script before running it. Then run it
from the repository root:

```bash
python examples/channel_examples/email/smtp_channel_example.py
```

| Script | Channel | Needs | What it shows |
|---|---|---|---|
| [`smtp_channel_example.py`](smtp_channel_example.py) | `SmtpChannel` | `SMTP_USERNAME` and `SMTP_PASSWORD` (an app password for Gmail) | Two agents sharing an email thread in your own mailbox |
| [`agentmail_channel_example.py`](agentmail_channel_example.py) | `AgentMailChannel` | `AGENTMAIL_API_KEY` and an inbox at [agentmail.to](https://agentmail.to) | Two agents sharing an AgentMail email thread |

`SmtpChannel` does not work with Microsoft 365, which no longer accepts
passwords over IMAP.

Docs: [SmtpChannel](https://docs.swarms.world/api/channels/smtp-channel),
[AgentMailChannel](https://docs.swarms.world/api/channels/agentmail-channel)
