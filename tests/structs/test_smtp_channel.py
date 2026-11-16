import imaplib
import smtplib
from datetime import datetime, timedelta, timezone
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

from swarms.structs.agent import Agent
from swarms.structs.smtp_channel import SmtpChannel

MAILBOX = "agents@yourco.com"


def parse(raw):
    return BytesParser(policy=policy.default).parsebytes(raw)


@pytest.fixture
def server(monkeypatch):
    """An SMTP relay delivering into one IMAP mailbox, unless held."""
    srv = SimpleNamespace(
        mail=[], sent=[], hold=False, no=set(), uidplus=True
    )
    srv.deliver = lambda raw, flagged=False: srv.mail.append(
        {"uid": len(srv.mail) + 100, "raw": raw, "flagged": flagged}
    )

    def send(msg, to_addrs):
        raw = msg.as_bytes(policy=policy.SMTP)
        srv.sent.append((parse(raw), to_addrs))
        if not srv.hold:
            srv.deliver(raw)

    def uid(command, *args):
        if command in srv.no:
            return "NO", [b"server said no"]
        if command == "SEARCH":
            root = args[0].split('"')[1]
            hits = [
                str(m["uid"]).encode()
                for m in srv.mail
                if root in str(parse(m["raw"])["Message-ID"])
                or root in str(parse(m["raw"])["References"])
            ]
            return "OK", [b" ".join(hits) or None]
        uids = [int(u) for u in args[0].split(",")]
        chosen = [m for m in srv.mail if m["uid"] in uids]
        if command == "FETCH":
            assert args[1] == "(BODY.PEEK[])"
            data = []
            for m in chosen:
                data += [(b"1 (BODY[] {1}", m["raw"]), b")"]
            return "OK", data
        if command == "STORE":
            for m in chosen:
                m["flagged"] = True
        if command == "EXPUNGE":
            srv.mail = [m for m in srv.mail if m not in chosen]
        return "OK", [None]

    srv.smtp, srv.imap = MagicMock(), MagicMock()
    srv.smtp.__enter__.return_value = srv.smtp
    srv.smtp.send_message.side_effect = send
    srv.imap.__enter__.return_value = srv.imap
    srv.imap.uid.side_effect = uid
    srv.imap.select.return_value = ("OK", [b"1"])
    srv.imap.capability.side_effect = lambda: (
        "OK",
        [b"IMAP4rev1 UIDPLUS" if srv.uidplus else b"IMAP4rev1"],
    )
    for name in ("SMTP", "SMTP_SSL"):
        monkeypatch.setattr(
            smtplib, name, MagicMock(return_value=srv.smtp)
        )
    monkeypatch.setattr(
        imaplib, "IMAP4_SSL", MagicMock(return_value=srv.imap)
    )
    return srv


def channel(**kwargs):
    kwargs.setdefault("password", "app-password")
    return SmtpChannel(
        from_address=MAILBOX,
        to="you@yourco.com",
        smtp_host="smtp.yourco.com",
        imap_host="imap.yourco.com",
        **kwargs,
    )


def reply(sender, root, body, minutes, message_id=None):
    msg = EmailMessage()
    msg["From"] = f"Someone <{sender}>"
    msg["Date"] = format_datetime(
        datetime.now(timezone.utc) + timedelta(minutes=minutes)
    )
    if message_id:
        msg["Message-ID"] = message_id
    msg["References"] = root
    msg.set_content(body)
    return msg.as_bytes(policy=policy.SMTP)


@pytest.mark.parametrize(
    "port, steps",
    [
        (587, ["starttls", "login", "send_message"]),
        (465, ["login", "send_message"]),
    ],
)
def test_a_thread_round_trips_through_the_mailbox(
    server, port, steps
):
    posts = [
        ("Human", "Report on EVs."),
        ("Researcher", "> Note: draft\nGrowth was 31%.\n"),
        ("Writer", "Café one\r\nline two"),
    ]
    writer = channel(smtp_port=port)
    for agent, body in posts:
        writer.post(agent, body)

    made = [call[0] for call in server.smtp.method_calls]
    assert made[: len(steps)] == steps
    (first, _), (second, _), (third, recipients) = server.sent
    root = first["Message-ID"]
    assert first["In-Reply-To"] is None
    assert second["In-Reply-To"] == root
    assert third["In-Reply-To"] == second["Message-ID"]
    assert third["References"].split() == [root, second["Message-ID"]]
    assert third["Bcc"] is None
    assert recipients == ["you@yourco.com", MAILBOX]

    # (sender, minutes, written, read); from 60 on, no Message-ID.
    replies = [
        ("o@x", 30, "Hi.\n\n___\nFrom: A\nSent: Tue\n\nOld.", "Hi."),
        (
            "g@x",
            10,
            "Hi.\n\nOn Tue, A <\na@x> wrote:\n> Old.\n",
            "Hi.",
        ),
        (
            "i@x",
            20,
            "On A wrote:\n> B\nWhy?\n> C\n",
            "On A wrote:\n> B\nWhy?",
        ),
        ("r@x", 40, "Hi.\n\n-----Original Message-----\nOld.", "Hi."),
        ("s@x", 50, "Sign here:\n____", "Sign here:\n____"),
        ("n@x", 60, "One.", "One."),
        ("n@x", 70, "Two.", "Two."),
    ]
    for sender, minutes, body, _ in replies:
        message_id = f"<{minutes}@x>" if minutes < 60 else None
        server.deliver(reply(sender, root, body, minutes, message_id))
    server.deliver(server.mail[4]["raw"])
    server.deliver(reply("z@x", "<other@x>", "unrelated", 5))

    assert channel(thread_id=root).messages() == [
        {"agent_name": speaker, "content": text}
        for speaker, text in [
            *posts,
            *(
                (row[0], row[3])
                for row in sorted(replies, key=lambda row: row[1])
            ),
        ]
    ]


def test_agents_sharing_a_channel_see_answers_before_delivery(server):
    server.hold = True
    chan = channel()

    def agent(name, answer):
        with patch("swarms.agents.tool_manager.LiteLLM"):
            built = Agent(
                agent_name=name,
                max_loops=1,
                print_on=False,
                channel=chan,
            )
        built.call_llm = Mock(return_value=answer)
        return built

    researcher = agent("Researcher", "Growth was 31%.")
    writer = agent("Writer", "Report.")
    researcher.run("Write the report.")
    assert len(chan.messages()) == 2
    root = server.sent[0][0]["Message-ID"]
    server.deliver(reply("a@x.com", root, "Add charts.", 1, "<a@x>"))
    writer.run("Write the report.")

    request = writer.call_llm.call_args.kwargs["messages"]
    assert {"Growth was 31%.", "Add charts."} <= {
        m["content"] for m in request
    }
    speakers = [researcher.user_name, "Researcher", "Writer"]
    assert [m["X-Swarms-Agent"] for m, _ in server.sent] == speakers
    for message, _ in server.sent:
        server.deliver(message.as_bytes(policy=policy.SMTP))
    assert [m["agent_name"] for m in chan.messages()] == [
        *speakers,
        "a@x.com",
    ]


def test_clear_deletes_only_the_thread_and_only_on_success(server):
    server.deliver(
        reply("y@x.com", "<y@x>", "mine", 0, "<k@x>"), True
    )
    chan = channel()
    chan.post("Human", "first thread")
    before = chan.messages()

    server.uidplus = False
    with pytest.raises(imaplib.IMAP4.error):
        chan.clear()
    assert not server.mail[1]["flagged"]
    server.uidplus = True
    for command in ("SEARCH", "STORE", "EXPUNGE"):
        server.no = {command}
        with pytest.raises(imaplib.IMAP4.error):
            chan.clear()
        server.no = set()
        assert chan.messages() == before

    chan.clear()
    chan.clear()
    assert [m["uid"] for m in server.mail] == [100]
    assert chan.messages() == []
    chan.post("Human", "second thread")
    assert server.sent[1][0]["In-Reply-To"] is None


def test_failures_leave_a_consistent_thread(server, monkeypatch):
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    for bad in (
        {"password": None},
        {"mailbox": "INBOX\r\nX DELETE A"},
        {"thread_id": "swarms"},
    ):
        with pytest.raises(ValueError):
            channel(**bad)

    chan = channel()
    chan.post("Human", "kept")
    deliver = server.smtp.send_message.side_effect
    server.smtp.send_message.side_effect = smtplib.SMTPDataError(
        554, b""
    )
    with pytest.raises(smtplib.SMTPException):
        chan.post("Researcher", "lost")
    server.smtp.send_message.side_effect = deliver
    assert [m["content"] for m in chan.messages()] == ["kept"]

    server.hold = True
    server.smtp.__exit__.side_effect = smtplib.SMTPResponseException(
        421, b"closing"
    )
    with pytest.raises(smtplib.SMTPException):
        chan.post("Writer", "sent before QUIT failed")
    server.smtp.__exit__.side_effect = None
    assert [m["content"] for m in chan.messages()] == [
        "kept",
        "sent before QUIT failed",
    ]

    server.no = {"SEARCH"}
    with pytest.raises(imaplib.IMAP4.error):
        chan.messages()
