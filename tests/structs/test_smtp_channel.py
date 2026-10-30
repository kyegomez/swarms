import os
import email
from email.message import EmailMessage
from unittest.mock import MagicMock, patch
import pytest

from swarms.structs.comms_channel import CommsChannel
from swarms.structs.smtp_channel import SmtpChannel, strip_quoted_history


class FakeSMTP:
    def __init__(self, host, port, timeout=30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.logged_in = None
        self.sent_messages = []

    def starttls(self, context=None):
        self.started_tls = True

    def login(self, username, password):
        self.logged_in = (username, password)

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.sent_messages.append((msg, from_addr, to_addrs))

    def quit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeIMAP:
    def __init__(self, host, port, ssl_context=None):
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.logged_in = None
        self.selected_mailbox = None
        self.messages_store = {}  # id -> raw_bytes
        self.deleted_ids = []
        self.expunged = False

    def login(self, username, password):
        self.logged_in = (username, password)

    def select(self, mailbox="INBOX"):
        self.selected_mailbox = mailbox
        return ("OK", [b"1"])

    def search(self, charset, *criteria):
        # Return all keys in messages_store
        ids = [str(k).encode("ascii") for k in sorted(self.messages_store.keys())]
        return ("OK", [b" ".join(ids)])

    def fetch(self, msg_id, part):
        int_id = int(msg_id.decode("ascii") if isinstance(msg_id, bytes) else msg_id)
        raw = self.messages_store.get(int_id, b"")
        return ("OK", [(f"{int_id} (RFC822 {{{len(raw)}}})".encode("ascii"), raw)])

    def store(self, msg_id, command, flags):
        int_id = int(msg_id.decode("ascii") if isinstance(msg_id, bytes) else msg_id)
        if "\\Deleted" in flags:
            self.deleted_ids.append(int_id)
        return ("OK", [b""])

    def expunge(self):
        self.expunged = True
        return ("OK", [b""])

    def logout(self):
        return ("BYE", [b"Logging out"])


class TestSmtpChannel:
    def test_inherits_comms_channel(self):
        assert issubclass(SmtpChannel, CommsChannel)

    def test_missing_password_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("SMTP_PASSWORD", raising=False)
        with pytest.raises(ValueError) as excinfo:
            SmtpChannel(
                from_address="agents@example.com",
                to="user@example.com",
                smtp_host="smtp.example.com",
                imap_host="imap.example.com",
                password=None,
            )
        assert "password" in str(excinfo.value).lower()

    def test_password_from_env_var(self, monkeypatch):
        monkeypatch.setenv("SMTP_PASSWORD", "secret123")
        monkeypatch.setenv("SMTP_USERNAME", "custom_user")
        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
        )
        assert channel.password == "secret123"
        assert channel.username == "custom_user"

    def test_username_defaults_to_from_address(self, monkeypatch):
        monkeypatch.delenv("SMTP_USERNAME", raising=False)
        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
        )
        assert channel.username == "agents@example.com"

    def test_messages_empty_before_first_post(self):
        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
        )
        assert channel.messages() == []

    @patch("smtplib.SMTP")
    def test_first_post_creates_root_message_and_bccs_from_address(self, mock_smtp):
        fake_smtp = FakeSMTP("smtp.example.com", 587)
        mock_smtp.return_value = fake_smtp

        channel = SmtpChannel(
            from_address="agents@example.com",
            to=["user@example.com", "lead@example.com"],
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
            subject="Market Analysis",
        )

        channel.post("Researcher", "Initial finding: market grew 25%.")

        assert len(fake_smtp.sent_messages) == 1
        msg, from_addr, to_addrs = fake_smtp.sent_messages[0]

        assert from_addr == "agents@example.com"
        # Bcc must include from_address along with 'to' recipients
        assert set(to_addrs) == {"user@example.com", "lead@example.com", "agents@example.com"}

        assert msg["Subject"] == "Market Analysis"
        assert msg["X-Swarms-Agent"] == "Researcher"
        assert msg["From"] == "agents@example.com"
        assert "user@example.com" in msg["To"]
        assert msg["Message-ID"] is not None
        assert channel.thread_id == msg["Message-ID"]
        assert channel.last_message_id == msg["Message-ID"]

    @patch("smtplib.SMTP")
    def test_second_post_sets_in_reply_to_and_references(self, mock_smtp):
        fake_smtp = FakeSMTP("smtp.example.com", 587)
        mock_smtp.return_value = fake_smtp

        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
        )

        channel.post("Agent1", "Turn 1")
        first_id = channel.thread_id

        channel.post("Agent2", "Turn 2")
        second_id = channel.last_message_id

        assert len(fake_smtp.sent_messages) == 2
        second_msg, _, _ = fake_smtp.sent_messages[1]

        assert second_msg["In-Reply-To"] == first_id
        assert first_id in second_msg["References"]
        assert second_msg["X-Swarms-Agent"] == "Agent2"
        assert second_id != first_id

    @patch("imaplib.IMAP4_SSL")
    def test_messages_fetches_orders_by_date_and_deduplicates(self, mock_imap_ssl):
        fake_imap = FakeIMAP("imap.example.com", 993)
        mock_imap_ssl.return_value = fake_imap

        # Build 3 raw emails: msg1 (early), msg2 (duplicate of msg1), msg3 (later)
        msg1 = EmailMessage()
        msg1["Message-ID"] = "<msg-1@example.com>"
        msg1["From"] = "agents@example.com"
        msg1["To"] = "user@example.com"
        msg1["Date"] = "Wed, 29 Oct 2026 10:00:00 +0000"
        msg1["X-Swarms-Agent"] = "Planner"
        msg1.set_content("Plan step 1.")

        # Duplicate of msg1 (e.g. from recipient copy in same inbox)
        msg2 = EmailMessage()
        msg2["Message-ID"] = "<msg-1@example.com>"
        msg2["From"] = "agents@example.com"
        msg2["To"] = "user@example.com"
        msg2["Date"] = "Wed, 29 Oct 2026 10:00:00 +0000"
        msg2["X-Swarms-Agent"] = "Planner"
        msg2.set_content("Plan step 1.")

        # msg3 from human reply without X-Swarms-Agent header
        msg3 = EmailMessage()
        msg3["Message-ID"] = "<msg-2@example.com>"
        msg3["From"] = "Alice <alice@example.com>"
        msg3["To"] = "agents@example.com"
        msg3["Date"] = "Wed, 29 Oct 2026 10:05:00 +0000"
        msg3.set_content("Looks good, proceed.\n\nOn Wed, Oct 29, 2026 at 10:00 AM Planner wrote:\n> Plan step 1.")

        fake_imap.messages_store[1] = msg3.as_bytes()  # arrives out of order
        fake_imap.messages_store[2] = msg1.as_bytes()
        fake_imap.messages_store[3] = msg2.as_bytes()

        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
            thread_id="<msg-1@example.com>",
        )

        messages = channel.messages()

        assert len(messages) == 2
        # Oldest first: msg1, then msg3
        assert messages[0] == {
            "agent_name": "Planner",
            "content": "Plan step 1.",
        }
        # Human reply speaker is From header
        assert messages[1]["agent_name"] in ["Alice <alice@example.com>", "alice@example.com"]
        assert "Looks good, proceed." in messages[1]["content"]
        assert "Plan step 1." not in messages[1]["content"]

    @patch("imaplib.IMAP4_SSL")
    def test_clear_flags_deleted_and_expunges(self, mock_imap_ssl):
        fake_imap = FakeIMAP("imap.example.com", 993)
        mock_imap_ssl.return_value = fake_imap
        fake_imap.messages_store[1] = b"raw email"

        channel = SmtpChannel(
            from_address="agents@example.com",
            to="user@example.com",
            smtp_host="smtp.example.com",
            imap_host="imap.example.com",
            password="pwd",
            thread_id="<root@example.com>",
        )

        channel.clear()

        assert fake_imap.expunged is True
        assert 1 in fake_imap.deleted_ids
        assert channel.thread_id is None
        assert channel.last_message_id is None

    def test_quote_stripping_logic(self):
        text = (
            "I approve this task.\n"
            "> Previous message line 1\n"
            ">> Nested quote\n"
            "On Wed, Oct 29, 2026 at 10:00 AM agents wrote:\n"
            "Should be removed"
        )
        cleaned = strip_quoted_history(text)
        assert cleaned == "I approve this task."
