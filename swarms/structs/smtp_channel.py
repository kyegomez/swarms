"""A CommsChannel kept as one email thread, sent over SMTP and read over IMAP."""

import imaplib
import os
import re
import smtplib
import ssl
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from email import message_from_bytes, policy
from email.message import EmailMessage
from email.utils import format_datetime, parseaddr
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from swarms.structs.comms_channel import CommsChannel

_EPOCH = datetime.fromtimestamp(0, timezone.utc)
_QUOTED = re.compile(
    r"^(?:_+\n)?From:.*\n(?:Sent|Date):[\s\S]*"
    r"|^-{2,}\s*Original Message\s*-{2,}[ \t]*$[\s\S]*"
    r"|^On .*(?:\n.*)?wrote:[ \t]*$(?:\n[ \t]*(?:>.*)?)*\Z",
    re.MULTILINE | re.IGNORECASE,
)


class SmtpChannel(CommsChannel):
    """A conversation kept as one email thread in a mailbox you already run.

    Each post is emailed to `to`, who can reply from their own mail client,
    and to `from_address`, whose mailbox `messages()` reads over IMAP. A post
    reads back exactly as it was sent. A person's reply is spoken by its From
    address, with the quoted history below it removed.

    Posts made through this instance are visible at once. Another instance
    or process sees a post only after the mailbox receives its copy, so until
    then it can miss an answer or post a task again.

    Gmail needs an app password; if posts do not reach its inbox, use
    mailbox="[Gmail]/All Mail". Microsoft 365 is not supported, because it no
    longer accepts passwords over IMAP. clear() needs IMAP UIDPLUS.

    Args:
        from_address: The mailbox the agents send as and read from.
        to: Who receives each post and can reply.
        smtp_host: The SMTP server.
        imap_host: The IMAP server.
        username: Defaults to SMTP_USERNAME, then from_address.
        password: Defaults to SMTP_PASSWORD. Missing raises ValueError.
        smtp_port: 465 uses implicit TLS; any other port uses STARTTLS.
        imap_port: IMAP over TLS.
        mailbox: The IMAP folder holding the thread.
        subject: The thread's subject line.
        thread_id: The Message-ID of a thread's first message, to join it.
        timeout: Seconds to wait on each server.
    """

    def __init__(
        self,
        from_address: str,
        to: Union[str, List[str]],
        smtp_host: str,
        imap_host: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        smtp_port: int = 587,
        imap_port: int = 993,
        mailbox: str = "INBOX",
        subject: str = "Swarms conversation",
        thread_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.username = (
            username
            or os.environ.get("SMTP_USERNAME")
            or from_address
        )
        self._password = password or os.environ.get("SMTP_PASSWORD")
        if not self._password:
            raise ValueError("SmtpChannel needs a password")
        # Python before 3.13 sends IMAP arguments to the server unescaped.
        joined = (
            f"{mailbox}{self.username}{self._password}{thread_id}"
        )
        if re.search(r"[\x00\r\n]", joined):
            raise ValueError("NUL, CR and LF are not allowed")
        # IMAP HEADER search matches substrings, and clear() deletes the matches.
        if thread_id and not re.fullmatch(r"<[^\s<>]+>", thread_id):
            raise ValueError("thread_id must look like <id@domain>")
        self.from_address = from_address
        self.to = [to] if isinstance(to, str) else list(to)
        self.smtp_host = smtp_host
        self.imap_host = imap_host
        self.smtp_port = smtp_port
        self.imap_port = imap_port
        self.mailbox = mailbox
        self.subject = subject
        self.thread_id = thread_id
        self.timeout = timeout
        self._posted: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    def post(self, agent_name: str, content: str) -> None:
        """
        Email one message to the thread.

        Args:
            agent_name (str): Who is speaking.
            content (str): The message text.
        """
        text = str(content)
        domain = self.from_address.rpartition("@")[2]
        message_id = f"<swarms-{uuid.uuid4().hex}@{domain}>"
        date = datetime.now(timezone.utc).replace(microsecond=0)
        msg = EmailMessage()
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.to)
        msg["Subject"] = self.subject
        msg["Date"] = format_datetime(date)
        msg["Message-ID"] = message_id
        msg["X-Swarms-Agent"] = agent_name
        # Base64 of the exact bytes: Agent.run finds its task by exact text.
        msg.set_content(text.encode(), "text", "plain", cte="base64")
        msg.set_param("charset", "utf-8")
        context = ssl.create_default_context()
        with self._lock:
            if self.thread_id:
                chain = dict.fromkeys([self.thread_id, *self._posted])
                msg["In-Reply-To"] = list(chain)[-1]
                msg["References"] = " ".join(chain)
            implicit = self.smtp_port == 465
            with (smtplib.SMTP_SSL if implicit else smtplib.SMTP)(
                self.smtp_host,
                self.smtp_port,
                timeout=self.timeout,
                **({"context": context} if implicit else {}),
            ) as smtp:
                if not implicit:
                    smtp.starttls(context=context)
                smtp.login(self.username, self._password)
                refused = smtp.send_message(
                    msg, to_addrs=[*self.to, self.from_address]
                )
                if refused:
                    logger.warning(f"{list(refused)} refused a post")
                # Recorded before QUIT: the message is out even if QUIT fails.
                self.thread_id = self.thread_id or message_id
                row = (date, 0, len(self._posted), agent_name, text)
                self._posted[message_id] = row

    def messages(self) -> List[Dict[str, Any]]:
        """
        Read the thread, oldest first. This instance's posts need no delivery.

        Returns:
            List[Dict[str, Any]]: Rows shaped like {"agent_name": ..., "content": ...}.
        """
        with self._lock:
            if not self.thread_id:
                return []
            with self._thread() as (imap, uids):
                fetched = uids and _ok(
                    imap.uid("FETCH", uids, "(BODY.PEEK[])")
                )
            rows = dict(
                _row(item[1], n)
                for n, item in enumerate(fetched)
                if isinstance(item, tuple)
            )
            rows.update(self._posted)
        return [
            {"agent_name": agent_name, "content": content}
            for *_, agent_name, content in sorted(rows.values())
        ]

    def clear(self) -> None:
        """
        Delete the thread from the mailbox and start a new one on the next post.

        Needs UIDPLUS, as a plain EXPUNGE also removes unrelated mail flagged deleted.
        """
        with self._lock:
            if not self.thread_id:
                return
            with self._thread() as (imap, uids):
                capabilities = _ok(imap.capability())[0]
                if b"UIDPLUS" not in capabilities.upper().split():
                    raise imaplib.IMAP4.error(
                        "SmtpChannel.clear needs IMAP UIDPLUS"
                    )
                if uids:
                    flag = "(\\Deleted)"
                    _ok(imap.uid("STORE", uids, "+FLAGS", flag))
                    _ok(imap.uid("EXPUNGE", uids))
            self.thread_id = None
            self._posted = {}

    def close(self) -> None:
        """Do nothing: each read logs in and out of IMAP by itself."""

    @contextmanager
    def _thread(self):
        with imaplib.IMAP4_SSL(
            self.imap_host,
            self.imap_port,
            ssl_context=ssl.create_default_context(),
            timeout=self.timeout,
        ) as imap:
            # Debug output would print the LOGIN command with the password.
            imap.debug = 0
            imap.login(self.username, self._password)
            _ok(imap.select(_quote(self.mailbox)))
            root = _quote(self.thread_id)
            query = f"OR HEADER Message-ID {root} HEADER References {root}"
            found = _ok(imap.uid("SEARCH", query))
            yield imap, ",".join((found[0] or b"").decode().split())


def _ok(result) -> list:
    status, data = result
    if status != "OK":
        raise imaplib.IMAP4.error(f"IMAP command failed: {data!r}")
    return data


def _quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _row(raw: bytes, n: int) -> tuple:
    msg = message_from_bytes(raw, policy=policy.default)
    date = getattr(msg["Date"], "datetime", None) or _EPOCH
    date = date.replace(tzinfo=date.tzinfo or timezone.utc)
    body = msg.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body is not None else ""
    agent = msg["X-Swarms-Agent"]
    if agent is None:
        agent = parseaddr(str(msg["From"]))[1]
        lines = _QUOTED.sub("", text.replace("\r\n", "\n")).split(
            "\n"
        )
        while lines and lines[-1].lstrip()[:1] in ("", ">"):
            lines.pop()
        text = "\n".join(lines).strip()
    key = str(msg["Message-ID"] or n).strip()
    return key, (date, 1, n, str(agent), text)
