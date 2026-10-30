"""A CommsChannel backed by an email thread over SMTP and IMAP."""

import email
import email.utils
from email.message import EmailMessage
import imaplib
import os
import re
import smtplib
import ssl
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from swarms.structs.comms_channel import CommsChannel

def _is_thread_message(msg: email.message.Message, thread_id: str) -> bool:
    """Check if an email message belongs strictly to the given thread_id."""
    if not thread_id:
        return False
    target = thread_id.strip()
    target_unwrapped = target.strip("<>")

    msg_id = (msg.get("Message-ID") or "").strip()
    if msg_id and (msg_id == target or msg_id.strip("<>") == target_unwrapped):
        return True

    references = msg.get("References") or ""
    if target in references or target_unwrapped in references:
        return True

    in_reply_to = (msg.get("In-Reply-To") or "").strip()
    if in_reply_to and (in_reply_to == target or in_reply_to.strip("<>") == target_unwrapped or target in in_reply_to):
        return True

    return False



def strip_quoted_history(text: str) -> str:
    """Remove quoted email history (lines starting with > or after 'On ... wrote:').

    Args:
        text (str): Raw email body text.

    Returns:
        str: Cleaned text without quoted replies.
    """
    lines = text.splitlines()
    cleaned_lines = []
    quote_intro_regex = re.compile(r"^on\s.+wrote:\s*$", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        # Cut off everything once an "On ... wrote:" header line is encountered
        if quote_intro_regex.match(stripped):
            break
        # Skip quoted lines starting with '>'
        if stripped.startswith(">"):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


class SmtpChannel(CommsChannel):
    """An email thread shared by agents and human collaborators.

    One SmtpChannel corresponds to one email thread. Agents post their
    turns as emails via SMTP, and read prior turns over IMAP before each run.
    Sent messages are Bcc'd to `from_address` so they appear in the IMAP mailbox
    for unified single-thread retrieval.

    Args:
        from_address (str): The email address agents send from.
        to (Union[str, List[str]]): Recipient email address(es).
        smtp_host (str): SMTP server hostname.
        imap_host (str): IMAP server hostname.
        username (Optional[str]): SMTP/IMAP username (defaults to SMTP_USERNAME or from_address).
        password (Optional[str]): SMTP/IMAP password (defaults to SMTP_PASSWORD env var).
        smtp_port (int): SMTP port (default 587 uses STARTTLS, 465 uses implicit SSL).
        imap_port (int): IMAP port (default 993 uses IMAP4_SSL).
        mailbox (str): IMAP mailbox to read and search (default "INBOX").
        subject (str): Thread subject line (default "Swarms conversation").
        thread_id (Optional[str]): Root Message-ID to join an existing thread.
        timeout (float): Connection timeout in seconds (default 30.0).
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
        self.from_address = from_address
        self.to = [to] if isinstance(to, str) else list(to)
        self.smtp_host = smtp_host
        self.imap_host = imap_host
        self.username = username or os.environ.get("SMTP_USERNAME") or from_address
        
        resolved_password = password or os.environ.get("SMTP_PASSWORD")
        if not resolved_password:
            raise ValueError(
                "Password is required for SmtpChannel. Pass password explicitly or set SMTP_PASSWORD env var."
            )
        self.password = resolved_password

        self.smtp_port = smtp_port
        self.imap_port = imap_port
        self.mailbox = mailbox
        self.subject = subject
        self.thread_id = thread_id
        self.last_message_id = thread_id
        self.timeout = timeout
        self._imap_client: Optional[imaplib.IMAP4_SSL] = None

    def _get_domain(self) -> str:
        """Extract domain from from_address, or fallback to localhost."""
        if "@" in self.from_address:
            return self.from_address.split("@", 1)[1].strip()
        return "localhost"

    def _generate_message_id(self) -> str:
        """Generate a unique Message-ID header value for a post."""
        return f"<swarms-{uuid.uuid4().hex}@{self._get_domain()}>"

    def post(self, agent_name: str, content: str) -> None:
        """Append one message to the email thread over SMTP.

        Args:
            agent_name (str): Who is speaking.
            content (str): The message text.
        """
        msg = EmailMessage()
        msg_id = self._generate_message_id()
        msg["Message-ID"] = msg_id
        msg["From"] = self.from_address
        msg["To"] = ", ".join(self.to)
        msg["Subject"] = self.subject
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["X-Swarms-Agent"] = agent_name

        if self.thread_id is None:
            self.thread_id = msg_id
            self.last_message_id = msg_id
        else:
            msg["In-Reply-To"] = self.last_message_id or self.thread_id
            msg["References"] = f"{self.thread_id} {self.last_message_id or ''}".strip()
            self.last_message_id = msg_id

        msg.set_content(str(content))

        # Bcc from_address so sent messages land in the IMAP mailbox for search
        all_recipients = list(dict.fromkeys(self.to + [self.from_address]))

        ssl_ctx = ssl.create_default_context()
        if self.smtp_port == 465:
            with smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, context=ssl_ctx, timeout=self.timeout) as smtp:
                smtp.login(self.username, self.password)
                smtp.send_message(msg, from_addr=self.from_address, to_addrs=all_recipients)
        else:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=self.timeout) as smtp:
                smtp.starttls(context=ssl_ctx)
                smtp.login(self.username, self.password)
                smtp.send_message(msg, from_addr=self.from_address, to_addrs=all_recipients)

        logger.debug(f"Posted message from '{agent_name}' with Message-ID {msg_id}")

    def _get_imap_client(self) -> imaplib.IMAP4_SSL:
        """Obtain or reconnect an authenticated IMAP4_SSL client."""
        if self._imap_client is None:
            ssl_ctx = ssl.create_default_context()
            client = imaplib.IMAP4_SSL(self.imap_host, self.imap_port, ssl_context=ssl_ctx)
            client.login(self.username, self.password)
            self._imap_client = client
        return self._imap_client

    def _find_thread_messages(
        self, client: imaplib.IMAP4_SSL
    ) -> List[Tuple[Any, email.message.Message]]:
        """Locate and verify all messages in the mailbox belonging to this thread.

        Returns:
            List[Tuple[Any, email.message.Message]]: List of (msg_id, parsed_msg) pairs
            that strictly belong to self.thread_id.
        """
        if self.thread_id is None:
            return []

        client.select(self.mailbox)

        # 1. Try search criteria targeted at Message-ID and References
        candidate_ids = []
        search_queries = [
            f'OR HEADER Message-ID "{self.thread_id}" HEADER References "{self.thread_id}"',
            f'HEADER Message-ID "{self.thread_id}"',
            f'HEADER References "{self.thread_id}"',
        ]

        for q in search_queries:
            try:
                status, data = client.search(None, q)
                if status == "OK" and data and data[0]:
                    for mid in data[0].split():
                        if mid not in candidate_ids:
                            candidate_ids.append(mid)
                    if candidate_ids:
                        break
            except Exception as e:
                logger.debug(f"IMAP search query '{q}' failed: {e}")

        # If the IMAP server failed (status != OK or exception) or returned no results
        # (e.g. server answers BAD or does not support HEADER search), fallback to ALL
        # to inspect headers client-side safely.
        if not candidate_ids:
            try:
                status, data = client.search(None, "ALL")
                if status == "OK" and data and data[0]:
                    candidate_ids = data[0].split()
            except Exception as e:
                logger.debug(f"IMAP search ALL failed: {e}")

        if not candidate_ids:
            return []

        verified_messages = []
        seen_message_ids = set()

        for mid in candidate_ids:
            try:
                res_status, fetch_data = client.fetch(mid, "(RFC822)")
                if res_status != "OK" or not fetch_data:
                    continue
                raw_bytes = None
                for part in fetch_data:
                    if isinstance(part, tuple) and len(part) >= 2:
                        raw_bytes = part[1]
                        break
                if not raw_bytes:
                    continue

                msg = email.message_from_bytes(raw_bytes)

                # Strict check: only accept messages belonging to this thread
                if not _is_thread_message(msg, self.thread_id):
                    continue

                msg_id_hdr = (msg.get("Message-ID") or "").strip()
                if msg_id_hdr and msg_id_hdr in seen_message_ids:
                    continue
                if msg_id_hdr:
                    seen_message_ids.add(msg_id_hdr)

                verified_messages.append((mid, msg))
            except Exception as e:
                logger.warning(f"Failed to fetch/parse message {mid}: {e}")
                continue

        return verified_messages

    def messages(self) -> List[Dict[str, Any]]:
        """Fetch every message in this thread from the IMAP mailbox, oldest first.

        Returns:
            List[Dict[str, Any]]: List of messages shaped like {"agent_name": ..., "content": ...}.
        """
        if self.thread_id is None:
            return []

        client = self._get_imap_client()
        verified = self._find_thread_messages(client)
        if not verified:
            return []

        parsed_entries = []
        for mid, msg in verified:
            # Parse date for chronological sorting
            date_str = msg.get("Date")
            dt = None
            if date_str:
                try:
                    dt = email.utils.parsedate_to_datetime(date_str)
                except Exception:
                    pass
            if dt is None:
                dt = datetime.now(timezone.utc)

            # Determine agent_name (from X-Swarms-Agent header or From header)
            agent_name = msg.get("X-Swarms-Agent")
            if not agent_name:
                from_hdr = msg.get("From", "")
                _, addr = email.utils.parseaddr(from_hdr)
                agent_name = from_hdr if from_hdr else addr

            # Extract plain text content and strip quotes
            content = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        payload = part.get_payload(decode=True)
                        charset = part.get_content_charset() or "utf-8"
                        content = payload.decode(charset, errors="replace")
                        break
            else:
                payload = msg.get_payload(decode=True)
                if payload:
                    charset = msg.get_content_charset() or "utf-8"
                    content = payload.decode(charset, errors="replace")
                else:
                    content = str(msg.get_payload() or "")

            content = strip_quoted_history(content)
            parsed_entries.append((dt, {"agent_name": str(agent_name), "content": content}))

        # Sort chronologically, oldest first
        parsed_entries.sort(key=lambda x: x[0])
        return [entry[1] for entry in parsed_entries]

    def clear(self) -> None:
        """Mark matched messages in this thread as deleted and expunge them."""
        if self.thread_id is None:
            return

        client = self._get_imap_client()
        verified = self._find_thread_messages(client)

        if verified:
            for mid, _ in verified:
                try:
                    client.store(mid, "+FLAGS", "\\Deleted")
                except Exception as e:
                    logger.warning(f"Failed to flag message {mid} for deletion: {e}")

            # Only expunge when verified thread messages were flagged.
            # Prefer UID EXPUNGE (RFC 4315) when supported to avoid expunging
            # other pre-existing deleted messages in the mailbox.
            try:
                expunged = False
                capabilities = getattr(client, "capabilities", ())
                if "UIDPLUS" in capabilities:
                    try:
                        uids = [
                            m.decode("ascii") if isinstance(m, bytes) else str(m)
                            for m, _ in verified
                        ]
                        client.uid("EXPUNGE", ",".join(uids))
                        expunged = True
                    except Exception:
                        pass
                if not expunged:
                    client.expunge()
            except Exception as e:
                logger.warning(f"Failed to expunge mailbox: {e}")

        self.thread_id = None
        self.last_message_id = None

    def close(self) -> None:
        """Log out of IMAP and close active connection."""
        if self._imap_client is not None:
            try:
                self._imap_client.logout()
            except Exception:
                pass
            self._imap_client = None
