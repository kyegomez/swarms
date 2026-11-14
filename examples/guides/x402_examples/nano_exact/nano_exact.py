"""Accept nano:mainnet exact payments in front of a Swarms agent.

Standard library only. The seller holds no key and no seed: it issues
a 402 offer, then reads the payer's send block from a Nano node the
operator configures and decides whether that block pays that offer.

Each offer is one order. The payer binds the send block to the order
by setting the block's representative to a commitment of the order id,
the pay-to account and the exact raw price (xno-binding v1, option C),
so a block paid for one request cannot be presented for another. Every
credited block is appended and fsynced to a claims file that is read
back on start, so a replayed block is refused after a restart too.

Amounts are integer raw, 1 XNO is 10**30 raw. Floats are refused.
"""

import base64
import hashlib
import ipaddress
import json
import os
import re
import secrets
import time
import urllib.parse
import urllib.request

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

RAW_PER_XNO = 10**30
NETWORK = "nano:mainnet"
SCHEME = "exact"

MATCHED = "matched"
ALREADY_CLAIMED = "already_claimed"
UNKNOWN_ORDER = "unknown_order"
WRONG_RESOURCE = "wrong_resource"
WRONG_PAY_TO = "wrong_pay_to"
WRONG_AMOUNT = "wrong_amount"
NOT_BOUND = "not_bound_to_order"
NOT_CONFIRMED = "not_confirmed"
NOT_A_SEND = "not_a_send"
LATE = "paid_after_expiry"
INVALID_PAYMENT = "invalid_payment"
INCONCLUSIVE = "inconclusive"

_ALPHABET = "13456789abcdefghijkmnopqrstuwxyz"
_ACCOUNT_RE = re.compile(
    r"\A(?:nano|xrb)_[13][13456789abcdefghijkmnopqrstuwxyz]{59}\Z"
)
_HASH_RE = re.compile(r"\A[0-9A-Fa-f]{64}\Z")
_RAW_RE = re.compile(r"\A[0-9]{1,40}\Z")


def _b32_encode(value, length):
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 31])
        value >>= 5
    return "".join(reversed(out))


def _b32_decode(text):
    value = 0
    for ch in text:
        value = value * 32 + _ALPHABET.index(ch)
    return value


def account_public_key(account):
    """Return the 32 byte public key of a Nano account.

    Raises:
        ValueError: if the account is malformed or its checksum fails.
    """
    if not isinstance(account, str) or not _ACCOUNT_RE.match(account):
        raise ValueError("not a Nano account")
    body = account.split("_", 1)[1]
    key_value = _b32_decode(body[:52])
    if key_value >> 256:
        raise ValueError("not a Nano account")
    key = key_value.to_bytes(32, "big")
    check = hashlib.blake2b(key, digest_size=5).digest()[::-1]
    if _b32_decode(body[52:]).to_bytes(5, "big") != check:
        raise ValueError("Nano account checksum mismatch")
    return key


def is_valid_account(account):
    """Return True when the account is well formed with a valid checksum."""
    try:
        account_public_key(account)
    except ValueError:
        return False
    return True


def public_key_to_account(key):
    """Encode a 32 byte public key as a nano_ account."""
    if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
        raise ValueError("a public key is 32 bytes")
    check = hashlib.blake2b(bytes(key), digest_size=5).digest()[::-1]
    return (
        "nano_"
        + _b32_encode(int.from_bytes(key, "big"), 52)
        + _b32_encode(int.from_bytes(check, "big"), 8)
    )


def _require_raw(price_raw):
    if (
        isinstance(price_raw, bool)
        or not isinstance(price_raw, int)
        or price_raw <= 0
    ):
        raise TypeError("price_raw must be a positive int of raw")


def commitment_rep_account(order_id, pay_to, price_raw):
    """Return the representative the payer sets for this order."""
    _require_raw(price_raw)
    msg = "|".join(
        ["xno-binding/v1/rep", order_id, pay_to, str(price_raw)]
    )
    digest = hashlib.sha256(msg.encode()).digest()
    return public_key_to_account(digest)


def _same_account(a, b):
    try:
        return account_public_key(a) == account_public_key(b)
    except ValueError:
        return False


def check_send(info, pay_to, price_raw, order_id):
    """Decide whether a block_info reply pays one order exactly.

    Args:
        info: the node's block_info reply, read with json_block true.
        pay_to: the seller account of the order.
        price_raw: the exact raw amount of the order.
        order_id: the order the block must be bound to.

    Returns:
        MATCHED, or the verdict naming the first check that failed.
    """
    contents = info.get("contents") or {}
    if info.get("subtype") != "send":
        return NOT_A_SEND
    if info.get("confirmed") != "true":
        return NOT_CONFIRMED
    if not _same_account(contents.get("link_as_account"), pay_to):
        return WRONG_PAY_TO
    amount = info.get("amount")
    if not isinstance(amount, str) or not _RAW_RE.match(amount):
        return WRONG_AMOUNT
    if int(amount) != price_raw:
        return WRONG_AMOUNT
    rep = commitment_rep_account(order_id, pay_to, price_raw)
    if not _same_account(contents.get("representative"), rep):
        return NOT_BOUND
    return MATCHED


class NanoNode:
    """Reads blocks from a Nano node RPC URL set by the operator.

    Plain http is accepted only for a loopback host. Replies are
    bounded in time and in size.
    """

    def __init__(self, url, timeout=10, max_bytes=65536):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme == "http":
            try:
                loopback = ipaddress.ip_address(
                    parts.hostname or ""
                ).is_loopback
            except ValueError:
                loopback = parts.hostname == "localhost"
            if not loopback:
                raise ValueError(
                    "plain http only for a loopback node"
                )
        elif parts.scheme != "https":
            raise ValueError("node URL must be https")
        self.url = url
        self.timeout = timeout
        self.max_bytes = max_bytes

    def read_block(self, block_hash):
        """Return the block_info reply for a block hash."""
        body = json.dumps(
            {
                "action": "block_info",
                "json_block": "true",
                "hash": block_hash,
            }
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(
            req, timeout=self.timeout
        ) as resp:
            data = resp.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError("node reply too large")
        return json.loads(data)


class _JsonLines:
    """An append only JSON lines file, locked while it is used."""

    def __init__(self, path):
        self.path = path

    def locked(self, fn):
        fd = os.open(
            self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600
        )
        with os.fdopen(fd, "r+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            content = fh.read()
            complete = content[: content.rfind("\n") + 1]
            if len(complete) != len(content):
                # A torn tail is a write cut short before it was used.
                fh.truncate(len(complete.encode("utf-8")))
            rows = [
                json.loads(line)
                for line in complete.split("\n")
                if line.strip()
            ]
            return fn(rows, fh)

    @staticmethod
    def append(fh, row):
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class NanoExactSeller:
    """Issues nano:mainnet exact offers and verifies their payments.

    Args:
        pay_to: the seller's nano_ account, checksum validated.
        price_raw: the price of one request in integer raw.
        node: an object with read_block(hash), such as NanoNode.
        state_dir: where the orders and claims files are kept.
        ttl_seconds: how long an offer can be paid.
    """

    def __init__(
        self, pay_to, price_raw, node, state_dir, ttl_seconds=600
    ):
        account_public_key(pay_to)
        _require_raw(price_raw)
        os.makedirs(state_dir, exist_ok=True)
        self.pay_to = pay_to
        self.price_raw = price_raw
        self.node = node
        self.ttl_seconds = ttl_seconds
        self.orders = _JsonLines(
            os.path.join(state_dir, "orders.jsonl")
        )
        self.claims = _JsonLines(
            os.path.join(state_dir, "claims.jsonl")
        )

    def offer(self, resource, description=""):
        """Create one order for a resource and return the 402 body."""
        order_id = secrets.token_hex(16)
        expires_at = int(time.time()) + self.ttl_seconds
        row = {
            "order": order_id,
            "resource": resource,
            "pay_to": self.pay_to,
            "price_raw": str(self.price_raw),
            "expires_at": expires_at,
        }
        self.orders.locked(
            lambda rows, fh: _JsonLines.append(fh, row)
        )
        rep = commitment_rep_account(
            order_id, self.pay_to, self.price_raw
        )
        return {
            "x402Version": 1,
            "error": "payment required",
            "accepts": [
                {
                    "scheme": SCHEME,
                    "network": NETWORK,
                    "maxAmountRequired": str(self.price_raw),
                    "resource": resource,
                    "description": description,
                    "payTo": self.pay_to,
                    "asset": "XNO",
                    "maxTimeoutSeconds": self.ttl_seconds,
                    "extra": {
                        "orderId": order_id,
                        "representative": rep,
                        "expiresAt": expires_at,
                    },
                }
            ],
        }

    def _order(self, order_id):
        found = self.orders.locked(
            lambda rows, fh: [
                r for r in rows if r["order"] == order_id
            ]
        )
        return found[0] if found else None

    def verify(self, header, resource):
        """Check an X-PAYMENT header for a resource and claim its block.

        Args:
            header: base64 JSON with payload blockHash and orderId.
            resource: the resource of the request being paid for.

        Returns:
            MATCHED when the request may be served, else a verdict.
        """
        try:
            payment = json.loads(
                base64.b64decode(header, validate=True)
            )
            payload = payment["payload"]
            block_hash = payload["blockHash"]
            order_id = payload["orderId"]
        except (ValueError, KeyError, TypeError):
            return INVALID_PAYMENT
        if (
            payment.get("scheme") != SCHEME
            or payment.get("network") != NETWORK
            or not isinstance(block_hash, str)
            or not _HASH_RE.match(block_hash)
            or not isinstance(order_id, str)
        ):
            return INVALID_PAYMENT
        order = self._order(order_id)
        if order is None:
            return UNKNOWN_ORDER
        if order["resource"] != resource:
            return WRONG_RESOURCE
        try:
            info = self.node.read_block(block_hash.upper())
        except (OSError, ValueError):
            return INCONCLUSIVE
        if not isinstance(info, dict):
            return INCONCLUSIVE
        price_raw = int(order["price_raw"])
        verdict = check_send(
            info, order["pay_to"], price_raw, order_id
        )
        if verdict != MATCHED:
            return verdict
        stamp = info.get("local_timestamp")
        if not isinstance(stamp, str) or not stamp.isdigit():
            return INCONCLUSIVE
        if int(stamp) == 0:
            return INCONCLUSIVE
        if int(stamp) > order["expires_at"]:
            return LATE
        return self.claims.locked(
            lambda rows, fh: self._claim(
                rows, fh, block_hash, order_id
            )
        )

    @staticmethod
    def _claim(rows, fh, block_hash, order_id):
        block = block_hash.upper()
        for row in rows:
            if row["block"] == block or row["order"] == order_id:
                return ALREADY_CLAIMED
        _JsonLines.append(fh, {"block": block, "order": order_id})
        return MATCHED
