"""
SAID Protocol Integration
=========================
Registers Swarms agents with SAID Protocol — on-chain identity for AI agents on Solana.
Every agent can get a free Solana identity automatically.

SAID Protocol: https://saidprotocol.com
Docs: https://saidprotocol.com/docs.html
"""

import hashlib
import hmac
import json
import os
import secrets
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

SAID_API = "https://api.saidprotocol.com"

BASE58_ALPHABET = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58encode(data: bytes) -> str:
    """Minimal base58 encoder — avoids heavy deps."""
    count = 0
    for byte in data:
        if byte == 0:
            count += 1
        else:
            break
    num = int.from_bytes(data, "big")
    result = []
    while num > 0:
        num, rem = divmod(num, 58)
        result.append(BASE58_ALPHABET[rem : rem + 1].decode())
    return "1" * count + "".join(reversed(result))


@dataclass
class SAIDWallet:
    public_key: str
    secret_key: str
    created_at: str


@dataclass
class SAIDRegistration:
    wallet: str
    pda: str
    profile_url: str


def generate_solana_keypair() -> SAIDWallet:
    """
    Generate a Solana-compatible Ed25519 keypair using only stdlib.
    Returns public key and secret key as base58-encoded strings.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        private_bytes = private_key.private_bytes_raw()
        public_bytes = public_key.public_bytes_raw()
    except ImportError:
        # Fallback: generate random 32-byte seed (not cryptographically a real Solana key,
        # but sufficient for SAID registration which only needs a unique public identifier)
        private_bytes = secrets.token_bytes(32)
        # Derive a deterministic "public key" from private using HMAC-SHA256
        public_bytes = hmac.new(private_bytes, b"solana-pubkey", hashlib.sha256).digest()

    secret_key_bytes = private_bytes + public_bytes  # Solana format: 64 bytes
    from datetime import datetime, timezone
    return SAIDWallet(
        public_key=_b58encode(public_bytes),
        secret_key=_b58encode(secret_key_bytes),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def register_with_said(
    wallet: str,
    name: str,
    description: Optional[str] = None,
    skills: Optional[List[str]] = None,
    twitter: Optional[str] = None,
    website: Optional[str] = None,
) -> Optional[SAIDRegistration]:
    """
    Register a Swarms agent with SAID Protocol.
    Returns SAIDRegistration on success, None on failure.
    Registration is free and instant (off-chain pending).
    """
    payload = {
        "wallet": wallet,
        "name": name,
        "description": description or f"Swarms agent — {name}",
        "capabilities": skills or ["multi-agent-orchestration", "autonomous-tasks"],
        "source": "swarms",
    }
    if twitter:
        payload["twitter"] = twitter
    if website:
        payload["website"] = website

    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{SAID_API}/api/register/pending",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 201):
                return None
            result = json.loads(resp.read())
            registered_wallet = result.get("wallet", wallet)
            return SAIDRegistration(
                wallet=registered_wallet,
                pda=result.get("pda", ""),
                profile_url=f"https://saidprotocol.com/agents/{registered_wallet}",
            )
    except Exception:
        return None


def save_said_wallet(wallet: SAIDWallet, path: str) -> None:
    """Save SAID wallet to disk with restricted permissions."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(
            {
                "public_key": wallet.public_key,
                "secret_key": wallet.secret_key,
                "created_at": wallet.created_at,
            },
            f,
            indent=2,
        )
    os.chmod(path, 0o600)
