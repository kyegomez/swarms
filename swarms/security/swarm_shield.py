"""
SwarmShield integration for Swarms framework.

This module provides enterprise-grade security for swarm communications,
including encryption, conversation management, and audit capabilities.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import (
    Cipher,
    algorithms,
    modes,
)
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from loguru import logger

from swarms.utils.loguru_logger import initialize_logger

# Initialize logger for security module
security_logger = initialize_logger(log_folder="security")


class EncryptionStrength(Enum):
    """Encryption strength levels for SwarmShield"""
    
    STANDARD = "standard"  # AES-256
    ENHANCED = "enhanced"  # AES-256 + SHA-512
    MAXIMUM = "maximum"  # AES-256 + SHA-512 + HMAC


class SwarmShield:
    """
    SwarmShield: Advanced security system for swarm agents
    
    Features:
    - Multi-layer message encryption
    - Secure conversation storage
    - Automatic key rotation
    - Message integrity verification
    - Integration with Swarms framework
    """

    def __init__(
        self,
        encryption_strength: EncryptionStrength = EncryptionStrength.MAXIMUM,
        key_rotation_interval: int = 3600,  # 1 hour
        storage_path: Optional[str] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize SwarmShield with security settings
        
        Args:
            encryption_strength: Level of encryption to use
            key_rotation_interval: Key rotation interval in seconds
            storage_path: Path for encrypted storage
            enable_logging: Enable security logging
        """
        self.encryption_strength = encryption_strength
        self.key_rotation_interval = key_rotation_interval
        self.enable_logging = enable_logging
        
        # Set storage path within swarms framework
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path("swarms_security_storage")

        # Initialize storage and locks
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._conv_lock = threading.Lock()
        self._conversations: Dict[str, List[Dict]] = {}

        # Initialize security components
        self._initialize_security()
        self._load_conversations()

        if self.enable_logging:
            # Handle encryption_strength safely (could be enum or string)
            encryption_str = (
                encryption_strength.value 
                if hasattr(encryption_strength, 'value') 
                else str(encryption_strength)
            )
            security_logger.info(
                f"SwarmShield initialized with {encryption_str} encryption"
            )

    def _initialize_security(self) -> None:
        """Set up encryption keys and components"""
        try:
            # Generate master key and salt
            self.master_key = secrets.token_bytes(32)
            self.salt = os.urandom(16)

            # Initialize key derivation
            self.kdf = PBKDF2HMAC(
                algorithm=hashes.SHA512(),
                length=32,
                salt=self.salt,
                iterations=600000,
                backend=default_backend(),
            )

            # Generate initial keys
            self._rotate_keys()
            self.hmac_key = secrets.token_bytes(32)

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Security initialization failed: {e}")
            raise

    def _rotate_keys(self) -> None:
        """Perform security key rotation"""
        try:
            self.encryption_key = self.kdf.derive(self.master_key)
            self.iv = os.urandom(16)
            self.last_rotation = time.time()
            if self.enable_logging:
                security_logger.debug("Security keys rotated successfully")
        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Key rotation failed: {e}")
            raise

    def _check_rotation(self) -> None:
        """Check and perform key rotation if needed"""
        if (
            time.time() - self.last_rotation
            >= self.key_rotation_interval
        ):
            self._rotate_keys()

    def _save_conversation(self, conversation_id: str) -> None:
        """Save conversation to encrypted storage"""
        try:
            if conversation_id not in self._conversations:
                return

            # Encrypt conversation data
            json_data = json.dumps(
                self._conversations[conversation_id]
            ).encode()
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(self.iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            encrypted_data = (
                encryptor.update(json_data) + encryptor.finalize()
            )
            
            # Combine encrypted data with authentication tag
            combined_data = encrypted_data + encryptor.tag

            # Save atomically using temporary file
            conv_path = self.storage_path / f"{conversation_id}.conv"
            temp_path = conv_path.with_suffix(".tmp")

            with open(temp_path, "wb") as f:
                f.write(combined_data)
            temp_path.replace(conv_path)

            if self.enable_logging:
                security_logger.debug(f"Saved conversation {conversation_id}")

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to save conversation: {e}")
            raise

    def _load_conversations(self) -> None:
        """Load existing conversations from storage"""
        try:
            for file_path in self.storage_path.glob("*.conv"):
                try:
                    with open(file_path, "rb") as f:
                        combined_data = f.read()
                    conversation_id = file_path.stem

                    # Split combined data into encrypted data and authentication tag
                    if len(combined_data) < 16:  # Minimum size for GCM tag
                        continue
                    
                    encrypted_data = combined_data[:-16]
                    auth_tag = combined_data[-16:]

                    # Decrypt conversation data
                    cipher = Cipher(
                        algorithms.AES(self.encryption_key),
                        modes.GCM(self.iv, auth_tag),
                        backend=default_backend(),
                    )
                    decryptor = cipher.decryptor()
                    json_data = (
                        decryptor.update(encrypted_data)
                        + decryptor.finalize()
                    )

                    self._conversations[conversation_id] = json.loads(
                        json_data
                    )
                    if self.enable_logging:
                        security_logger.debug(
                            f"Loaded conversation {conversation_id}"
                        )

                except Exception as e:
                    if self.enable_logging:
                        security_logger.error(
                            f"Failed to load conversation {file_path}: {e}"
                        )
                    continue

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to load conversations: {e}")
            raise

    def protect_message(self, agent_name: str, message: str) -> str:
        """
        Encrypt a message with multiple security layers
        
        Args:
            agent_name: Name of the sending agent
            message: Message to encrypt
            
        Returns:
            Encrypted message string
        """
        try:
            self._check_rotation()

            # Validate inputs
            if not isinstance(agent_name, str) or not isinstance(
                message, str
            ):
                raise ValueError(
                    "Agent name and message must be strings"
                )
            if not agent_name.strip() or not message.strip():
                raise ValueError(
                    "Agent name and message cannot be empty"
                )

            # Generate message ID and timestamp
            message_id = secrets.token_hex(16)
            timestamp = datetime.now(timezone.utc).isoformat()

            # Encrypt message content
            message_bytes = message.encode()
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(self.iv),
                backend=default_backend(),
            )
            encryptor = cipher.encryptor()
            ciphertext = (
                encryptor.update(message_bytes) + encryptor.finalize()
            )

            # Calculate message hash
            message_hash = hashlib.sha512(message_bytes).hexdigest()

            # Generate HMAC if maximum security
            hmac_signature = None
            if self.encryption_strength == EncryptionStrength.MAXIMUM:
                h = hmac.new(
                    self.hmac_key, ciphertext, hashlib.sha512
                )
                hmac_signature = h.digest()

            # Create secure package
            secure_package = {
                "id": message_id,
                "time": timestamp,
                "agent": agent_name,
                "cipher": base64.b64encode(ciphertext).decode(),
                "tag": base64.b64encode(encryptor.tag).decode(),
                "hash": message_hash,
                "hmac": (
                    base64.b64encode(hmac_signature).decode()
                    if hmac_signature
                    else None
                ),
            }

            return base64.b64encode(
                json.dumps(secure_package).encode()
            ).decode()

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to protect message: {e}")
            raise

    def retrieve_message(self, encrypted_str: str) -> Tuple[str, str]:
        """
        Decrypt and verify a message
        
        Args:
            encrypted_str: Encrypted message string
            
        Returns:
            Tuple of (agent_name, message)
        """
        try:
            # Decode secure package
            secure_package = json.loads(
                base64.b64decode(encrypted_str)
            )

            # Get components
            ciphertext = base64.b64decode(secure_package["cipher"])
            tag = base64.b64decode(secure_package["tag"])

            # Verify HMAC if present
            if secure_package["hmac"]:
                hmac_signature = base64.b64decode(
                    secure_package["hmac"]
                )
                h = hmac.new(
                    self.hmac_key, ciphertext, hashlib.sha512
                )
                if not hmac.compare_digest(
                    hmac_signature, h.digest()
                ):
                    raise ValueError("HMAC verification failed")

            # Decrypt message
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(self.iv, tag),
                backend=default_backend(),
            )
            decryptor = cipher.decryptor()
            decrypted_data = (
                decryptor.update(ciphertext) + decryptor.finalize()
            )

            # Verify hash
            if (
                hashlib.sha512(decrypted_data).hexdigest()
                != secure_package["hash"]
            ):
                raise ValueError("Message hash verification failed")

            return secure_package["agent"], decrypted_data.decode()

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to retrieve message: {e}")
            raise

    def create_conversation(self, name: str = "") -> str:
        """Create a new secure conversation"""
        conversation_id = str(uuid.uuid4())
        with self._conv_lock:
            self._conversations[conversation_id] = {
                "id": conversation_id,
                "name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "messages": [],
            }
            self._save_conversation(conversation_id)
        if self.enable_logging:
            security_logger.info(f"Created conversation {conversation_id}")
        return conversation_id

    def add_message(
        self, conversation_id: str, agent_name: str, message: str
    ) -> None:
        """
        Add an encrypted message to a conversation
        
        Args:
            conversation_id: Target conversation ID
            agent_name: Name of the sending agent
            message: Message content
        """
        try:
            # Encrypt message
            encrypted = self.protect_message(agent_name, message)

            # Add to conversation
            with self._conv_lock:
                if conversation_id not in self._conversations:
                    raise ValueError(
                        f"Invalid conversation ID: {conversation_id}"
                    )

                self._conversations[conversation_id][
                    "messages"
                ].append(
                    {
                        "timestamp": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "data": encrypted,
                    }
                )

                # Save changes
                self._save_conversation(conversation_id)

            if self.enable_logging:
                security_logger.info(
                    f"Added message to conversation {conversation_id}"
                )

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to add message: {e}")
            raise

    def get_messages(
        self, conversation_id: str
    ) -> List[Tuple[str, str, datetime]]:
        """
        Get decrypted messages from a conversation
        
        Args:
            conversation_id: Target conversation ID
            
        Returns:
            List of (agent_name, message, timestamp) tuples
        """
        try:
            with self._conv_lock:
                if conversation_id not in self._conversations:
                    raise ValueError(
                        f"Invalid conversation ID: {conversation_id}"
                    )

                history = []
                for msg in self._conversations[conversation_id][
                    "messages"
                ]:
                    agent_name, message = self.retrieve_message(
                        msg["data"]
                    )
                    timestamp = datetime.fromisoformat(
                        msg["timestamp"]
                    )
                    history.append((agent_name, message, timestamp))

                return history

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to get messages: {e}")
            raise

    def get_conversation_summary(self, conversation_id: str) -> Dict:
        """
        Get summary statistics for a conversation
        
        Args:
            conversation_id: Target conversation ID
            
        Returns:
            Dictionary with conversation statistics
        """
        try:
            with self._conv_lock:
                if conversation_id not in self._conversations:
                    raise ValueError(
                        f"Invalid conversation ID: {conversation_id}"
                    )

                conv = self._conversations[conversation_id]
                messages = conv["messages"]
                
                # Get unique agents
                agents = set()
                for msg in messages:
                    agent_name, _ = self.retrieve_message(msg["data"])
                    agents.add(agent_name)

                return {
                    "id": conversation_id,
                    "name": conv["name"],
                    "created_at": conv["created_at"],
                    "message_count": len(messages),
                    "agents": list(agents),
                    "last_message": (
                        messages[-1]["timestamp"] if messages else None
                    ),
                }

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to get summary: {e}")
            raise

    def delete_conversation(self, conversation_id: str) -> None:
        """
        Delete a conversation and its storage
        
        Args:
            conversation_id: Target conversation ID
        """
        try:
            with self._conv_lock:
                if conversation_id not in self._conversations:
                    raise ValueError(
                        f"Invalid conversation ID: {conversation_id}"
                    )

                # Remove from memory
                del self._conversations[conversation_id]

                # Remove from storage
                conv_path = self.storage_path / f"{conversation_id}.conv"
                if conv_path.exists():
                    conv_path.unlink()

            if self.enable_logging:
                security_logger.info(f"Deleted conversation {conversation_id}")

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to delete conversation: {e}")
            raise

    def export_conversation(
        self, conversation_id: str, format: str = "json", path: str = None
    ) -> str:
        """
        Export a conversation to a file
        
        Args:
            conversation_id: Target conversation ID
            format: Export format (json, txt)
            path: Output file path
            
        Returns:
            Path to exported file
        """
        try:
            messages = self.get_messages(conversation_id)
            summary = self.get_conversation_summary(conversation_id)

            if not path:
                path = f"conversation_{conversation_id}.{format}"

            if format.lower() == "json":
                export_data = {
                    "summary": summary,
                    "messages": [
                        {
                            "agent": agent,
                            "message": message,
                            "timestamp": timestamp.isoformat(),
                        }
                        for agent, message, timestamp in messages
                    ],
                }
                with open(path, "w") as f:
                    json.dump(export_data, f, indent=2)

            elif format.lower() == "txt":
                with open(path, "w") as f:
                    f.write(f"Conversation: {summary['name']}\n")
                    f.write(f"Created: {summary['created_at']}\n")
                    f.write(f"Messages: {summary['message_count']}\n")
                    f.write(f"Agents: {', '.join(summary['agents'])}\n\n")
                    
                    for agent, message, timestamp in messages:
                        f.write(f"[{timestamp}] {agent}: {message}\n")

            else:
                raise ValueError(f"Unsupported format: {format}")

            if self.enable_logging:
                security_logger.info(f"Exported conversation to {path}")

            return path

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to export conversation: {e}")
            raise

    def backup_conversations(self, backup_path: str = None) -> str:
        """
        Create a backup of all conversations
        
        Args:
            backup_path: Backup directory path
            
        Returns:
            Path to backup directory
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"swarm_shield_backup_{timestamp}"

            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)

            # Export all conversations
            for conversation_id in self._conversations:
                self.export_conversation(
                    conversation_id,
                    format="json",
                    path=str(backup_dir / f"{conversation_id}.json"),
                )

            if self.enable_logging:
                security_logger.info(f"Backup created at {backup_path}")

            return backup_path

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to create backup: {e}")
            raise

    def get_agent_stats(self, agent_name: str) -> Dict:
        """
        Get statistics for a specific agent
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Dictionary with agent statistics
        """
        try:
            total_messages = 0
            conversations = set()

            for conversation_id, conv in self._conversations.items():
                for msg in conv["messages"]:
                    msg_agent, _ = self.retrieve_message(msg["data"])
                    if msg_agent == agent_name:
                        total_messages += 1
                        conversations.add(conversation_id)

            return {
                "agent_name": agent_name,
                "total_messages": total_messages,
                "conversations": len(conversations),
                "conversation_ids": list(conversations),
            }

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to get agent stats: {e}")
            raise

    def query_conversations(
        self,
        agent_name: str = None,
        text: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Search conversations with filters
        
        Args:
            agent_name: Filter by agent name
            text: Search for text in messages
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum results to return
            
        Returns:
            List of matching conversation summaries
        """
        try:
            results = []

            for conversation_id, conv in self._conversations.items():
                # Check date range
                conv_date = datetime.fromisoformat(conv["created_at"])
                if start_date and conv_date < start_date:
                    continue
                if end_date and conv_date > end_date:
                    continue

                # Check agent filter
                if agent_name:
                    conv_agents = set()
                    for msg in conv["messages"]:
                        msg_agent, _ = self.retrieve_message(msg["data"])
                        conv_agents.add(msg_agent)
                    if agent_name not in conv_agents:
                        continue

                # Check text filter
                if text:
                    text_found = False
                    for msg in conv["messages"]:
                        _, message = self.retrieve_message(msg["data"])
                        if text.lower() in message.lower():
                            text_found = True
                            break
                    if not text_found:
                        continue

                # Add to results
                summary = self.get_conversation_summary(conversation_id)
                results.append(summary)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            if self.enable_logging:
                security_logger.error(f"Failed to query conversations: {e}")
            raise 