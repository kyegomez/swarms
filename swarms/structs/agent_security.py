import base64
import json
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Union, Dict, List

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


@dataclass
class EncryptedMessage:
    """Structure for encrypted messages between agents"""

    sender_id: str
    receiver_id: str
    encrypted_content: bytes
    timestamp: float
    message_id: str
    session_id: str


class EncryptionSession:
    """Represents an encrypted communication session between agents"""

    def __init__(
        self,
        session_id: str,
        agent_ids: List[str],
        encrypted_keys: Dict[str, bytes],
        created_at: datetime,
    ):
        self.session_id = session_id
        self.agent_ids = agent_ids
        self.encrypted_keys = encrypted_keys
        self.created_at = created_at


class AgentEncryption:
    """
    Handles encryption for agent data both at rest and in transit.
    Supports both symmetric (for data at rest) and asymmetric (for data in transit) encryption.
    Also supports secure multi-agent communication.
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        encryption_key: Optional[str] = None,
        enable_transit_encryption: bool = False,
        enable_rest_encryption: bool = False,
        enable_multi_agent: bool = False,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.enable_transit_encryption = enable_transit_encryption
        self.enable_rest_encryption = enable_rest_encryption
        self.enable_multi_agent = enable_multi_agent

        # Multi-agent communication storage
        self.sessions: Dict[str, EncryptionSession] = {}
        self.known_agents: Dict[str, "AgentEncryption"] = {}

        if enable_rest_encryption:
            # Initialize encryption for data at rest
            if encryption_key:
                self.encryption_key = base64.urlsafe_b64encode(
                    PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=f"agent_{self.agent_id}".encode(),  # Unique salt per agent
                        iterations=100000,
                    ).derive(encryption_key.encode())
                )
            else:
                self.encryption_key = Fernet.generate_key()

            self.cipher_suite = Fernet(self.encryption_key)

        if enable_transit_encryption or enable_multi_agent:
            # Generate RSA key pair for transit encryption
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048
            )
            self.public_key = self.private_key.public_key()

    def register_agent(
        self, agent_id: str, agent_encryption: "AgentEncryption"
    ) -> None:
        """Register another agent for secure communication"""
        if not self.enable_multi_agent:
            raise ValueError("Multi-agent support is not enabled")
        self.known_agents[agent_id] = agent_encryption

    def create_session(self, agent_ids: List[str]) -> str:
        """Create a new encrypted session between multiple agents"""
        if not self.enable_multi_agent:
            raise ValueError("Multi-agent support is not enabled")

        session_id = str(uuid.uuid4())

        # Generate a shared session key
        session_key = Fernet.generate_key()

        # Create encrypted copies of the session key for each agent
        encrypted_keys = {}
        for agent_id in agent_ids:
            if (
                agent_id not in self.known_agents
                and agent_id != self.agent_id
            ):
                raise ValueError(f"Agent {agent_id} not registered")

            if agent_id == self.agent_id:
                agent_public_key = self.public_key
            else:
                agent_public_key = self.known_agents[
                    agent_id
                ].public_key

            encrypted_key = agent_public_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            encrypted_keys[agent_id] = encrypted_key

        # Store session information
        self.sessions[session_id] = EncryptionSession(
            session_id=session_id,
            agent_ids=agent_ids,
            encrypted_keys=encrypted_keys,
            created_at=datetime.now(),
        )

        return session_id

    def encrypt_message(
        self,
        content: Union[str, dict],
        receiver_id: str,
        session_id: str,
    ) -> EncryptedMessage:
        """Encrypt a message for another agent within a session"""
        if not self.enable_multi_agent:
            raise ValueError("Multi-agent support is not enabled")

        if session_id not in self.sessions:
            raise ValueError("Invalid session ID")

        session = self.sessions[session_id]
        if (
            self.agent_id not in session.agent_ids
            or receiver_id not in session.agent_ids
        ):
            raise ValueError("Sender or receiver not in session")

        # Serialize content if it's a dictionary
        if isinstance(content, dict):
            content = json.dumps(content)

        # Get the session key
        encrypted_session_key = session.encrypted_keys[self.agent_id]
        session_key = self.decrypt_session_key(encrypted_session_key)

        # Create Fernet cipher with session key
        cipher = Fernet(session_key)

        # Encrypt the message
        encrypted_content = cipher.encrypt(content.encode())

        return EncryptedMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            encrypted_content=encrypted_content,
            timestamp=datetime.now().timestamp(),
            message_id=str(uuid.uuid4()),
            session_id=session_id,
        )

    def decrypt_message(
        self, message: EncryptedMessage
    ) -> Union[str, dict]:
        """Decrypt a message from another agent"""
        if not self.enable_multi_agent:
            raise ValueError("Multi-agent support is not enabled")

        if message.session_id not in self.sessions:
            raise ValueError("Invalid session ID")

        if self.agent_id != message.receiver_id:
            raise ValueError("Message not intended for this agent")

        session = self.sessions[message.session_id]

        # Get the session key
        encrypted_session_key = session.encrypted_keys[self.agent_id]
        session_key = self.decrypt_session_key(encrypted_session_key)

        # Create Fernet cipher with session key
        cipher = Fernet(session_key)

        # Decrypt the message
        decrypted_content = cipher.decrypt(
            message.encrypted_content
        ).decode()

        # Try to parse as JSON
        try:
            return json.loads(decrypted_content)
        except json.JSONDecodeError:
            return decrypted_content

    def decrypt_session_key(self, encrypted_key: bytes) -> bytes:
        """Decrypt a session key using the agent's private key"""
        return self.private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    # Original methods preserved below
    def encrypt_at_rest(self, data: Union[str, dict, bytes]) -> bytes:
        """Encrypts data for storage"""
        if not self.enable_rest_encryption:
            return (
                data
                if isinstance(data, bytes)
                else str(data).encode()
            )

        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data.encode()

        return self.cipher_suite.encrypt(data)

    def decrypt_at_rest(
        self, encrypted_data: bytes
    ) -> Union[str, dict]:
        """Decrypts stored data"""
        if not self.enable_rest_encryption:
            return encrypted_data.decode()

        decrypted_data = self.cipher_suite.decrypt(encrypted_data)

        try:
            return json.loads(decrypted_data)
        except json.JSONDecodeError:
            return decrypted_data.decode()

    def encrypt_for_transit(self, data: Union[str, dict]) -> bytes:
        """Encrypts data for transmission"""
        if not self.enable_transit_encryption:
            return str(data).encode()

        if isinstance(data, dict):
            data = json.dumps(data)

        return self.public_key.encrypt(
            data.encode(),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt_from_transit(
        self, data: Union[bytes, str]
    ) -> Union[str, dict]:
        """Decrypts received data, handling both encrypted and unencrypted inputs"""
        if not self.enable_transit_encryption:
            return data.decode() if isinstance(data, bytes) else data

        try:
            if isinstance(data, bytes) and len(data) == 256:
                decrypted_data = self.private_key.decrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None,
                    ),
                ).decode()
            else:
                return (
                    data.decode() if isinstance(data, bytes) else data
                )

            try:
                return json.loads(decrypted_data)
            except json.JSONDecodeError:
                return decrypted_data
        except ValueError:
            return data.decode() if isinstance(data, bytes) else data

    def get_public_key_pem(self) -> bytes:
        """Returns the public key in PEM format for sharing"""
        if (
            not self.enable_transit_encryption
            and not self.enable_multi_agent
        ):
            return b""

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
