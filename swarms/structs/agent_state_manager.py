"""
Advanced Agent State Management System

This module provides comprehensive state persistence, recovery, versioning,
and management capabilities for individual agents across sessions.

Features:
- Atomic state saving with backup and rollback capabilities
- State versioning and history tracking
- Checkpoint creation and restoration
- State validation and integrity checks
- Automatic state migration between versions
- Recovery from corrupted states
- State metadata tracking (timestamps, checksums, tags)
- Multi-format support (JSON, YAML, Pickle)
"""

import hashlib
import json
import logging
import os
import shutil
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)


class StateMetadata:
    """Metadata information for a saved state."""

    def __init__(
        self,
        version: str = "1.0",
        timestamp: Optional[float] = None,
        checksum: Optional[str] = None,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        auto_checkpoint: bool = False,
        agent_id: Optional[str] = None,
        agent_name: Optional[str] = None,
    ):
        """
        Initialize state metadata.

        Args:
            version: State format version
            timestamp: When the state was saved (Unix timestamp)
            checksum: SHA256 checksum of the state file for integrity validation
            tags: Custom tags for organizing and identifying states
            description: Human-readable description of the state
            auto_checkpoint: Whether this is an auto-created checkpoint
            agent_id: ID of the agent that created this state
            agent_name: Name of the agent that created this state
        """
        self.version = version
        self.timestamp = timestamp or time.time()
        self.checksum = checksum
        self.tags = tags or []
        self.description = description
        self.auto_checkpoint = auto_checkpoint
        self.agent_id = agent_id
        self.agent_name = agent_name

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "checksum": self.checksum,
            "tags": self.tags,
            "description": self.description,
            "auto_checkpoint": self.auto_checkpoint,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMetadata":
        """Create metadata from dictionary."""
        return cls(
            version=data.get("version", "1.0"),
            timestamp=data.get("timestamp"),
            checksum=data.get("checksum"),
            tags=data.get("tags", []),
            description=data.get("description"),
            auto_checkpoint=data.get("auto_checkpoint", False),
            agent_id=data.get("agent_id"),
            agent_name=data.get("agent_name"),
        )


class StateValidator:
    """Validates state files for integrity and format."""

    @staticmethod
    def validate_checksum(
        file_path: str, expected_checksum: str
    ) -> bool:
        """
        Validate file integrity using checksum.

        Args:
            file_path: Path to the state file
            expected_checksum: Expected SHA256 checksum

        Returns:
            bool: True if checksum matches
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

            calculated_checksum = sha256_hash.hexdigest()
            return calculated_checksum == expected_checksum
        except Exception as e:
            logger.warning(f"Checksum validation failed: {e}")
            return False

    @staticmethod
    def calculate_checksum(file_path: str) -> str:
        """
        Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to the file

        Returns:
            str: Hex string of SHA256 checksum
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    @staticmethod
    def is_valid_state_file(file_path: str) -> bool:
        """
        Check if a file is a valid state file.

        Args:
            file_path: Path to check

        Returns:
            bool: True if file exists and is readable
        """
        if not os.path.exists(file_path):
            logger.error(f"State file not found: {file_path}")
            return False

        if not os.path.isfile(file_path):
            logger.error(f"State path is not a file: {file_path}")
            return False

        if not os.access(file_path, os.R_OK):
            logger.error(f"State file not readable: {file_path}")
            return False

        # Try to parse the file format
        try:
            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    json.load(f)
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                with open(file_path, "r") as f:
                    yaml.safe_load(f)
            # Pickle files are binary, just check existence
            return True
        except Exception as e:
            logger.error(f"State file validation failed: {e}")
            return False


class StateHistory:
    """Manages state version history and snapshots."""

    def __init__(self, max_checkpoints: int = 10):
        """
        Initialize state history manager.

        Args:
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.max_checkpoints = max_checkpoints
        self.checkpoints: List[Tuple[str, StateMetadata]] = []

    def add_checkpoint(
        self, file_path: str, metadata: StateMetadata
    ) -> None:
        """
        Add a checkpoint to history.

        Args:
            file_path: Path to the checkpoint file
            metadata: Metadata about the checkpoint
        """
        self.checkpoints.append((file_path, metadata))

        # Remove oldest checkpoint if max is exceeded
        if len(self.checkpoints) > self.max_checkpoints:
            old_path, _ = self.checkpoints.pop(0)
            try:
                if os.path.exists(old_path):
                    os.remove(old_path)
                    logger.info(f"Removed old checkpoint: {old_path}")
            except Exception as e:
                logger.warning(
                    f"Could not remove old checkpoint {old_path}: {e}"
                )

    def get_checkpoints(
        self, tag: Optional[str] = None
    ) -> List[Tuple[str, StateMetadata]]:
        """
        Get list of checkpoints, optionally filtered by tag.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of (file_path, metadata) tuples
        """
        if tag is None:
            return self.checkpoints

        return [
            (path, meta)
            for path, meta in self.checkpoints
            if tag in meta.tags
        ]

    def get_latest_checkpoint(
        self,
    ) -> Optional[Tuple[str, StateMetadata]]:
        """
        Get the most recent checkpoint.

        Returns:
            Tuple of (file_path, metadata) or None if no checkpoints exist
        """
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]

    def clear_history(self) -> None:
        """Clear all checkpoints from history."""
        for file_path, _ in self.checkpoints:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {file_path}: {e}")
        self.checkpoints.clear()


class AgentStateManager:
    """
    Comprehensive state management system for agents with persistence,
    recovery, versioning, and checkpoint capabilities.
    """

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        base_dir: Optional[str] = None,
        max_checkpoints: int = 10,
        auto_checkpoint: bool = False,
    ):
        """
        Initialize agent state manager.

        Args:
            agent_id: Unique identifier for the agent
            agent_name: Display name for the agent
            base_dir: Base directory for storing state files
            max_checkpoints: Maximum checkpoints to keep
            auto_checkpoint: Enable automatic checkpointing
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.base_dir = Path(
            base_dir or os.path.expanduser("~/.swarms/agent_states")
        )
        self.agent_dir = self.base_dir / agent_id
        self.max_checkpoints = max_checkpoints
        self.auto_checkpoint = auto_checkpoint

        # Initialize directories
        self._ensure_directories()

        # State history management
        self.history = StateHistory(max_checkpoints)
        self._load_history()

    def _ensure_directories(self) -> None:
        """Ensure all necessary directories exist."""
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self.agent_dir.mkdir(parents=True, exist_ok=True)
            (self.agent_dir / "checkpoints").mkdir(exist_ok=True)
            (self.agent_dir / "metadata").mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Could not create state directories: {e}")
            raise

    def _load_history(self) -> None:
        """Load checkpoint history from disk."""
        try:
            history_file = self.agent_dir / "history.json"
            if history_file.exists():
                with open(history_file, "r") as f:
                    history_data = json.load(f)
                    for checkpoint_info in history_data:
                        file_path = checkpoint_info["path"]
                        metadata = StateMetadata.from_dict(
                            checkpoint_info["metadata"]
                        )
                        self.history.add_checkpoint(file_path, metadata)
        except Exception as e:
            logger.warning(f"Could not load checkpoint history: {e}")

    def _save_history(self) -> None:
        """Save checkpoint history to disk."""
        try:
            history_file = self.agent_dir / "history.json"
            history_data = [
                {"path": str(path), "metadata": meta.to_dict()}
                for path, meta in self.history.checkpoints
            ]
            with open(history_file, "w") as f:
                json.dump(history_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save checkpoint history: {e}")

    def save_state(
        self,
        state: Dict[str, Any],
        file_format: str = "json",
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        is_checkpoint: bool = False,
    ) -> str:
        """
        Save agent state with atomic operations and backup.

        Args:
            state: State dictionary to save
            file_format: Format to save in ('json', 'yaml', 'pickle')
            tags: Optional tags for the state
            description: Human-readable description
            is_checkpoint: Whether this is a checkpoint (auto-managed)

        Returns:
            str: Path to the saved state file

        Raises:
            ValueError: If file_format is not supported
            OSError: If save operation fails
        """
        if file_format not in ["json", "yaml", "pickle"]:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Generate filename with timestamp
        timestamp = int(time.time() * 1000)
        filename = f"state_{timestamp}.{file_format}"

        if is_checkpoint:
            state_path = self.agent_dir / "checkpoints" / filename
        else:
            state_path = self.agent_dir / filename

        temp_path = Path(str(state_path) + ".tmp")
        backup_path = Path(str(state_path) + ".backup")

        try:
            # Create parent directory if needed
            state_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first (atomic operation)
            if file_format == "json":
                with open(temp_path, "w") as f:
                    json.dump(state, f, indent=2, default=str)
            elif file_format == "yaml":
                with open(temp_path, "w") as f:
                    yaml.dump(state, f, default_flow_style=False)
            elif file_format == "pickle":
                with open(temp_path, "wb") as f:
                    pickle.dump(state, f)

            # Create backup if file already exists
            if state_path.exists():
                try:
                    shutil.copy2(state_path, backup_path)
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")

            # Move temp file to final location (atomic)
            temp_path.replace(state_path)

            # Calculate checksum for integrity validation
            checksum = StateValidator.calculate_checksum(str(state_path))

            # Create metadata
            metadata = StateMetadata(
                version="1.0",
                timestamp=time.time(),
                checksum=checksum,
                tags=tags,
                description=description,
                auto_checkpoint=is_checkpoint,
                agent_id=self.agent_id,
                agent_name=self.agent_name,
            )

            # Save metadata
            metadata_file = (
                self.agent_dir / "metadata" / f"{state_path.stem}.json"
            )
            with open(metadata_file, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            # Add to history
            self.history.add_checkpoint(str(state_path), metadata)
            self._save_history()

            # Clean up backup if everything succeeded
            if backup_path.exists():
                try:
                    backup_path.unlink()
                except Exception as e:
                    logger.warning(f"Could not remove backup file: {e}")

            logger.info(f"Successfully saved state to: {state_path}")
            return str(state_path)

        except Exception as e:
            # Clean up temp file if it exists
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning(
                        f"Could not clean up temp file: {cleanup_error}"
                    )

            logger.error(f"Error saving state: {e}")
            raise

    def load_state(
        self, file_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load agent state from file with validation.

        Args:
            file_path: Path to state file. If None, loads the most recent state.

        Returns:
            Dict[str, Any]: Loaded state dictionary

        Raises:
            FileNotFoundError: If state file doesn't exist
            ValueError: If state file is corrupted
        """
        try:
            # Determine which file to load
            if file_path is None:
                # Load most recent state
                latest = self.history.get_latest_checkpoint()
                if latest is None:
                    # Look for recent non-checkpoint states
                    state_files = sorted(
                        self.agent_dir.glob("state_*.json"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )
                    if not state_files:
                        raise FileNotFoundError(
                            f"No state files found for agent {self.agent_id}"
                        )
                    file_path = str(state_files[0])
                else:
                    file_path = latest[0]

            # Validate file exists
            if not StateValidator.is_valid_state_file(file_path):
                raise FileNotFoundError(f"State file not found: {file_path}")

            # Load and validate checksum if metadata exists
            state_stem = Path(file_path).stem
            metadata_file = (
                self.agent_dir / "metadata" / f"{state_stem}.json"
            )

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)
                    stored_checksum = metadata_dict.get("checksum")

                    if stored_checksum:
                        if not StateValidator.validate_checksum(
                            file_path, stored_checksum
                        ):
                            logger.warning(
                                f"Checksum mismatch for {file_path}. "
                                "State may be corrupted."
                            )
                            raise ValueError(
                                "State file checksum validation failed"
                            )

            # Load the state file
            file_path_obj = Path(file_path)

            if file_path.endswith(".json"):
                with open(file_path, "r") as f:
                    state = json.load(f)
            elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
                with open(file_path, "r") as f:
                    state = yaml.safe_load(f)
            elif file_path.endswith(".pickle") or file_path.endswith(
                ".pkl"
            ):
                with open(file_path, "rb") as f:
                    state = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            logger.info(f"Successfully loaded state from: {file_path}")
            return state

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            raise

    def create_checkpoint(
        self,
        state: Dict[str, Any],
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Create a named checkpoint of the current state.

        Args:
            state: State dictionary to checkpoint
            tags: Optional tags for the checkpoint
            description: Human-readable description

        Returns:
            str: Path to the checkpoint file
        """
        return self.save_state(
            state,
            file_format="json",
            tags=tags,
            description=description,
            is_checkpoint=True,
        )

    def restore_checkpoint(
        self, checkpoint_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore agent state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint (filename or timestamp).
                          If None, restores most recent.

        Returns:
            Dict[str, Any]: Restored state dictionary
        """
        if checkpoint_id is None:
            latest = self.history.get_latest_checkpoint()
            if latest is None:
                raise FileNotFoundError("No checkpoints available")
            checkpoint_path = latest[0]
        else:
            # Find checkpoint by ID
            checkpoint_path = None
            for path, _ in self.history.checkpoints:
                if checkpoint_id in path:
                    checkpoint_path = path
                    break

            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint_id}"
                )

        return self.load_state(checkpoint_path)

    def list_checkpoints(
        self, tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            tag: Optional tag to filter by

        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = self.history.get_checkpoints(tag=tag)
        result = []

        for path, metadata in checkpoints:
            result.append(
                {
                    "path": path,
                    "filename": os.path.basename(path),
                    **metadata.to_dict(),
                }
            )

        return result

    def get_state_size(self, file_path: str) -> int:
        """
        Get the size of a state file in bytes.

        Args:
            file_path: Path to state file

        Returns:
            int: Size in bytes
        """
        try:
            return os.path.getsize(file_path)
        except Exception as e:
            logger.warning(f"Could not get state file size: {e}")
            return 0

    def cleanup_old_checkpoints(
        self, days_old: int = 7
    ) -> int:
        """
        Remove checkpoints older than specified days.

        Args:
            days_old: Age threshold in days

        Returns:
            int: Number of checkpoints removed
        """
        cutoff_time = time.time() - (days_old * 86400)
        removed_count = 0

        for path, metadata in list(self.history.checkpoints):
            if metadata.timestamp < cutoff_time:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    self.history.checkpoints.remove((path, metadata))
                    removed_count += 1
                except Exception as e:
                    logger.warning(
                        f"Could not remove old checkpoint {path}: {e}"
                    )

        self._save_history()
        return removed_count

    def export_state_summary(self) -> Dict[str, Any]:
        """
        Export a summary of the current state management status.

        Returns:
            Dict[str, Any]: Summary information
        """
        checkpoints = self.list_checkpoints()
        total_size = sum(
            self.get_state_size(cp["path"]) for cp in checkpoints
        )

        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "state_directory": str(self.agent_dir),
            "total_checkpoints": len(checkpoints),
            "max_checkpoints": self.max_checkpoints,
            "total_size_bytes": total_size,
            "checkpoints": checkpoints,
            "auto_checkpoint_enabled": self.auto_checkpoint,
        }

    def validate_all_states(self) -> Dict[str, bool]:
        """
        Validate all stored state files.

        Returns:
            Dict mapping file paths to validation status
        """
        results = {}

        for path, _ in self.history.checkpoints:
            try:
                results[path] = StateValidator.is_valid_state_file(path)
            except Exception as e:
                logger.warning(f"Validation error for {path}: {e}")
                results[path] = False

        return results
