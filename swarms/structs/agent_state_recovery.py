"""
State Recovery and Migration System for Agent State Management

This module provides capabilities for:
- Recovery from corrupted or partial states
- Migration between different state format versions
- State health checks and diagnostics
- Automatic rollback capabilities
- State transformation and upgrade utilities
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


class StateRecovery:
    """Handles recovery from corrupted, incomplete, or damaged states."""

    @staticmethod
    def find_recoverable_states(
        agent_dir: Path,
    ) -> List[Tuple[str, Path]]:
        """
        Find potentially recoverable state files in the agent directory.

        Args:
            agent_dir: Directory containing agent state files

        Returns:
            List of (type, path) tuples where type is 'backup', 'checkpoint', or 'state'
        """
        recoverable = []

        # Check for backup files
        backup_files = list(agent_dir.glob("**/*.backup"))
        for backup_file in backup_files:
            recoverable.append(("backup", backup_file))

        # Check for checkpoint files
        checkpoint_dir = agent_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("state_*.json"))
            for cp_file in checkpoint_files:
                recoverable.append(("checkpoint", cp_file))

        # Check for state files
        state_files = list(agent_dir.glob("state_*.json"))
        for state_file in state_files:
            recoverable.append(("state", state_file))

        # Sort by modification time (newest first)
        recoverable.sort(
            key=lambda x: x[1].stat().st_mtime, reverse=True
        )

        return recoverable

    @staticmethod
    def attempt_recovery(
        agent_dir: Path, max_attempts: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to recover state from the most recent available file.

        Args:
            agent_dir: Directory containing agent state files
            max_attempts: Maximum number of files to try loading

        Returns:
            Recovered state dictionary or None if recovery fails
        """
        recoverable = StateRecovery.find_recoverable_states(agent_dir)

        for attempt, (file_type, file_path) in enumerate(
            recoverable[:max_attempts]
        ):
            try:
                logger.info(
                    f"Recovery attempt {attempt + 1}: "
                    f"Loading {file_type} from {file_path}"
                )

                with open(file_path, "r") as f:
                    state = json.load(f)

                logger.info(
                    f"Successfully recovered state from {file_path}"
                )
                return state

            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse {file_type} {file_path}: {e}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to load {file_type} {file_path}: {e}"
                )

        logger.error("Could not recover state from any available files")
        return None

    @staticmethod
    def validate_state_structure(
        state: Dict[str, Any], required_keys: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate the structure of a loaded state.

        Args:
            state: State dictionary to validate
            required_keys: List of keys that must be present

        Returns:
            Tuple of (is_valid, missing_keys)
        """
        if not isinstance(state, dict):
            return False, ["State is not a dictionary"]

        if not state:
            return False, ["State is empty"]

        missing_keys = []
        if required_keys:
            for key in required_keys:
                if key not in state:
                    missing_keys.append(key)

        is_valid = len(missing_keys) == 0

        return is_valid, missing_keys

    @staticmethod
    def repair_state(
        state: Dict[str, Any],
        repair_rules: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Attempt to repair a damaged state using repair rules.

        Args:
            state: State dictionary to repair
            repair_rules: Dictionary of field name -> repair function

        Returns:
            Repaired state dictionary
        """
        if not repair_rules:
            return state

        repaired = state.copy()

        for field, repair_func in repair_rules.items():
            try:
                if field in repaired:
                    repaired[field] = repair_func(repaired[field])
                else:
                    logger.debug(f"Field {field} not found for repair")

            except Exception as e:
                logger.warning(f"Could not repair field {field}: {e}")

        return repaired


class StateMigration:
    """Handles migration of state between different versions and formats."""

    # Version migration functions
    _migrations: Dict[Tuple[str, str], Callable] = {}

    @classmethod
    def register_migration(
        cls, from_version: str, to_version: str
    ) -> Callable:
        """
        Register a migration function.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            cls._migrations[(from_version, to_version)] = func
            return func

        return decorator

    @classmethod
    def migrate_state(
        cls,
        state: Dict[str, Any],
        from_version: str,
        to_version: str,
    ) -> Dict[str, Any]:
        """
        Migrate state from one version to another.

        Args:
            state: State to migrate
            from_version: Current version
            to_version: Target version

        Returns:
            Migrated state dictionary

        Raises:
            ValueError: If no migration path exists
        """
        if from_version == to_version:
            return state

        # Find migration path
        migration_path = cls._find_migration_path(from_version, to_version)

        if not migration_path:
            raise ValueError(
                f"No migration path from {from_version} to {to_version}"
            )

        # Apply migrations in sequence
        current_state = state.copy()
        current_version = from_version

        for next_version in migration_path[1:]:
            migration_func = cls._migrations.get(
                (current_version, next_version)
            )

            if migration_func:
                try:
                    current_state = migration_func(current_state)
                    logger.info(
                        f"Migrated state from {current_version} "
                        f"to {next_version}"
                    )
                except Exception as e:
                    logger.error(
                        f"Migration from {current_version} "
                        f"to {next_version} failed: {e}"
                    )
                    raise

            current_version = next_version

        return current_state

    @classmethod
    def _find_migration_path(
        cls, from_version: str, to_version: str
    ) -> Optional[List[str]]:
        """
        Find a path between two versions using BFS.

        Args:
            from_version: Source version
            to_version: Target version

        Returns:
            List of versions forming the path, or None if no path exists
        """
        from collections import deque

        # Build graph
        graph: Dict[str, List[str]] = {}
        for from_v, to_v in cls._migrations.keys():
            if from_v not in graph:
                graph[from_v] = []
            graph[from_v].append(to_v)

        # BFS
        queue = deque([(from_version, [from_version])])
        visited = {from_version}

        while queue:
            current, path = queue.popleft()

            if current == to_version:
                return path

            for neighbor in graph.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None


class StateTransformation:
    """Utilities for transforming and filtering state data."""

    @staticmethod
    def extract_subset(
        state: Dict[str, Any], keys: List[str]
    ) -> Dict[str, Any]:
        """
        Extract a subset of the state containing only specified keys.

        Args:
            state: Full state dictionary
            keys: List of keys to extract

        Returns:
            New dictionary containing only specified keys
        """
        return {k: state[k] for k in keys if k in state}

    @staticmethod
    def filter_sensitive_data(
        state: Dict[str, Any],
        sensitive_keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a copy of state with sensitive data removed or masked.

        Args:
            state: State dictionary
            sensitive_keys: Keys to consider sensitive

        Returns:
            Filtered state dictionary
        """
        if sensitive_keys is None:
            sensitive_keys = ["api_key", "token", "password", "secret"]

        filtered = {}

        for key, value in state.items():
            if any(
                sensitive in key.lower() for sensitive in sensitive_keys
            ):
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value

        return filtered

    @staticmethod
    def transform_values(
        state: Dict[str, Any],
        transformers: Dict[str, Callable],
    ) -> Dict[str, Any]:
        """
        Apply transformation functions to specific state values.

        Args:
            state: State dictionary
            transformers: Dict mapping keys to transformation functions

        Returns:
            Transformed state dictionary
        """
        transformed = state.copy()

        for key, transformer in transformers.items():
            if key in transformed:
                try:
                    transformed[key] = transformer(transformed[key])
                except Exception as e:
                    logger.warning(
                        f"Could not transform {key}: {e}"
                    )

        return transformed

    @staticmethod
    def convert_format(
        state_dict: Dict[str, Any],
        from_format: str,
        to_format: str,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Convert state between different file formats.

        Args:
            state_dict: State dictionary
            from_format: Source format ('json', 'yaml', etc.)
            to_format: Target format ('json', 'yaml', etc.)
            file_path: Optional path to save converted state

        Returns:
            str: Path to converted state file

        Raises:
            ValueError: If format is not supported
        """
        if to_format == "json":
            content = json.dumps(state_dict, indent=2, default=str)
            extension = ".json"
        elif to_format == "yaml":
            content = yaml.dump(state_dict, default_flow_style=False)
            extension = ".yaml"
        else:
            raise ValueError(f"Unsupported format: {to_format}")

        if not file_path:
            file_path = f"state_converted{extension}"

        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(
            f"Converted state from {from_format} to {to_format}: {file_path}"
        )
        return file_path


class StateDiagnostics:
    """Provides diagnostic information about state files and health."""

    @staticmethod
    def get_state_statistics(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about a state dictionary.

        Args:
            state: State dictionary

        Returns:
            Dictionary of statistics
        """
        def count_nested(obj, depth=0):
            if depth > 100:  # Prevent infinite recursion
                return 0
            if isinstance(obj, dict):
                return sum(count_nested(v, depth + 1) for v in obj.values())
            elif isinstance(obj, (list, tuple)):
                return sum(
                    count_nested(v, depth + 1) for v in obj
                )
            else:
                return 1

        def estimate_size(obj):
            """Rough estimation of object size in bytes."""
            try:
                return len(json.dumps(obj, default=str).encode("utf-8"))
            except:
                return 0

        return {
            "total_keys": len(state),
            "total_items": count_nested(state),
            "estimated_size_bytes": estimate_size(state),
            "types_present": list(set(type(v).__name__ for v in state.values())),
            "nested_depth": StateDiagnostics._calculate_depth(state),
        }

    @staticmethod
    def _calculate_depth(obj: Any, max_depth: int = 100) -> int:
        """Calculate maximum nesting depth."""
        if max_depth <= 0:
            return 0

        if isinstance(obj, dict):
            if not obj:
                return 1
            return 1 + max(StateDiagnostics._calculate_depth(v, max_depth - 1) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            if not obj:
                return 1
            return 1 + max(
                StateDiagnostics._calculate_depth(v, max_depth - 1)
                for v in obj
            )
        else:
            return 0

    @staticmethod
    def check_state_health(
        state: Dict[str, Any],
        validators: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Check the health and validity of a state.

        Args:
            state: State dictionary to check
            validators: Optional custom validation functions

        Returns:
            Dictionary with health check results
        """
        health = {
            "is_valid": True,
            "is_empty": len(state) == 0,
            "errors": [],
            "warnings": [],
            "statistics": StateDiagnostics.get_state_statistics(state),
        }

        # Check for empty state
        if health["is_empty"]:
            health["warnings"].append("State is empty")
            health["is_valid"] = False

        # Check for large state
        stats = health["statistics"]
        if stats["estimated_size_bytes"] > 100 * 1024 * 1024:  # 100MB
            health["warnings"].append(
                f"State is very large: "
                f"{stats['estimated_size_bytes'] / 1024 / 1024:.2f}MB"
            )

        # Run custom validators
        if validators:
            for validator_name, validator_func in validators.items():
                try:
                    result = validator_func(state)
                    if not result:
                        health["errors"].append(
                            f"Validator '{validator_name}' failed"
                        )
                        health["is_valid"] = False
                except Exception as e:
                    health["errors"].append(
                        f"Validator '{validator_name}' raised exception: {e}"
                    )
                    health["is_valid"] = False

        return health
