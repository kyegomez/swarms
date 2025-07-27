"""
Board of Directors Configuration Module

This module provides configuration management for the Board of Directors feature
in the Swarms Framework. It allows users to enable and configure the Board of
Directors feature manually through environment variables or configuration files.

The implementation follows the Swarms philosophy of:
- Readable code with comprehensive type annotations and documentation
- Performance optimization through caching and efficient loading
- Simplified abstractions for configuration management
"""

import os
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, Field
from loguru import logger


class BoardFeatureStatus(str, Enum):
    """Enumeration of Board of Directors feature status.
    
    This enum defines the possible states of the Board of Directors feature
    within the Swarms Framework.
    
    Attributes:
        ENABLED: Feature is explicitly enabled
        DISABLED: Feature is explicitly disabled
        AUTO: Feature state is determined automatically
    """
    
    ENABLED = "enabled"
    DISABLED = "disabled"
    AUTO = "auto"


class BoardConfigModel(BaseModel):
    """
    Configuration model for Board of Directors feature.
    
    This model defines all configurable parameters for the Board of Directors
    feature, including feature status, board composition, and operational settings.
    
    Attributes:
        board_feature_enabled: Whether the Board of Directors feature is enabled globally
        default_board_size: Default number of board members when creating a new board
        decision_threshold: Threshold for majority decisions (0.0-1.0)
        enable_voting: Enable voting mechanisms for board decisions
        enable_consensus: Enable consensus-building mechanisms
        default_board_model: Default model for board member agents
        verbose_logging: Enable verbose logging for board operations
        max_board_meeting_duration: Maximum duration for board meetings in seconds
        auto_fallback_to_director: Automatically fall back to Director mode if Board fails
        custom_board_templates: Custom board templates for different use cases
    """
    
    # Feature control
    board_feature_enabled: bool = Field(
        default=False,
        description="Whether the Board of Directors feature is enabled globally."
    )
    
    # Board composition
    default_board_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Default number of board members when creating a new board."
    )
    
    # Operational settings
    decision_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for majority decisions (0.0-1.0)."
    )
    
    enable_voting: bool = Field(
        default=True,
        description="Enable voting mechanisms for board decisions."
    )
    
    enable_consensus: bool = Field(
        default=True,
        description="Enable consensus-building mechanisms."
    )
    
    # Model settings
    default_board_model: str = Field(
        default="gpt-4o-mini",
        description="Default model for board member agents."
    )
    
    # Logging and monitoring
    verbose_logging: bool = Field(
        default=False,
        description="Enable verbose logging for board operations."
    )
    
    # Performance settings
    max_board_meeting_duration: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Maximum duration for board meetings in seconds."
    )
    
    # Integration settings
    auto_fallback_to_director: bool = Field(
        default=True,
        description="Automatically fall back to Director mode if Board fails."
    )
    
    # Custom board templates
    custom_board_templates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Custom board templates for different use cases."
    )


@dataclass
class BoardConfig:
    """
    Board of Directors configuration manager.
    
    This class manages the configuration for the Board of Directors feature,
    including loading from environment variables, configuration files, and
    providing default values.
    
    Attributes:
        config_file_path: Optional path to configuration file
        config_data: Optional configuration data dictionary
        config: The current configuration model instance
    """
    
    config_file_path: Optional[str] = None
    config_data: Optional[Dict[str, Any]] = None
    config: BoardConfigModel = field(init=False)
    
    def __post_init__(self) -> None:
        """Initialize the configuration after object creation."""
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from various sources.
        
        Priority order:
        1. Environment variables
        2. Configuration file
        3. Default values
        
        Raises:
            Exception: If configuration loading fails
        """
        try:
            # Start with default configuration
            self.config = BoardConfigModel()
            
            # Load from configuration file if specified
            if self.config_file_path and os.path.exists(self.config_file_path):
                self._load_from_file()
            
            # Override with environment variables
            self._load_from_environment()
            
            # Override with explicit config data
            if self.config_data:
                self._load_from_dict(self.config_data)
                
        except Exception as e:
            logger.error(f"Failed to load Board of Directors configuration: {str(e)}")
            raise
    
    def _load_from_file(self) -> None:
        """
        Load configuration from file.
        
        Raises:
            Exception: If file loading fails
        """
        try:
            import yaml
            with open(self.config_file_path, 'r') as f:
                file_config = yaml.safe_load(f)
                self._load_from_dict(file_config)
                logger.info(f"Loaded Board of Directors config from: {self.config_file_path}")
        except Exception as e:
            logger.warning(f"Failed to load config file {self.config_file_path}: {e}")
            raise
    
    def _load_from_environment(self) -> None:
        """
        Load configuration from environment variables.
        
        This method maps environment variables to configuration parameters
        and handles type conversion appropriately.
        """
        env_mappings = {
            'SWARMS_BOARD_FEATURE_ENABLED': 'board_feature_enabled',
            'SWARMS_BOARD_DEFAULT_SIZE': 'default_board_size',
            'SWARMS_BOARD_DECISION_THRESHOLD': 'decision_threshold',
            'SWARMS_BOARD_ENABLE_VOTING': 'enable_voting',
            'SWARMS_BOARD_ENABLE_CONSENSUS': 'enable_consensus',
            'SWARMS_BOARD_DEFAULT_MODEL': 'default_board_model',
            'SWARMS_BOARD_VERBOSE_LOGGING': 'verbose_logging',
            'SWARMS_BOARD_MAX_MEETING_DURATION': 'max_board_meeting_duration',
            'SWARMS_BOARD_AUTO_FALLBACK': 'auto_fallback_to_director',
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Convert string values to appropriate types
                    if config_key in ['board_feature_enabled', 'enable_voting', 'enable_consensus', 'verbose_logging', 'auto_fallback_to_director']:
                        converted_value = value.lower() in ['true', '1', 'yes', 'on']
                    elif config_key in ['default_board_size', 'max_board_meeting_duration']:
                        converted_value = int(value)
                    elif config_key in ['decision_threshold']:
                        converted_value = float(value)
                    else:
                        converted_value = value
                    
                    setattr(self.config, config_key, converted_value)
                    logger.debug(f"Loaded {config_key} from environment: {converted_value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to parse environment variable {env_var}: {e}")
    
    def _load_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Load configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Raises:
            ValueError: If configuration values are invalid
        """
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                try:
                    setattr(self.config, key, value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to set config {key}: {e}")
                    raise ValueError(f"Invalid configuration value for {key}: {e}")
    
    def is_enabled(self) -> bool:
        """
        Check if the Board of Directors feature is enabled.
        
        Returns:
            bool: True if the feature is enabled, False otherwise
        """
        return self.config.board_feature_enabled
    
    def get_config(self) -> BoardConfigModel:
        """
        Get the current configuration.
        
        Returns:
            BoardConfigModel: The current configuration
        """
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Raises:
            ValueError: If any update values are invalid
        """
        try:
            self._load_from_dict(updates)
        except ValueError as e:
            logger.error(f"Failed to update configuration: {e}")
            raise
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        Save the current configuration to a file.
        
        Args:
            file_path: Optional file path to save to (uses config_file_path if not provided)
            
        Raises:
            Exception: If saving fails
        """
        save_path = file_path or self.config_file_path
        if not save_path:
            logger.warning("No file path specified for saving configuration")
            return
        
        try:
            import yaml
            # Convert config to dictionary
            config_dict = self.config.model_dump()
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved Board of Directors config to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise
    
    @lru_cache(maxsize=128)
    def get_default_board_template(self, template_name: str = "standard") -> Dict[str, Any]:
        """
        Get a default board template.
        
        This method provides predefined board templates for common use cases.
        Templates are cached for improved performance.
        
        Args:
            template_name: Name of the template to retrieve
            
        Returns:
            Dict[str, Any]: Board template configuration
        """
        templates = {
            "standard": {
                "roles": [
                    {"name": "Chairman", "weight": 1.5, "expertise": ["leadership", "strategy"]},
                    {"name": "Vice-Chairman", "weight": 1.2, "expertise": ["operations", "coordination"]},
                    {"name": "Secretary", "weight": 1.0, "expertise": ["documentation", "communication"]},
                ]
            },
            "executive": {
                "roles": [
                    {"name": "CEO", "weight": 2.0, "expertise": ["executive_leadership", "strategy"]},
                    {"name": "CFO", "weight": 1.5, "expertise": ["finance", "risk_management"]},
                    {"name": "CTO", "weight": 1.5, "expertise": ["technology", "innovation"]},
                    {"name": "COO", "weight": 1.3, "expertise": ["operations", "efficiency"]},
                ]
            },
            "advisory": {
                "roles": [
                    {"name": "Lead_Advisor", "weight": 1.3, "expertise": ["strategy", "consulting"]},
                    {"name": "Technical_Advisor", "weight": 1.2, "expertise": ["technology", "architecture"]},
                    {"name": "Business_Advisor", "weight": 1.2, "expertise": ["business", "market_analysis"]},
                    {"name": "Legal_Advisor", "weight": 1.1, "expertise": ["legal", "compliance"]},
                ]
            },
            "minimal": {
                "roles": [
                    {"name": "Chairman", "weight": 1.0, "expertise": ["leadership"]},
                    {"name": "Member", "weight": 1.0, "expertise": ["general"]},
                ]
            }
        }
        
        # Check custom templates first
        if template_name in self.config.custom_board_templates:
            return self.config.custom_board_templates[template_name]
        
        # Return standard template if requested template not found
        return templates.get(template_name, templates["standard"])
    
    def validate_config(self) -> List[str]:
        """
        Validate the current configuration.
        
        This method performs comprehensive validation of the configuration
        to ensure all values are within acceptable ranges and constraints.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Validate the configuration model
            self.config.model_validate(self.config.model_dump())
        except Exception as e:
            errors.append(f"Configuration validation failed: {e}")
        
        # Additional custom validations
        if self.config.decision_threshold < 0.5:
            errors.append("Decision threshold should be at least 0.5 for meaningful majority decisions")
        
        if self.config.default_board_size < 2:
            errors.append("Board size should be at least 2 for meaningful discussions")
        
        if self.config.max_board_meeting_duration < 60:
            errors.append("Board meeting duration should be at least 60 seconds")
        
        return errors


# Global configuration instance
_board_config: Optional[BoardConfig] = None


@lru_cache(maxsize=1)
def get_board_config(config_file_path: Optional[str] = None) -> BoardConfig:
    """
    Get the global Board of Directors configuration instance.
    
    This function provides a singleton pattern for accessing the Board of Directors
    configuration. The configuration is cached for improved performance.
    
    Args:
        config_file_path: Optional path to configuration file
        
    Returns:
        BoardConfig: The global configuration instance
    """
    global _board_config
    
    if _board_config is None:
        _board_config = BoardConfig(config_file_path=config_file_path)
    
    return _board_config


def enable_board_feature(config_file_path: Optional[str] = None) -> None:
    """
    Enable the Board of Directors feature globally.
    
    This function enables the Board of Directors feature and saves the configuration
    to the specified file path.
    
    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"board_feature_enabled": True})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info("Board of Directors feature enabled")


def disable_board_feature(config_file_path: Optional[str] = None) -> None:
    """
    Disable the Board of Directors feature globally.
    
    This function disables the Board of Directors feature and saves the configuration
    to the specified file path.
    
    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"board_feature_enabled": False})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info("Board of Directors feature disabled")


def is_board_feature_enabled(config_file_path: Optional[str] = None) -> bool:
    """
    Check if the Board of Directors feature is enabled.
    
    Args:
        config_file_path: Optional path to configuration file
        
    Returns:
        bool: True if the feature is enabled, False otherwise
    """
    config = get_board_config(config_file_path)
    return config.is_enabled()


def create_default_config_file(file_path: str = "swarms_board_config.yaml") -> None:
    """
    Create a default configuration file.
    
    This function creates a default Board of Directors configuration file
    with recommended settings.
    
    Args:
        file_path: Path where to create the configuration file
    """
    default_config = {
        "board_feature_enabled": False,
        "default_board_size": 3,
        "decision_threshold": 0.6,
        "enable_voting": True,
        "enable_consensus": True,
        "default_board_model": "gpt-4o-mini",
        "verbose_logging": False,
        "max_board_meeting_duration": 300,
        "auto_fallback_to_director": True,
        "custom_board_templates": {}
    }
    
    config = BoardConfig(config_file_path=file_path, config_data=default_config)
    config.save_config(file_path)
    
    logger.info(f"Created default Board of Directors config file: {file_path}")


def set_board_size(size: int, config_file_path: Optional[str] = None) -> None:
    """
    Set the default board size.
    
    Args:
        size: The default board size (1-10)
        config_file_path: Optional path to save the configuration
    """
    if not 1 <= size <= 10:
        raise ValueError("Board size must be between 1 and 10")
    
    config = get_board_config(config_file_path)
    config.update_config({"default_board_size": size})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info(f"Default board size set to: {size}")


def set_decision_threshold(threshold: float, config_file_path: Optional[str] = None) -> None:
    """
    Set the decision threshold for majority decisions.
    
    Args:
        threshold: The decision threshold (0.0-1.0)
        config_file_path: Optional path to save the configuration
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0.0 and 1.0")
    
    config = get_board_config(config_file_path)
    config.update_config({"decision_threshold": threshold})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info(f"Decision threshold set to: {threshold}")


def set_board_model(model: str, config_file_path: Optional[str] = None) -> None:
    """
    Set the default board model.
    
    Args:
        model: The default model name for board members
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"default_board_model": model})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info(f"Default board model set to: {model}")


def enable_verbose_logging(config_file_path: Optional[str] = None) -> None:
    """
    Enable verbose logging for board operations.
    
    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"verbose_logging": True})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info("Verbose logging enabled for Board of Directors")


def disable_verbose_logging(config_file_path: Optional[str] = None) -> None:
    """
    Disable verbose logging for board operations.
    
    Args:
        config_file_path: Optional path to save the configuration
    """
    config = get_board_config(config_file_path)
    config.update_config({"verbose_logging": False})
    
    if config_file_path:
        config.save_config(config_file_path)
    
    logger.info("Verbose logging disabled for Board of Directors")