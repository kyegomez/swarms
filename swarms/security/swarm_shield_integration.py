"""
SwarmShield integration for Swarms framework.

This module provides the main integration point for all security features,
combining input validation, output filtering, safety checking, rate limiting,
and encryption into a unified security shield.
"""

from typing import Dict, Any, Optional, Tuple, Union, List
from datetime import datetime

from loguru import logger

from swarms.utils.loguru_logger import initialize_logger
from .shield_config import ShieldConfig
from .swarm_shield import SwarmShield, EncryptionStrength
from .input_validator import InputValidator
from .output_filter import OutputFilter
from .safety_checker import SafetyChecker
from .rate_limiter import RateLimiter

# Initialize logger for shield integration
shield_logger = initialize_logger(log_folder="shield_integration")


class SwarmShieldIntegration:
    """
    Main SwarmShield integration class
    
    This class provides a unified interface for all security features:
    - Input validation and sanitization
    - Output filtering and sensitive data protection
    - Safety checking and ethical AI compliance
    - Rate limiting and abuse prevention
    - Encrypted communication and storage
    """
    
    def __init__(self, config: Optional[ShieldConfig] = None):
        """
        Initialize SwarmShield integration
        
        Args:
            config: Shield configuration. If None, uses default configuration.
        """
        self.config = config or ShieldConfig()
        
        # Initialize security components
        self._initialize_components()
        
        # Handle security_level safely (could be enum or string)
        security_level_str = (
            self.config.security_level.value 
            if hasattr(self.config.security_level, 'value') 
            else str(self.config.security_level)
        )
        shield_logger.info(f"SwarmShield integration initialized with {security_level_str} security")
    
    def _initialize_components(self) -> None:
        """Initialize all security components"""
        try:
            # Initialize SwarmShield for encryption
            if self.config.integrate_with_conversation:
                self.swarm_shield = SwarmShield(
                    **self.config.get_encryption_config(),
                    enable_logging=self.config.enable_audit_logging
                )
            else:
                self.swarm_shield = None
            
            # Initialize input validator
            self.input_validator = InputValidator(
                self.config.get_validation_config()
            )
            
            # Initialize output filter
            self.output_filter = OutputFilter(
                self.config.get_filtering_config()
            )
            
            # Initialize safety checker
            self.safety_checker = SafetyChecker(
                self.config.get_safety_config()
            )
            
            # Initialize rate limiter
            self.rate_limiter = RateLimiter(
                self.config.get_rate_limit_config()
            )
            
        except Exception as e:
            shield_logger.error(f"Failed to initialize security components: {e}")
            raise
    
    def validate_and_protect_input(self, input_data: str, agent_name: str, input_type: str = "text") -> Tuple[bool, str, Optional[str]]:
        """
        Validate and protect input data
        
        Args:
            input_data: Input data to validate
            agent_name: Name of the agent
            input_type: Type of input
            
        Returns:
            Tuple of (is_valid, protected_data, error_message)
        """
        try:
            # Check rate limits
            is_allowed, error = self.rate_limiter.check_agent_limit(agent_name)
            if not is_allowed:
                return False, "", error
            
            # Validate input
            is_valid, sanitized, error = self.input_validator.validate_input(input_data, input_type)
            if not is_valid:
                return False, "", error
            
            # Check safety
            is_safe, safe_content, error = self.safety_checker.check_safety(sanitized, input_type)
            if not is_safe:
                return False, "", error
            
            # Protect with encryption if enabled
            if self.swarm_shield and self.config.protect_agent_communications:
                try:
                    protected = self.swarm_shield.protect_message(agent_name, safe_content)
                    return True, protected, None
                except Exception as e:
                    shield_logger.error(f"Encryption failed: {e}")
                    return False, "", f"Encryption error: {str(e)}"
            
            return True, safe_content, None
            
        except Exception as e:
            shield_logger.error(f"Input validation error: {e}")
            return False, "", f"Validation error: {str(e)}"
    
    def filter_and_protect_output(self, output_data: Union[str, Dict, List], agent_name: str, output_type: str = "text") -> Tuple[bool, Union[str, Dict, List], Optional[str]]:
        """
        Filter and protect output data
        
        Args:
            output_data: Output data to filter
            agent_name: Name of the agent
            output_type: Type of output
            
        Returns:
            Tuple of (is_safe, filtered_data, warning_message)
        """
        try:
            # Filter output
            is_safe, filtered, warning = self.output_filter.filter_output(output_data, output_type)
            if not is_safe:
                return False, "", "Output contains unsafe content"
            
            # Check safety
            if isinstance(filtered, str):
                is_safe, safe_content, error = self.safety_checker.check_safety(filtered, output_type)
                if not is_safe:
                    return False, "", error
                filtered = safe_content
            
            # Track request
            self.rate_limiter.track_request(agent_name, f"output_{output_type}")
            
            return True, filtered, warning
            
        except Exception as e:
            shield_logger.error(f"Output filtering error: {e}")
            return False, "", f"Filtering error: {str(e)}"
    
    def process_agent_communication(self, agent_name: str, message: str, direction: str = "outbound") -> Tuple[bool, str, Optional[str]]:
        """
        Process agent communication (inbound or outbound)
        
        Args:
            agent_name: Name of the agent
            message: Message content
            direction: "inbound" or "outbound"
            
        Returns:
            Tuple of (is_valid, processed_message, error_message)
        """
        try:
            if direction == "inbound":
                return self.validate_and_protect_input(message, agent_name, "text")
            elif direction == "outbound":
                return self.filter_and_protect_output(message, agent_name, "text")
            else:
                return False, "", f"Invalid direction: {direction}"
                
        except Exception as e:
            shield_logger.error(f"Communication processing error: {e}")
            return False, "", f"Processing error: {str(e)}"
    
    def validate_task(self, task: str, agent_name: str) -> Tuple[bool, str, Optional[str]]:
        """Validate swarm task"""
        return self.validate_and_protect_input(task, agent_name, "task")
    
    def validate_agent_config(self, config: Dict[str, Any], agent_name: str) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Validate agent configuration"""
        try:
            # Validate config structure
            is_valid, validated_config, error = self.input_validator.validate_config(config)
            if not is_valid:
                return False, {}, error
            
            # Check safety
            is_safe, safe_config, error = self.safety_checker.check_config_safety(validated_config)
            if not is_safe:
                return False, {}, error
            
            # Filter sensitive data
            is_safe, filtered_config, warning = self.output_filter.filter_config_output(safe_config)
            if not is_safe:
                return False, {}, "Configuration contains unsafe content"
            
            return True, filtered_config, warning
            
        except Exception as e:
            shield_logger.error(f"Config validation error: {e}")
            return False, {}, f"Config validation error: {str(e)}"
    
    def create_secure_conversation(self, name: str = "") -> Optional[str]:
        """Create a secure conversation if encryption is enabled"""
        if self.swarm_shield and self.config.integrate_with_conversation:
            try:
                return self.swarm_shield.create_conversation(name)
            except Exception as e:
                shield_logger.error(f"Failed to create secure conversation: {e}")
                return None
        return None
    
    def add_secure_message(self, conversation_id: str, agent_name: str, message: str) -> bool:
        """Add a message to secure conversation"""
        if self.swarm_shield and self.config.integrate_with_conversation:
            try:
                self.swarm_shield.add_message(conversation_id, agent_name, message)
                return True
            except Exception as e:
                shield_logger.error(f"Failed to add secure message: {e}")
                return False
        return False
    
    def get_secure_messages(self, conversation_id: str) -> List[Tuple[str, str, datetime]]:
        """Get messages from secure conversation"""
        if self.swarm_shield and self.config.integrate_with_conversation:
            try:
                return self.swarm_shield.get_messages(conversation_id)
            except Exception as e:
                shield_logger.error(f"Failed to get secure messages: {e}")
                return []
        return []
    
    def check_rate_limit(self, agent_name: str, request_size: int = 1) -> Tuple[bool, Optional[str]]:
        """Check rate limits for an agent"""
        return self.rate_limiter.check_rate_limit(agent_name, request_size)
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get comprehensive security statistics"""
        stats = {
            "security_enabled": self.config.enabled,
            "input_validations": getattr(self.input_validator, 'validation_count', 0),
            "rate_limit_checks": getattr(self.rate_limiter, 'check_count', 0),
            "blocked_requests": getattr(self.rate_limiter, 'blocked_count', 0),
            "filtered_outputs": getattr(self.output_filter, 'filter_count', 0),
            "violations": getattr(self.safety_checker, 'violation_count', 0),
            "encryption_enabled": self.swarm_shield is not None,
        }
        
        # Handle security_level safely
        if hasattr(self.config.security_level, 'value'):
            stats["security_level"] = self.config.security_level.value
        else:
            stats["security_level"] = str(self.config.security_level)
        
        # Handle encryption_strength safely
        if self.swarm_shield and hasattr(self.swarm_shield.encryption_strength, 'value'):
            stats["encryption_strength"] = self.swarm_shield.encryption_strength.value
        elif self.swarm_shield:
            stats["encryption_strength"] = str(self.swarm_shield.encryption_strength)
        else:
            stats["encryption_strength"] = "none"
        
        return stats
    
    def update_config(self, new_config: ShieldConfig) -> bool:
        """Update shield configuration"""
        try:
            self.config = new_config
            self._initialize_components()
            shield_logger.info("Shield configuration updated")
            return True
        except Exception as e:
            shield_logger.error(f"Failed to update configuration: {e}")
            return False
    
    def enable_security(self) -> None:
        """Enable all security features"""
        self.config.enabled = True
        shield_logger.info("Security features enabled")
    
    def disable_security(self) -> None:
        """Disable all security features"""
        self.config.enabled = False
        shield_logger.info("Security features disabled")
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if hasattr(self, 'rate_limiter'):
                self.rate_limiter.stop()
            shield_logger.info("SwarmShield integration cleaned up")
        except Exception as e:
            shield_logger.error(f"Cleanup error: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup() 