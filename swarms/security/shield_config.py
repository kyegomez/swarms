"""
Shield configuration for Swarms framework.

This module provides configuration options for the security shield,
allowing users to customize security settings for their swarm deployments.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

from .swarm_shield import EncryptionStrength


class SecurityLevel(Enum):
    """Security levels for shield configuration"""
    
    BASIC = "basic"  # Basic input validation and output filtering
    STANDARD = "standard"  # Standard security with encryption
    ENHANCED = "enhanced"  # Enhanced security with additional checks
    MAXIMUM = "maximum"  # Maximum security with all features enabled


class ShieldConfig(BaseModel):
    """
    Configuration for SwarmShield security features
    
    This class provides comprehensive configuration options for
    enabling and customizing security features across all swarm architectures.
    """
    
    # Core security settings
    enabled: bool = Field(default=True, description="Enable shield protection")
    security_level: SecurityLevel = Field(
        default=SecurityLevel.STANDARD,
        description="Overall security level"
    )
    
    # SwarmShield settings
    encryption_strength: EncryptionStrength = Field(
        default=EncryptionStrength.MAXIMUM,
        description="Encryption strength for message protection"
    )
    key_rotation_interval: int = Field(
        default=3600,
        ge=300,  # Minimum 5 minutes
        description="Key rotation interval in seconds"
    )
    storage_path: Optional[str] = Field(
        default=None,
        description="Path for encrypted storage"
    )
    
    # Input validation settings
    enable_input_validation: bool = Field(
        default=True,
        description="Enable input validation and sanitization"
    )
    max_input_length: int = Field(
        default=10000,
        ge=100,
        description="Maximum input length in characters"
    )
    blocked_patterns: List[str] = Field(
        default=[
            r"<script.*?>.*?</script>",  # XSS prevention
            r"javascript:",  # JavaScript injection
            r"data:text/html",  # HTML injection
            r"vbscript:",  # VBScript injection
            r"on\w+\s*=",  # Event handler injection
        ],
        description="Regex patterns to block in inputs"
    )
    allowed_domains: List[str] = Field(
        default=[],
        description="Allowed domains for external requests"
    )
    
    # Output filtering settings
    enable_output_filtering: bool = Field(
        default=True,
        description="Enable output filtering and sanitization"
    )
    filter_sensitive_data: bool = Field(
        default=True,
        description="Filter sensitive data from outputs"
    )
    sensitive_patterns: List[str] = Field(
        default=[
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP address
        ],
        description="Regex patterns for sensitive data"
    )
    
    # Safety checking settings
    enable_safety_checks: bool = Field(
        default=True,
        description="Enable safety and content filtering"
    )
    safety_prompt_on: bool = Field(
        default=True,
        description="Enable safety prompt integration"
    )
    content_filter_level: str = Field(
        default="moderate",
        description="Content filtering level (low, moderate, high)"
    )
    
    # Rate limiting settings
    enable_rate_limiting: bool = Field(
        default=True,
        description="Enable rate limiting and abuse prevention"
    )
    max_requests_per_minute: int = Field(
        default=60,
        ge=1,
        description="Maximum requests per minute per agent"
    )
    max_tokens_per_request: int = Field(
        default=10000,
        ge=100,
        description="Maximum tokens per request"
    )
    rate_limit_window: int = Field(
        default=60,
        ge=10,
        description="Rate limit window in seconds"
    )
    
    # Audit and logging settings
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable comprehensive audit logging"
    )
    log_security_events: bool = Field(
        default=True,
        description="Log security-related events"
    )
    log_input_output: bool = Field(
        default=False,
        description="Log input and output data (use with caution)"
    )
    audit_retention_days: int = Field(
        default=90,
        ge=1,
        description="Audit log retention period in days"
    )
    
    # Performance settings
    enable_caching: bool = Field(
        default=True,
        description="Enable security result caching"
    )
    cache_ttl: int = Field(
        default=300,
        ge=60,
        description="Cache TTL in seconds"
    )
    max_cache_size: int = Field(
        default=1000,
        ge=100,
        description="Maximum cache entries"
    )
    
    # Integration settings
    integrate_with_conversation: bool = Field(
        default=True,
        description="Integrate with conversation management"
    )
    protect_agent_communications: bool = Field(
        default=True,
        description="Protect inter-agent communications"
    )
    encrypt_storage: bool = Field(
        default=True,
        description="Encrypt persistent storage"
    )
    
    # Custom settings
    custom_rules: Dict[str, Any] = Field(
        default={},
        description="Custom security rules and configurations"
    )
    
    # Compatibility fields for examples
    encryption_enabled: bool = Field(
        default=True,
        description="Enable encryption (alias for enabled)"
    )
    input_validation_enabled: bool = Field(
        default=True,
        description="Enable input validation (alias for enable_input_validation)"
    )
    output_filtering_enabled: bool = Field(
        default=True,
        description="Enable output filtering (alias for enable_output_filtering)"
    )
    rate_limiting_enabled: bool = Field(
        default=True,
        description="Enable rate limiting (alias for enable_rate_limiting)"
    )
    safety_checking_enabled: bool = Field(
        default=True,
        description="Enable safety checking (alias for enable_safety_checks)"
    )
    block_suspicious_content: bool = Field(
        default=True,
        description="Block suspicious content patterns"
    )
    custom_blocked_patterns: List[str] = Field(
        default=[],
        description="Custom patterns to block in addition to default ones"
    )
    safety_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Safety threshold for content filtering"
    )
    bias_detection_enabled: bool = Field(
        default=False,
        description="Enable bias detection in content"
    )
    content_moderation_enabled: bool = Field(
        default=True,
        description="Enable content moderation"
    )
    
    class Config:
        """Pydantic configuration"""
        use_enum_values = True
        validate_assignment = True
        
    def get_encryption_config(self) -> Dict[str, Any]:
        """Get encryption configuration for SwarmShield"""
        return {
            "encryption_strength": self.encryption_strength,
            "key_rotation_interval": self.key_rotation_interval,
            "storage_path": self.storage_path,
        }
    
    def get_validation_config(self) -> Dict[str, Any]:
        """Get input validation configuration"""
        return {
            "enabled": self.enable_input_validation,
            "max_length": self.max_input_length,
            "blocked_patterns": self.blocked_patterns,
            "allowed_domains": self.allowed_domains,
        }
    
    def get_filtering_config(self) -> Dict[str, Any]:
        """Get output filtering configuration"""
        return {
            "enabled": self.enable_output_filtering,
            "filter_sensitive": self.filter_sensitive_data,
            "sensitive_patterns": self.sensitive_patterns,
        }
    
    def get_safety_config(self) -> Dict[str, Any]:
        """Get safety checking configuration"""
        return {
            "enabled": self.enable_safety_checks,
            "safety_prompt": self.safety_prompt_on,
            "filter_level": self.content_filter_level,
        }
    
    def get_rate_limit_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return {
            "enabled": self.enable_rate_limiting,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_tokens_per_request": self.max_tokens_per_request,
            "window": self.rate_limit_window,
        }
    
    def get_audit_config(self) -> Dict[str, Any]:
        """Get audit logging configuration"""
        return {
            "enabled": self.enable_audit_logging,
            "log_security": self.log_security_events,
            "log_io": self.log_input_output,
            "retention_days": self.audit_retention_days,
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return {
            "enable_caching": self.enable_caching,
            "cache_ttl": self.cache_ttl,
            "max_cache_size": self.max_cache_size,
        }
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get integration configuration"""
        return {
            "conversation": self.integrate_with_conversation,
            "agent_communications": self.protect_agent_communications,
            "encrypt_storage": self.encrypt_storage,
        }
    
    @classmethod
    def create_basic_config(cls) -> "ShieldConfig":
        """Create a basic security configuration"""
        return cls(
            security_level=SecurityLevel.BASIC,
            encryption_strength=EncryptionStrength.STANDARD,
            enable_safety_checks=False,
            enable_rate_limiting=False,
            enable_audit_logging=False,
        )
    
    @classmethod
    def create_standard_config(cls) -> "ShieldConfig":
        """Create a standard security configuration"""
        return cls(
            security_level=SecurityLevel.STANDARD,
            encryption_strength=EncryptionStrength.ENHANCED,
        )
    
    @classmethod
    def create_enhanced_config(cls) -> "ShieldConfig":
        """Create an enhanced security configuration"""
        return cls(
            security_level=SecurityLevel.ENHANCED,
            encryption_strength=EncryptionStrength.MAXIMUM,
            max_requests_per_minute=30,
            content_filter_level="high",
            log_input_output=True,
        )
    
    @classmethod
    def create_maximum_config(cls) -> "ShieldConfig":
        """Create a maximum security configuration"""
        return cls(
            security_level=SecurityLevel.MAXIMUM,
            encryption_strength=EncryptionStrength.MAXIMUM,
            key_rotation_interval=1800,  # 30 minutes
            max_requests_per_minute=20,
            content_filter_level="high",
            log_input_output=True,
            audit_retention_days=365,
            max_cache_size=500,
        )
    
    @classmethod
    def get_security_level(cls, level: str) -> "ShieldConfig":
        """Get a security configuration for the specified level"""
        level = level.lower()
        
        if level == "basic":
            return cls.create_basic_config()
        elif level == "standard":
            return cls.create_standard_config()
        elif level == "enhanced":
            return cls.create_enhanced_config()
        elif level == "maximum":
            return cls.create_maximum_config()
        else:
            raise ValueError(f"Unknown security level: {level}. Must be one of: basic, standard, enhanced, maximum") 