"""
Input validation and sanitization for Swarms framework.

This module provides comprehensive input validation, sanitization,
and security checks for all swarm inputs.
"""

import re
import html
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse
from datetime import datetime

from loguru import logger

from swarms.utils.loguru_logger import initialize_logger

# Initialize logger for input validation
validation_logger = initialize_logger(log_folder="input_validation")


class InputValidator:
    """
    Input validation and sanitization for swarm security
    
    Features:
    - Input length validation
    - Pattern-based blocking
    - XSS prevention
    - SQL injection prevention
    - URL validation
    - Content type validation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize input validator with configuration
        
        Args:
            config: Validation configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        self.max_length = config.get("max_length", 10000)
        self.blocked_patterns = config.get("blocked_patterns", [])
        self.allowed_domains = config.get("allowed_domains", [])
        
        # Compile regex patterns for performance
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_patterns
        ]
        
        # Common malicious patterns
        self._malicious_patterns = [
            re.compile(r"<script.*?>.*?</script>", re.IGNORECASE),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"data:text/html", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe.*?>.*?</iframe>", re.IGNORECASE),
            re.compile(r"<object.*?>.*?</object>", re.IGNORECASE),
            re.compile(r"<embed.*?>", re.IGNORECASE),
            re.compile(r"<link.*?>", re.IGNORECASE),
            re.compile(r"<meta.*?>", re.IGNORECASE),
        ]
        
        # SQL injection patterns
        self._sql_patterns = [
            re.compile(r"(\b(union|select|insert|update|delete|drop|create|alter)\b)", re.IGNORECASE),
            re.compile(r"(--|#|/\*|\*/)", re.IGNORECASE),
            re.compile(r"(\b(exec|execute|xp_|sp_)\b)", re.IGNORECASE),
        ]
        
        validation_logger.info("InputValidator initialized")
    
    def validate_input(self, input_data: str, input_type: str = "text") -> Tuple[bool, str, Optional[str]]:
        """
        Validate and sanitize input data
        
        Args:
            input_data: Input data to validate
            input_type: Type of input (text, url, code, etc.)
            
        Returns:
            Tuple of (is_valid, sanitized_data, error_message)
        """
        if not self.enabled:
            return True, input_data, None
            
        try:
            # Basic type validation
            if not isinstance(input_data, str):
                return False, "", "Input must be a string"
            
            # Length validation
            if len(input_data) > self.max_length:
                return False, "", f"Input exceeds maximum length of {self.max_length} characters"
            
            # Empty input check
            if not input_data.strip():
                return False, "", "Input cannot be empty"
            
            # Sanitize the input
            sanitized = self._sanitize_input(input_data)
            
            # Check for blocked patterns
            if self._check_blocked_patterns(sanitized):
                return False, "", "Input contains blocked patterns"
            
            # Check for malicious patterns
            if self._check_malicious_patterns(sanitized):
                return False, "", "Input contains potentially malicious content"
            
            # Type-specific validation
            if input_type == "url":
                if not self._validate_url(sanitized):
                    return False, "", "Invalid URL format"
            
            elif input_type == "code":
                if not self._validate_code(sanitized):
                    return False, "", "Invalid code content"
            
            elif input_type == "json":
                if not self._validate_json(sanitized):
                    return False, "", "Invalid JSON format"
            
            validation_logger.debug(f"Input validation passed for type: {input_type}")
            return True, sanitized, None
            
        except Exception as e:
            validation_logger.error(f"Input validation error: {e}")
            return False, "", f"Validation error: {str(e)}"
    
    def _sanitize_input(self, input_data: str) -> str:
        """Sanitize input data to prevent XSS and other attacks"""
        # HTML escape
        sanitized = html.escape(input_data)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _check_blocked_patterns(self, input_data: str) -> bool:
        """Check if input contains blocked patterns"""
        for pattern in self._compiled_patterns:
            if pattern.search(input_data):
                validation_logger.warning(f"Blocked pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _check_malicious_patterns(self, input_data: str) -> bool:
        """Check if input contains malicious patterns"""
        for pattern in self._malicious_patterns:
            if pattern.search(input_data):
                validation_logger.warning(f"Malicious pattern detected: {pattern.pattern}")
                return True
        return False
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL format and domain"""
        try:
            parsed = urlparse(url)
            
            # Check if it's a valid URL
            if not all([parsed.scheme, parsed.netloc]):
                return False
            
            # Check allowed domains if specified
            if self.allowed_domains:
                domain = parsed.netloc.lower()
                if not any(allowed in domain for allowed in self.allowed_domains):
                    validation_logger.warning(f"Domain not allowed: {domain}")
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_code(self, code: str) -> bool:
        """Validate code content for safety"""
        # Check for SQL injection patterns
        for pattern in self._sql_patterns:
            if pattern.search(code):
                validation_logger.warning(f"SQL injection pattern detected: {pattern.pattern}")
                return False
        
        # Check for dangerous system calls
        dangerous_calls = [
            'os.system', 'subprocess.call', 'eval(', 'exec(',
            '__import__', 'globals()', 'locals()'
        ]
        
        for call in dangerous_calls:
            if call in code:
                validation_logger.warning(f"Dangerous call detected: {call}")
                return False
        
        return True
    
    def _validate_json(self, json_str: str) -> bool:
        """Validate JSON format"""
        try:
            import json
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def validate_task(self, task: str) -> Tuple[bool, str, Optional[str]]:
        """Validate swarm task input"""
        return self.validate_input(task, "text")
    
    def validate_agent_name(self, agent_name: str) -> Tuple[bool, str, Optional[str]]:
        """Validate agent name input"""
        # Additional validation for agent names
        if not re.match(r'^[a-zA-Z0-9_-]+$', agent_name):
            return False, "", "Agent name can only contain letters, numbers, underscores, and hyphens"
        
        if len(agent_name) < 1 or len(agent_name) > 50:
            return False, "", "Agent name must be between 1 and 50 characters"
        
        return self.validate_input(agent_name, "text")
    
    def validate_message(self, message: str) -> Tuple[bool, str, Optional[str]]:
        """Validate message input"""
        return self.validate_input(message, "text")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Validate configuration input"""
        try:
            # Convert config to string for validation
            import json
            config_str = json.dumps(config)
            
            is_valid, sanitized, error = self.validate_input(config_str, "json")
            if not is_valid:
                return False, {}, error
            
            # Parse back to dict
            validated_config = json.loads(sanitized)
            return True, validated_config, None
            
        except Exception as e:
            return False, {}, f"Configuration validation error: {str(e)}"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "enabled": self.enabled,
            "max_length": self.max_length,
            "blocked_patterns_count": len(self.blocked_patterns),
            "allowed_domains_count": len(self.allowed_domains),
            "malicious_patterns_count": len(self._malicious_patterns),
            "sql_patterns_count": len(self._sql_patterns),
        } 