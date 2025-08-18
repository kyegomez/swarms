"""
Output filtering and sanitization for Swarms framework.

This module provides comprehensive output filtering, sanitization,
and sensitive data protection for all swarm outputs.
"""

import re
import json
from typing import List, Optional, Dict, Any, Tuple, Union
from datetime import datetime

from loguru import logger

from swarms.utils.loguru_logger import initialize_logger

# Initialize logger for output filtering
filter_logger = initialize_logger(log_folder="output_filtering")


class OutputFilter:
    """
    Output filtering and sanitization for swarm security
    
    Features:
    - Sensitive data filtering
    - Output sanitization
    - Content type filtering
    - PII protection
    - Malicious content detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize output filter with configuration
        
        Args:
            config: Filtering configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        self.filter_sensitive = config.get("filter_sensitive", True)
        self.sensitive_patterns = config.get("sensitive_patterns", [])
        
        # Compile regex patterns for performance
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.sensitive_patterns
        ]
        
        # Default sensitive data patterns
        self._default_patterns = [
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
            re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # Credit card
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
            re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IP address
            re.compile(r"\b\d{3}[\s-]?\d{3}[\s-]?\d{4}\b"),  # Phone number
            re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),  # IBAN
            re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{1,3}\b"),  # Extended CC
        ]
        
        # Malicious content patterns
        self._malicious_patterns = [
            re.compile(r"<script.*?>.*?</script>", re.IGNORECASE),
            re.compile(r"javascript:", re.IGNORECASE),
            re.compile(r"data:text/html", re.IGNORECASE),
            re.compile(r"vbscript:", re.IGNORECASE),
            re.compile(r"on\w+\s*=", re.IGNORECASE),
            re.compile(r"<iframe.*?>.*?</iframe>", re.IGNORECASE),
            re.compile(r"<object.*?>.*?</object>", re.IGNORECASE),
            re.compile(r"<embed.*?>", re.IGNORECASE),
        ]
        
        # API key patterns
        self._api_key_patterns = [
            re.compile(r"sk-[a-zA-Z0-9]{32,}"),  # OpenAI API key
            re.compile(r"pk_[a-zA-Z0-9]{32,}"),  # OpenAI API key (public)
            re.compile(r"[a-zA-Z0-9]{32,}"),  # Generic API key
        ]
        
        filter_logger.info("OutputFilter initialized")
    
    def filter_output(self, output_data: Union[str, Dict, List], output_type: str = "text") -> Tuple[bool, Union[str, Dict, List], Optional[str]]:
        """
        Filter and sanitize output data
        
        Args:
            output_data: Output data to filter
            output_type: Type of output (text, json, dict, etc.)
            
        Returns:
            Tuple of (is_safe, filtered_data, warning_message)
        """
        if not self.enabled:
            return True, output_data, None
            
        try:
            # Convert to string for processing
            if isinstance(output_data, (dict, list)):
                output_str = json.dumps(output_data, ensure_ascii=False)
            else:
                output_str = str(output_data)
            
            # Check for malicious content
            if self._check_malicious_content(output_str):
                return False, "", "Output contains potentially malicious content"
            
            # Filter sensitive data
            if self.filter_sensitive:
                filtered_str = self._filter_sensitive_data(output_str)
            else:
                filtered_str = output_str
            
            # Convert back to original type if needed
            if isinstance(output_data, (dict, list)) and output_type in ["json", "dict"]:
                try:
                    filtered_data = json.loads(filtered_str)
                except json.JSONDecodeError:
                    filtered_data = filtered_str
            else:
                filtered_data = filtered_str
            
            # Check if any sensitive data was filtered
            warning = None
            if filtered_str != output_str:
                warning = "Sensitive data was filtered from output"
            
            filter_logger.debug(f"Output filtering completed for type: {output_type}")
            return True, filtered_data, warning
            
        except Exception as e:
            filter_logger.error(f"Output filtering error: {e}")
            return False, "", f"Filtering error: {str(e)}"
    
    def _check_malicious_content(self, content: str) -> bool:
        """Check if content contains malicious patterns"""
        for pattern in self._malicious_patterns:
            if pattern.search(content):
                filter_logger.warning(f"Malicious content detected: {pattern.pattern}")
                return True
        return False
    
    def _filter_sensitive_data(self, content: str) -> str:
        """Filter sensitive data from content"""
        filtered_content = content
        
        # Filter custom sensitive patterns
        for pattern in self._compiled_patterns:
            filtered_content = pattern.sub("[SENSITIVE_DATA]", filtered_content)
        
        # Filter default sensitive patterns
        for pattern in self._default_patterns:
            filtered_content = pattern.sub("[SENSITIVE_DATA]", filtered_content)
        
        # Filter API keys
        for pattern in self._api_key_patterns:
            filtered_content = pattern.sub("[API_KEY]", filtered_content)
        
        return filtered_content
    
    def filter_agent_response(self, response: str, agent_name: str) -> Tuple[bool, str, Optional[str]]:
        """Filter agent response output"""
        return self.filter_output(response, "text")
    
    def filter_swarm_output(self, output: Union[str, Dict, List]) -> Tuple[bool, Union[str, Dict, List], Optional[str]]:
        """Filter swarm output"""
        return self.filter_output(output, "json")
    
    def filter_conversation_history(self, history: List[Dict]) -> Tuple[bool, List[Dict], Optional[str]]:
        """Filter conversation history"""
        try:
            filtered_history = []
            warnings = []
            
            for message in history:
                # Filter message content
                is_safe, filtered_content, warning = self.filter_output(
                    message.get("content", ""), "text"
                )
                
                if not is_safe:
                    return False, [], "Conversation history contains unsafe content"
                
                # Create filtered message
                filtered_message = message.copy()
                filtered_message["content"] = filtered_content
                
                if warning:
                    warnings.append(warning)
                
                filtered_history.append(filtered_message)
            
            warning_msg = "; ".join(set(warnings)) if warnings else None
            return True, filtered_history, warning_msg
            
        except Exception as e:
            filter_logger.error(f"Conversation history filtering error: {e}")
            return False, [], f"History filtering error: {str(e)}"
    
    def filter_config_output(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Filter configuration output"""
        try:
            # Create a copy to avoid modifying original
            filtered_config = config.copy()
            
            # Filter sensitive config fields
            sensitive_fields = [
                "api_key", "secret", "password", "token", "key",
                "credential", "auth", "private", "secret_key"
            ]
            
            warnings = []
            for field in sensitive_fields:
                if field in filtered_config:
                    if isinstance(filtered_config[field], str):
                        filtered_config[field] = "[SENSITIVE_CONFIG]"
                        warnings.append(f"Sensitive config field '{field}' was filtered")
            
            warning_msg = "; ".join(warnings) if warnings else None
            return True, filtered_config, warning_msg
            
        except Exception as e:
            filter_logger.error(f"Config filtering error: {e}")
            return False, {}, f"Config filtering error: {str(e)}"
    
    def sanitize_for_logging(self, data: Union[str, Dict, List]) -> str:
        """Sanitize data for logging purposes"""
        try:
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)
            
            # Apply aggressive filtering for logs
            sanitized = self._filter_sensitive_data(data_str)
            
            # Truncate if too long
            if len(sanitized) > 1000:
                sanitized = sanitized[:1000] + "... [TRUNCATED]"
            
            return sanitized
            
        except Exception as e:
            filter_logger.error(f"Log sanitization error: {e}")
            return "[SANITIZATION_ERROR]"
    
    def add_custom_pattern(self, pattern: str, description: str = "") -> None:
        """Add custom sensitive data pattern"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self._compiled_patterns.append(compiled_pattern)
            self.sensitive_patterns.append(pattern)
            
            filter_logger.info(f"Added custom pattern: {pattern} ({description})")
            
        except re.error as e:
            filter_logger.error(f"Invalid regex pattern: {pattern} - {e}")
    
    def remove_pattern(self, pattern: str) -> bool:
        """Remove sensitive data pattern"""
        try:
            if pattern in self.sensitive_patterns:
                self.sensitive_patterns.remove(pattern)
                
                # Recompile patterns
                self._compiled_patterns = [
                    re.compile(p, re.IGNORECASE) for p in self.sensitive_patterns
                ]
                
                filter_logger.info(f"Removed pattern: {pattern}")
                return True
            return False
            
        except Exception as e:
            filter_logger.error(f"Error removing pattern: {e}")
            return False
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            "enabled": self.enabled,
            "filter_sensitive": self.filter_sensitive,
            "sensitive_patterns_count": len(self.sensitive_patterns),
            "malicious_patterns_count": len(self._malicious_patterns),
            "api_key_patterns_count": len(self._api_key_patterns),
            "default_patterns_count": len(self._default_patterns),
        } 