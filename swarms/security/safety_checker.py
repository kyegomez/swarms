"""
Safety checking and content filtering for Swarms framework.

This module provides comprehensive safety checks, content filtering,
and ethical AI compliance for all swarm operations.
"""

import re
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from loguru import logger

from swarms.utils.loguru_logger import initialize_logger
from swarms.prompts.safety_prompt import SAFETY_PROMPT

# Initialize logger for safety checking
safety_logger = initialize_logger(log_folder="safety_checking")


class ContentLevel(Enum):
    """Content filtering levels"""
    
    LOW = "low"  # Minimal filtering
    MODERATE = "moderate"  # Standard filtering
    HIGH = "high"  # Aggressive filtering


class SafetyChecker:
    """
    Safety checking and content filtering for swarm security
    
    Features:
    - Content safety assessment
    - Ethical AI compliance
    - Harmful content detection
    - Bias detection
    - Safety prompt integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety checker with configuration
        
        Args:
            config: Safety configuration dictionary
        """
        self.enabled = config.get("enabled", True)
        self.safety_prompt = config.get("safety_prompt", True)
        self.filter_level = ContentLevel(config.get("filter_level", "moderate"))
        
        # Harmful content patterns
        self._harmful_patterns = [
            re.compile(r"\b(kill|murder|suicide|bomb|explosive|weapon)\b", re.IGNORECASE),
            re.compile(r"\b(hack|crack|steal|fraud|scam|phishing)\b", re.IGNORECASE),
            re.compile(r"\b(drug|illegal|contraband|smuggle)\b", re.IGNORECASE),
            re.compile(r"\b(hate|racist|sexist|discriminate)\b", re.IGNORECASE),
            re.compile(r"\b(terrorist|extremist|radical)\b", re.IGNORECASE),
        ]
        
        # Bias detection patterns
        self._bias_patterns = [
            re.compile(r"\b(all|every|always|never|none)\b", re.IGNORECASE),
            re.compile(r"\b(men are|women are|blacks are|whites are)\b", re.IGNORECASE),
            re.compile(r"\b(stereotypical|typical|usual)\b", re.IGNORECASE),
        ]
        
        # Age-inappropriate content
        self._age_inappropriate = [
            re.compile(r"\b(sex|porn|adult|explicit)\b", re.IGNORECASE),
            re.compile(r"\b(violence|gore|blood|death)\b", re.IGNORECASE),
        ]
        
        # Handle filter_level safely
        filter_level_str = (
            self.filter_level.value 
            if hasattr(self.filter_level, 'value') 
            else str(self.filter_level)
        )
        safety_logger.info(f"SafetyChecker initialized with level: {filter_level_str}")
    
    def check_safety(self, content: str, content_type: str = "text") -> Tuple[bool, str, Optional[str]]:
        """
        Check content for safety and ethical compliance
        
        Args:
            content: Content to check
            content_type: Type of content (text, code, config, etc.)
            
        Returns:
            Tuple of (is_safe, sanitized_content, warning_message)
        """
        if not self.enabled:
            return True, content, None
            
        try:
            # Basic type validation
            if not isinstance(content, str):
                return False, "", "Content must be a string"
            
            # Check for harmful content
            if self._check_harmful_content(content):
                return False, "", "Content contains potentially harmful material"
            
            # Check for bias
            if self._check_bias(content):
                return False, "", "Content contains potentially biased language"
            
            # Check age appropriateness
            if self._check_age_appropriate(content):
                return False, "", "Content may not be age-appropriate"
            
            # Apply content filtering based on level
            sanitized = self._filter_content(content)
            
            # Check if content was modified
            warning = None
            if sanitized != content:
                # Handle filter_level safely
                filter_level_str = (
                    self.filter_level.value 
                    if hasattr(self.filter_level, 'value') 
                    else str(self.filter_level)
                )
                warning = f"Content was filtered for {filter_level_str} safety level"
            
            safety_logger.debug(f"Safety check passed for type: {content_type}")
            return True, sanitized, warning
            
        except Exception as e:
            safety_logger.error(f"Safety check error: {e}")
            return False, "", f"Safety check error: {str(e)}"
    
    def _check_harmful_content(self, content: str) -> bool:
        """Check for harmful content patterns"""
        for pattern in self._harmful_patterns:
            if pattern.search(content):
                safety_logger.warning(f"Harmful content detected: {pattern.pattern}")
                return True
        return False
    
    def _check_bias(self, content: str) -> bool:
        """Check for biased language patterns"""
        for pattern in self._bias_patterns:
            if pattern.search(content):
                safety_logger.warning(f"Bias detected: {pattern.pattern}")
                return True
        return False
    
    def _check_age_appropriate(self, content: str) -> bool:
        """Check for age-inappropriate content"""
        for pattern in self._age_inappropriate:
            if pattern.search(content):
                safety_logger.warning(f"Age-inappropriate content detected: {pattern.pattern}")
                return True
        return False
    
    def _filter_content(self, content: str) -> str:
        """Filter content based on safety level"""
        filtered_content = content
        
        if self.filter_level == ContentLevel.HIGH:
            # Aggressive filtering
            for pattern in self._harmful_patterns + self._bias_patterns:
                filtered_content = pattern.sub("[FILTERED]", filtered_content)
        
        elif self.filter_level == ContentLevel.MODERATE:
            # Moderate filtering - only filter obvious harmful content
            for pattern in self._harmful_patterns:
                filtered_content = pattern.sub("[FILTERED]", filtered_content)
        
        # LOW level doesn't filter content, only detects
        
        return filtered_content
    
    def check_agent_safety(self, agent_name: str, system_prompt: str) -> Tuple[bool, str, Optional[str]]:
        """Check agent system prompt for safety"""
        return self.check_safety(system_prompt, "agent_prompt")
    
    def check_task_safety(self, task: str) -> Tuple[bool, str, Optional[str]]:
        """Check task description for safety"""
        return self.check_safety(task, "task")
    
    def check_response_safety(self, response: str, agent_name: str) -> Tuple[bool, str, Optional[str]]:
        """Check agent response for safety"""
        return self.check_safety(response, "response")
    
    def check_config_safety(self, config: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Optional[str]]:
        """Check configuration for safety"""
        try:
            import json
            config_str = json.dumps(config)
            
            is_safe, sanitized, error = self.check_safety(config_str, "config")
            if not is_safe:
                return False, {}, error
            
            # Parse back to dict
            safe_config = json.loads(sanitized)
            return True, safe_config, None
            
        except Exception as e:
            return False, {}, f"Config safety check error: {str(e)}"
    
    def get_safety_prompt(self) -> str:
        """Get safety prompt for integration"""
        if self.safety_prompt:
            return SAFETY_PROMPT
        return ""
    
    def add_harmful_pattern(self, pattern: str, description: str = "") -> None:
        """Add custom harmful content pattern"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self._harmful_patterns.append(compiled_pattern)
            
            safety_logger.info(f"Added harmful pattern: {pattern} ({description})")
            
        except re.error as e:
            safety_logger.error(f"Invalid regex pattern: {pattern} - {e}")
    
    def add_bias_pattern(self, pattern: str, description: str = "") -> None:
        """Add custom bias detection pattern"""
        try:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            self._bias_patterns.append(compiled_pattern)
            
            safety_logger.info(f"Added bias pattern: {pattern} ({description})")
            
        except re.error as e:
            safety_logger.error(f"Invalid regex pattern: {pattern} - {e}")
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety checking statistics"""
        # Handle filter_level safely
        filter_level_str = (
            self.filter_level.value 
            if hasattr(self.filter_level, 'value') 
            else str(self.filter_level)
        )
        
        return {
            "enabled": self.enabled,
            "safety_prompt": self.safety_prompt,
            "filter_level": filter_level_str,
            "harmful_patterns_count": len(self._harmful_patterns),
            "bias_patterns_count": len(self._bias_patterns),
            "age_inappropriate_patterns_count": len(self._age_inappropriate),
        } 