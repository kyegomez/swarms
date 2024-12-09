import os
import sys
import ast
import json
import traceback
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from swarms.utils.terminal_output import terminal
from swarms.structs.agent import Agent

@dataclass
class ErrorContext:
    """Context about an error that occurred"""
    error_type: str
    error_message: str
    traceback: str
    code_snippet: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None

class CodeFixerAgent(Agent):
    """An agent specialized in analyzing and fixing code errors"""
    
    def __init__(self, *args, **kwargs):
        system_prompt = """You are an expert code debugging and fixing agent. Your role is to analyze errors and propose fixes.

        When analyzing errors, follow these steps:
        1. Examine the error message and stack trace carefully
        2. Look at the code context where the error occurred
        3. Consider multiple potential causes and solutions
        4. Choose the most appropriate fix
        5. Explain your reasoning clearly

        Your output must follow this exact format:

        {
            "error_analysis": {
                "root_cause": "Brief explanation of what caused the error",
                "impact": "What effects this error has on the system",
                "severity": "high|medium|low"
            },
            "proposed_fix": {
                "file": "Path to the file that needs changes",
                "line_range": "start-end or single line number",
                "code_changes": "The actual code changes to make",
                "type": "syntax|import|permission|memory|other"
            },
            "explanation": {
                "why_it_works": "Why this fix will solve the problem",
                "side_effects": "Any potential side effects to consider",
                "alternatives": "Other possible solutions that were considered"
            },
            "prevention": {
                "recommendations": ["List of recommendations to prevent similar errors"],
                "best_practices": ["Relevant best practices to follow"]
            }
        }

        Always ensure your response is valid JSON and includes all the required fields.
        Be specific about file paths and line numbers.
        Include complete code snippets that can be directly applied.
        """
        
        kwargs["system_prompt"] = system_prompt
        kwargs["agent_name"] = kwargs.get("agent_name", "Code-Fixer-Agent")
        kwargs["output_type"] = "json"  # Ensure JSON output
        super().__init__(*args, **kwargs)
        self.error_history: List[ErrorContext] = []

class SelfHealingAgent(Agent):
    """An agent that can diagnose and fix runtime errors using LLM-based analysis
    
    This agent uses a specialized CodeFixerAgent to analyze errors and propose fixes.
    It can handle various types of errors including:
    - Syntax errors
    - Import errors
    - Permission errors
    - Memory errors
    - General runtime errors
    
    The agent maintains a history of errors and fixes, and can provide detailed reports
    of its healing activities.
    
    Attributes:
        error_history (List[ErrorContext]): History of errors encountered
        fixer_agent (CodeFixerAgent): Specialized agent for analyzing and fixing errors
        max_fix_attempts (int): Maximum number of fix attempts per error
    """
    
    def __init__(self, *args, **kwargs):
        system_prompt = """You are a self-healing agent capable of detecting, analyzing, and fixing runtime errors.
        Your responses should follow this format:

        {
            "status": {
                "state": "running|error|fixed|failed",
                "message": "Current status message",
                "timestamp": "ISO timestamp"
            },
            "error_details": {
                "type": "Error type if applicable",
                "message": "Error message if applicable",
                "location": "File and line number where error occurred"
            },
            "healing_actions": {
                "attempted_fixes": ["List of fixes attempted"],
                "successful_fixes": ["List of successful fixes"],
                "failed_fixes": ["List of failed fixes"]
            },
            "system_health": {
                "memory_usage": "Current memory usage",
                "cpu_usage": "Current CPU usage",
                "disk_usage": "Current disk usage"
            },
            "recommendations": {
                "immediate": ["Immediate actions needed"],
                "long_term": ["Long-term improvements suggested"]
            }
        }
        """
        
        kwargs["system_prompt"] = system_prompt
        kwargs["agent_name"] = kwargs.get("agent_name", "Self-Healing-Agent")
        kwargs["output_type"] = "json"  # Ensure JSON output
        super().__init__(*args, **kwargs)
        
        # Initialize the code fixer agent
        self.fixer_agent = CodeFixerAgent(
            llm=self.llm,
            max_loops=1,
            verbose=True
        )
        
        self.error_history = []
        self.max_fix_attempts = 3
        
    def diagnose_error(self, error: Exception) -> ErrorContext:
        """Gather detailed context about an error
        
        Args:
            error (Exception): The error that occurred
            
        Returns:
            ErrorContext: Detailed context about the error
        """
        tb = traceback.extract_tb(sys.exc_info()[2])
        file_path = None
        line_number = None
        code_snippet = ""
        
        if tb:
            last_frame = tb[-1]
            file_path = last_frame.filename
            line_number = last_frame.lineno
            
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    start = max(0, line_number - 5)
                    end = min(len(lines), line_number + 5)
                    code_snippet = ''.join(lines[start:end])
        
        return ErrorContext(
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
            code_snippet=code_snippet,
            file_path=file_path,
            line_number=line_number
        )
    
    def get_fix_prompt(self, error_context: ErrorContext) -> str:
        """Create a detailed prompt for the fixer agent
        
        Args:
            error_context (ErrorContext): Context about the error
            
        Returns:
            str: Prompt for the fixer agent
        """
        return f"""
        Analyze this error and propose a fix following the required JSON format.
        
        ERROR TYPE: {error_context.error_type}
        ERROR MESSAGE: {error_context.error_message}
        
        FILE: {error_context.file_path}
        LINE NUMBER: {error_context.line_number}
        
        CODE CONTEXT:
        ```python
        {error_context.code_snippet}
        ```
        
        FULL TRACEBACK:
        {error_context.traceback}
        """
    
    def apply_fix(self, fix_proposal: Dict[str, Any]) -> bool:
        """Apply a fix proposed by the fixer agent
        
        Args:
            fix_proposal (Dict[str, Any]): The fix proposal in JSON format
            
        Returns:
            bool: Whether the fix was successfully applied
        """
        try:
            # Extract fix details
            file_path = fix_proposal["proposed_fix"]["file"]
            line_range = fix_proposal["proposed_fix"]["line_range"]
            new_code = fix_proposal["proposed_fix"]["code_changes"]
            
            # Parse line range
            if '-' in line_range:
                start, end = map(int, line_range.split('-'))
            else:
                start = end = int(line_range)
            
            # Apply the changes
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Convert to 0-based indexing
            start -= 1
            lines[start:end] = new_code.splitlines(True)
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
            terminal.status_panel(
                f"Applied fix to {file_path} lines {start+1}-{end}", 
                "success"
            )
            return True
            
        except Exception as e:
            terminal.status_panel(f"Failed to apply fix: {str(e)}", "error")
            return False
    
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Run with self-healing capabilities using LLM-based analysis
        
        Returns:
            Dict[str, Any]: Structured output about the run and any healing actions
        """
        try:
            result = super().run(*args, **kwargs)
            
            # Return success status
            return {
                "status": {
                    "state": "running",
                    "message": "Operation completed successfully",
                    "timestamp": datetime.now().isoformat()
                },
                "error_details": None,
                "healing_actions": {
                    "attempted_fixes": [],
                    "successful_fixes": [],
                    "failed_fixes": []
                },
                "system_health": self.get_system_health(),
                "recommendations": {
                    "immediate": [],
                    "long_term": ["Monitor system health regularly"]
                }
            }
            
        except Exception as error:
            terminal.status_panel("Error detected, analyzing with LLM...", "warning")
            
            # Gather error context
            error_context = self.diagnose_error(error)
            self.error_history.append(error_context)
            
            # Get fix proposal from the fixer agent
            fix_prompt = self.get_fix_prompt(error_context)
            fix_proposal = self.fixer_agent.run(fix_prompt)
            
            terminal.status_panel("Fix proposed by LLM:", "info")
            terminal.status_panel(json.dumps(fix_proposal, indent=2), "info")
            
            # Track healing actions
            attempted_fixes = [fix_proposal["proposed_fix"]["type"]]
            successful_fixes = []
            failed_fixes = []
            
            # Apply the fix
            if self.apply_fix(fix_proposal):
                terminal.status_panel("Fix applied, retrying operation...", "info")
                successful_fixes.append(fix_proposal["proposed_fix"]["type"])
                
                try:
                    # Retry the operation
                    result = super().run(*args, **kwargs)
                    
                    return {
                        "status": {
                            "state": "fixed",
                            "message": "Error fixed and operation completed",
                            "timestamp": datetime.now().isoformat()
                        },
                        "error_details": {
                            "type": error_context.error_type,
                            "message": error_context.error_message,
                            "location": f"{error_context.file_path}:{error_context.line_number}"
                        },
                        "healing_actions": {
                            "attempted_fixes": attempted_fixes,
                            "successful_fixes": successful_fixes,
                            "failed_fixes": failed_fixes
                        },
                        "system_health": self.get_system_health(),
                        "recommendations": fix_proposal["prevention"]
                    }
                    
                except Exception as e:
                    failed_fixes.append(fix_proposal["proposed_fix"]["type"])
                    
                    return {
                        "status": {
                            "state": "failed",
                            "message": "Fix applied but error persists",
                            "timestamp": datetime.now().isoformat()
                        },
                        "error_details": {
                            "type": type(e).__name__,
                            "message": str(e),
                            "location": f"{error_context.file_path}:{error_context.line_number}"
                        },
                        "healing_actions": {
                            "attempted_fixes": attempted_fixes,
                            "successful_fixes": successful_fixes,
                            "failed_fixes": failed_fixes
                        },
                        "system_health": self.get_system_health(),
                        "recommendations": {
                            "immediate": ["Manual intervention required"],
                            "long_term": fix_proposal["prevention"]["recommendations"]
                        }
                    }
            else:
                failed_fixes.append(fix_proposal["proposed_fix"]["type"])
                return {
                    "status": {
                        "state": "error",
                        "message": "Unable to apply fix",
                        "timestamp": datetime.now().isoformat()
                    },
                    "error_details": {
                        "type": error_context.error_type,
                        "message": error_context.error_message,
                        "location": f"{error_context.file_path}:{error_context.line_number}"
                    },
                    "healing_actions": {
                        "attempted_fixes": attempted_fixes,
                        "successful_fixes": successful_fixes,
                        "failed_fixes": failed_fixes
                    },
                    "system_health": self.get_system_health(),
                    "recommendations": {
                        "immediate": ["Manual intervention required"],
                        "long_term": fix_proposal["prevention"]["recommendations"]
                    }
                }
    
    def get_system_health(self) -> Dict[str, str]:
        """Get current system health metrics
        
        Returns:
            Dict[str, str]: System health metrics
        """
        import psutil
        
        return {
            "memory_usage": f"{psutil.virtual_memory().percent}%",
            "cpu_usage": f"{psutil.cpu_percent()}%",
            "disk_usage": f"{psutil.disk_usage('/').percent}%"
        }