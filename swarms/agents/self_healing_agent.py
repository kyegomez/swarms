import os
import sys
import traceback
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

from swarms.utils.terminal_output import terminal
from swarms.structs.agent import Agent

@dataclass
class ErrorContext:
    error_type: str
    error_message: str
    traceback: str
    code_snippet: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None

class CodeFixerAgent(Agent):
    """An agent specialized in analyzing and fixing code errors"""
    
    def __init__(self, *args, **kwargs):
        system_prompt = """You are an expert code debugging and fixing agent. Your role is to:
        1. Analyze error messages and stack traces to understand the root cause
        2. Examine the code context where the error occurred
        3. Propose specific fixes with clear explanations
        4. Consider multiple potential solutions and their trade-offs
        5. Ensure fixes maintain code quality and follow best practices
        
        When proposing fixes:
        - Explain why the error occurred
        - Detail what changes need to be made and in which files
        - Consider potential side effects of the changes
        - Suggest any additional improvements or preventive measures
        
        Format your response as:
        ERROR ANALYSIS:
        <explanation of what caused the error>
        
        PROPOSED FIX:
        File: <file_path>
        Lines: <line_numbers>
        ```<language>
        <code changes>
        ```
        
        EXPLANATION:
        <why this fix works and any considerations>
        """
        
        kwargs["system_prompt"] = system_prompt
        kwargs["agent_name"] = kwargs.get("agent_name", "Code-Fixer-Agent")
        super().__init__(*args, **kwargs)
        self.error_history: List[ErrorContext] = []

class SelfHealingAgent(Agent):
    """An agent that can diagnose and fix runtime errors using LLM-based analysis"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize the code fixer agent
        self.fixer_agent = CodeFixerAgent(
            llm=self.llm,
            max_loops=1,
            verbose=True
        )
        
    def diagnose_error(self, error: Exception) -> ErrorContext:
        """Gather context about an error"""
        tb = traceback.extract_tb(sys.exc_info()[2])
        file_path = None
        line_number = None
        code_snippet = ""
        
        # Get the last frame from traceback which is usually where the error occurred
        if tb:
            last_frame = tb[-1]
            file_path = last_frame.filename
            line_number = last_frame.lineno
            
            # Try to get code context
            if file_path and os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    start = max(0, line_number - 5)  # Get more context
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
        """Create a detailed prompt for the fixer agent"""
        return f"""
        An error occurred in the code. Please analyze it and propose a fix.
        
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
        
        Please analyze this error and propose a specific fix. Include:
        1. What caused the error
        2. Exact changes needed (file paths and line numbers)
        3. The code that needs to be changed
        4. Why the fix will work
        5. Any potential side effects to consider
        """
    
    def apply_fix(self, fix_proposal: str) -> bool:
        """Apply the fix proposed by the fixer agent"""
        try:
            # Parse the fix proposal to extract file and changes
            import re
            
            # Extract file path
            file_match = re.search(r"File: (.+)", fix_proposal)
            if not file_match:
                terminal.status_panel("Could not find file path in fix proposal", "error")
                return False
                
            file_path = file_match.group(1).strip()
            
            # Extract code changes
            code_match = re.search(r"```(?:python)?\n(.*?)\n```", fix_proposal, re.DOTALL)
            if not code_match:
                terminal.status_panel("Could not find code changes in fix proposal", "error")
                return False
                
            new_code = code_match.group(1).strip()
            
            # Extract line numbers if specified
            lines_match = re.search(r"Lines: (.+)", fix_proposal)
            line_range = None
            if lines_match:
                try:
                    # Parse line range (e.g., "5-10" or "5")
                    line_spec = lines_match.group(1).strip()
                    if '-' in line_spec:
                        start, end = map(int, line_spec.split('-'))
                        line_range = (start, end)
                    else:
                        line_num = int(line_spec)
                        line_range = (line_num, line_num)
                except:
                    terminal.status_panel("Could not parse line numbers", "warning")
            
            # Apply the changes
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            if line_range:
                # Replace specific lines
                start, end = line_range
                start -= 1  # Convert to 0-based index
                lines[start:end] = new_code.splitlines(True)
            else:
                # Replace the entire file
                lines = new_code.splitlines(True)
            
            with open(file_path, 'w') as f:
                f.writelines(lines)
            
            terminal.status_panel(f"Applied fix to {file_path}", "success")
            return True
            
        except Exception as e:
            terminal.status_panel(f"Failed to apply fix: {str(e)}", "error")
            return False
    
    def run(self, *args, **kwargs) -> Any:
        """Run with self-healing capabilities using LLM-based analysis"""
        try:
            return super().run(*args, **kwargs)
        except Exception as error:
            terminal.status_panel("Error detected, analyzing with LLM...", "warning")
            
            # Gather error context
            error_context = self.diagnose_error(error)
            self.error_history.append(error_context)
            
            # Get fix proposal from the fixer agent
            fix_prompt = self.get_fix_prompt(error_context)
            fix_proposal = self.fixer_agent.run(fix_prompt)
            
            terminal.status_panel("Fix proposed by LLM:", "info")
            terminal.status_panel(fix_proposal, "info")
            
            # Apply the fix
            if self.apply_fix(fix_proposal):
                terminal.status_panel("Fix applied, retrying operation...", "info")
                # Retry the operation
                return super().run(*args, **kwargs)
            else:
                terminal.status_panel("Unable to apply fix automatically", "error")
                raise 