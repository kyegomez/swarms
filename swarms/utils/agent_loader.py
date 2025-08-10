import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from swarms.structs.agent import Agent


class MarkdownAgentConfig(BaseModel):
    """Configuration model for agents loaded from markdown files."""
    name: str
    description: str
    model_name: Optional[str] = "gpt-4"
    system_prompt: str
    focus_areas: Optional[List[str]] = []
    approach: Optional[List[str]] = []
    output: Optional[List[str]] = []
    max_loops: int = Field(default=1, ge=1)
    autosave: bool = False
    dashboard: bool = False
    verbose: bool = False
    dynamic_temperature_enabled: bool = False
    saved_state_path: Optional[str] = None
    user_name: str = "default_user"
    retry_attempts: int = Field(default=3, ge=1)
    context_length: int = Field(default=100000, ge=1000)
    return_step_meta: bool = False
    output_type: str = "str"
    auto_generate_prompt: bool = False
    artifacts_on: bool = False
    artifacts_file_extension: str = ".md"
    artifacts_output_path: str = ""

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError("System prompt must be a non-empty string")
        return v


class AgentLoader:
    """
    Loader for creating agents from markdown files.
    
    Supports both single markdown file and multiple markdown files.
    Maintains backwards compatibility with claude code sub agents markdown format.
    
    Features:
    - Single markdown file loading
    - Multiple markdown files loading (batch processing)
    - Flexible markdown parsing
    - Agent configuration extraction from markdown structure
    - Error handling and validation
    """
    
    def __init__(self):
        """
        Initialize the AgentLoader.
        """
        pass
        
    def parse_markdown_table(self, content: str) -> Dict[str, str]:
        """
        Parse markdown table to extract agent metadata.
        
        Args:
            content: Markdown content containing a table
            
        Returns:
            Dictionary with parsed table data
        """
        table_data = {}
        
        # Find markdown table pattern
        table_pattern = r'\|([^|]+)\|([^|]+)\|([^|]+)\|'
        lines = content.split('\n')
        
        header_found = False
        for line in lines:
            if '|' in line and not header_found:
                # Skip header separator line
                if '---' in line:
                    header_found = True
                    continue
                    
                # Parse header
                if 'name' in line.lower() and 'description' in line.lower():
                    continue
                    
            elif header_found and '|' in line:
                # Parse data row
                match = re.match(table_pattern, line)
                if match:
                    table_data['name'] = match.group(1).strip()
                    table_data['description'] = match.group(2).strip()
                    table_data['model_name'] = match.group(3).strip()
                break
                
        return table_data
    
    def extract_sections(self, content: str) -> Dict[str, List[str]]:
        """
        Extract structured sections from markdown content.
        
        Args:
            content: Markdown content
            
        Returns:
            Dictionary with section names as keys and content lists as values
        """
        sections = {}
        current_section = None
        current_content = []
        
        lines = content.split('\n')
        for line in lines:
            # Check for headers (## Section Name)
            if line.startswith('## '):
                # Save previous section
                if current_section:
                    sections[current_section.lower()] = current_content
                
                # Start new section
                current_section = line[3:].strip()
                current_content = []
                
            elif current_section and line.strip():
                # Add content to current section
                # Remove markdown list markers
                clean_line = re.sub(r'^[-*+]\s*', '', line.strip())
                clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                if clean_line:
                    current_content.append(clean_line)
        
        # Save last section
        if current_section:
            sections[current_section.lower()] = current_content
            
        return sections
    
    def build_system_prompt(self, config_data: Dict[str, Any]) -> str:
        """
        Build comprehensive system prompt from parsed markdown data.
        
        Args:
            config_data: Dictionary containing parsed agent configuration
            
        Returns:
            Complete system prompt string
        """
        prompt_parts = []
        
        # Add description
        if config_data.get('description'):
            prompt_parts.append(f"Role: {config_data['description']}")
        
        # Add focus areas
        if config_data.get('focus_areas'):
            prompt_parts.append("\nFocus Areas:")
            for area in config_data['focus_areas']:
                prompt_parts.append(f"- {area}")
        
        # Add approach
        if config_data.get('approach'):
            prompt_parts.append("\nApproach:")
            for i, step in enumerate(config_data['approach'], 1):
                prompt_parts.append(f"{i}. {step}")
        
        # Add expected output
        if config_data.get('output'):
            prompt_parts.append("\nExpected Output:")
            for output in config_data['output']:
                prompt_parts.append(f"- {output}")
        
        return '\n'.join(prompt_parts)
    
    def parse_markdown_file(self, file_path: str) -> MarkdownAgentConfig:
        """
        Parse a single markdown file to extract agent configuration.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            MarkdownAgentConfig object with parsed configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parsing fails
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file {file_path} not found.")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse table for basic metadata
            table_data = self.parse_markdown_table(content)
            
            # Extract sections
            sections = self.extract_sections(content)
            
            # Build configuration
            config_data = {
                'name': table_data.get('name', Path(file_path).stem),
                'description': table_data.get('description', 'Agent loaded from markdown'),
                'model_name': table_data.get('model_name', 'gpt-4'),
                'focus_areas': sections.get('focus areas', []),
                'approach': sections.get('approach', []),
                'output': sections.get('output', []),
            }
            
            # Build system prompt
            system_prompt = self.build_system_prompt(config_data)
            config_data['system_prompt'] = system_prompt
            
            logger.info(f"Successfully parsed markdown file: {file_path}")
            return MarkdownAgentConfig(**config_data)
            
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {str(e)}")
            raise ValueError(f"Error parsing markdown file {file_path}: {str(e)}")
    
    def load_agent_from_markdown(self, file_path: str, **kwargs) -> Agent:
        """
        Load a single agent from a markdown file.
        
        Args:
            file_path: Path to markdown file
            **kwargs: Additional arguments to override default configuration
            
        Returns:
            Configured Agent instance
        """
        config = self.parse_markdown_file(file_path)
        
        # Override with any provided kwargs
        config_dict = config.model_dump()
        config_dict.update(kwargs)
        
        # Remove fields not needed for Agent creation
        agent_fields = {
            'agent_name': config_dict['name'],
            'system_prompt': config_dict['system_prompt'],
            'model_name': config_dict.get('model_name', 'gpt-4'),
            # Don't pass llm explicitly - let Agent handle it internally
            'max_loops': config_dict['max_loops'],
            'autosave': config_dict['autosave'],
            'dashboard': config_dict['dashboard'],
            'verbose': config_dict['verbose'],
            'dynamic_temperature_enabled': config_dict['dynamic_temperature_enabled'],
            'saved_state_path': config_dict['saved_state_path'],
            'user_name': config_dict['user_name'],
            'retry_attempts': config_dict['retry_attempts'],
            'context_length': config_dict['context_length'],
            'return_step_meta': config_dict['return_step_meta'],
            'output_type': config_dict['output_type'],
            'auto_generate_prompt': config_dict['auto_generate_prompt'],
            'artifacts_on': config_dict['artifacts_on'],
            'artifacts_file_extension': config_dict['artifacts_file_extension'],
            'artifacts_output_path': config_dict['artifacts_output_path'],
        }
        
        try:
            logger.info(f"Creating agent '{config.name}' from {file_path}")
            agent = Agent(**agent_fields)
            logger.info(f"Successfully created agent '{config.name}' from {file_path}")
            return agent
        except Exception as e:
            import traceback
            logger.error(f"Error creating agent from {file_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise ValueError(f"Error creating agent from {file_path}: {str(e)}")
    
    def load_agents_from_markdown(self, file_paths: Union[str, List[str]], **kwargs) -> List[Agent]:
        """
        Load multiple agents from markdown files.
        
        Args:
            file_paths: Single file path, directory path, or list of file paths
            **kwargs: Additional arguments to override default configuration
            
        Returns:
            List of configured Agent instances
        """
        agents = []
        paths_to_process = []
        
        # Handle different input types
        if isinstance(file_paths, str):
            if os.path.isdir(file_paths):
                # Directory - find all .md files
                md_files = list(Path(file_paths).glob('*.md'))
                paths_to_process = [str(f) for f in md_files]
            elif os.path.isfile(file_paths):
                # Single file
                paths_to_process = [file_paths]
            else:
                raise FileNotFoundError(f"Path {file_paths} not found.")
        elif isinstance(file_paths, list):
            paths_to_process = file_paths
        else:
            raise ValueError("file_paths must be a string or list of strings")
        
        # Process each file
        for file_path in paths_to_process:
            try:
                agent = self.load_agent_from_markdown(file_path, **kwargs)
                agents.append(agent)
            except Exception as e:
                logger.warning(f"Skipping {file_path} due to error: {str(e)}")
                continue
        
        logger.info(f"Successfully loaded {len(agents)} agents from markdown files")
        return agents
    
    def load_single_agent(self, file_path: str, **kwargs) -> Agent:
        """
        Convenience method for loading a single agent.
        Backwards compatible with claude code sub agents markdown.
        
        Args:
            file_path: Path to markdown file
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured Agent instance
        """
        return self.load_agent_from_markdown(file_path, **kwargs)
    
    def load_multiple_agents(self, file_paths: Union[str, List[str]], **kwargs) -> List[Agent]:
        """
        Convenience method for loading multiple agents.
        Backwards compatible with claude code sub agents markdown.
        
        Args:
            file_paths: Directory path or list of file paths
            **kwargs: Additional configuration overrides
            
        Returns:
            List of configured Agent instances
        """
        return self.load_agents_from_markdown(file_paths, **kwargs)


# Convenience functions for backwards compatibility
def load_agent_from_markdown(file_path: str, **kwargs) -> Agent:
    """
    Load a single agent from a markdown file.
    
    Args:
        file_path: Path to markdown file
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured Agent instance
    """
    loader = AgentLoader()
    return loader.load_single_agent(file_path, **kwargs)


def load_agents_from_markdown(file_paths: Union[str, List[str]], **kwargs) -> List[Agent]:
    """
    Load multiple agents from markdown files.
    
    Args:
        file_paths: Directory path or list of file paths
        **kwargs: Additional configuration overrides
        
    Returns:
        List of configured Agent instances
    """
    loader = AgentLoader()
    return loader.load_multiple_agents(file_paths, **kwargs)