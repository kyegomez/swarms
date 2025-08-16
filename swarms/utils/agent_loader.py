import os
import re
import random
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from swarms.structs.agent import Agent


class MarkdownAgentConfig(BaseModel):
    """Configuration model for agents loaded from Claude Code markdown files."""
    name: str
    description: str
    model_name: Optional[str] = "gpt-4"
    temperature: Optional[float] = Field(default=0.1, ge=0.0, le=2.0)
    mcp_url: Optional[str] = None
    system_prompt: str
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
    Loader for creating agents from markdown files using Claude Code sub-agent format.
    
    Supports both single markdown file and multiple markdown files.
    Uses YAML frontmatter format for agent configuration.
    
    Features:
    - Single markdown file loading
    - Multiple markdown files loading (batch processing)
    - YAML frontmatter parsing
    - Agent configuration extraction from YAML metadata
    - Error handling and validation
    """
    
    def __init__(self):
        """
        Initialize the AgentLoader.
        """
        pass
        
    def parse_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML frontmatter from markdown content.
        
        Args:
            content: Markdown content with potential YAML frontmatter
            
        Returns:
            Dictionary with parsed YAML data and remaining content
        """
        lines = content.split('\n')
        
        # Check if content starts with YAML frontmatter
        if not lines[0].strip() == '---':
            return {"frontmatter": {}, "content": content}
        
        # Find end of frontmatter
        end_marker = -1
        for i, line in enumerate(lines[1:], 1):
            if line.strip() == '---':
                end_marker = i
                break
        
        if end_marker == -1:
            return {"frontmatter": {}, "content": content}
        
        # Extract frontmatter and content
        frontmatter_text = '\n'.join(lines[1:end_marker])
        remaining_content = '\n'.join(lines[end_marker + 1:]).strip()
        
        try:
            frontmatter_data = yaml.safe_load(frontmatter_text) or {}
        except yaml.YAMLError as e:
            logger.warning(f"Failed to parse YAML frontmatter: {e}")
            return {"frontmatter": {}, "content": content}
        
        return {"frontmatter": frontmatter_data, "content": remaining_content}

    
    
    
    def parse_markdown_file(self, file_path: str) -> MarkdownAgentConfig:
        """
        Parse a single markdown file to extract agent configuration.
        Uses Claude Code sub-agent YAML frontmatter format.
        
        Args:
            file_path: Path to markdown file
            
        Returns:
            MarkdownAgentConfig object with parsed configuration
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If parsing fails or no YAML frontmatter found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file {file_path} not found.")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse YAML frontmatter (Claude Code sub-agent format)
            yaml_result = self.parse_yaml_frontmatter(content)
            frontmatter = yaml_result["frontmatter"]
            remaining_content = yaml_result["content"]
            
            if not frontmatter:
                raise ValueError(f"No YAML frontmatter found in {file_path}. File must use Claude Code sub-agent format with YAML frontmatter.")
            
            # Use YAML frontmatter data
            config_data = {
                'name': frontmatter.get('name', Path(file_path).stem),
                'description': frontmatter.get('description', 'Agent loaded from markdown'),
                'model_name': frontmatter.get('model_name') or frontmatter.get('model', 'gpt-4'),
                'temperature': frontmatter.get('temperature', 0.1),
                'max_loops': frontmatter.get('max_loops', 1),
                'mcp_url': frontmatter.get('mcp_url'),
                'system_prompt': remaining_content.strip(),
            }
            
            # Generate random model if not specified
            if not config_data['model_name'] or config_data['model_name'] == 'random':
                models = ['gpt-4', 'gpt-4-turbo', 'claude-3-sonnet', 'claude-3-haiku']
                config_data['model_name'] = random.choice(models)
            
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
            'temperature': config_dict.get('temperature', 0.1),
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
        Uses Claude Code sub-agent YAML frontmatter format.
        
        Args:
            file_path: Path to markdown file with YAML frontmatter
            **kwargs: Additional configuration overrides
            
        Returns:
            Configured Agent instance
        """
        return self.load_agent_from_markdown(file_path, **kwargs)
    
    def load_multiple_agents(self, file_paths: Union[str, List[str]], **kwargs) -> List[Agent]:
        """
        Convenience method for loading multiple agents.
        Uses Claude Code sub-agent YAML frontmatter format.
        
        Args:
            file_paths: Directory path or list of file paths with YAML frontmatter
            **kwargs: Additional configuration overrides
            
        Returns:
            List of configured Agent instances
        """
        return self.load_agents_from_markdown(file_paths, **kwargs)


# Convenience functions
def load_agent_from_markdown(file_path: str, **kwargs) -> Agent:
    """
    Load a single agent from a markdown file with Claude Code YAML frontmatter format.
    
    Args:
        file_path: Path to markdown file with YAML frontmatter
        **kwargs: Additional configuration overrides
        
    Returns:
        Configured Agent instance
    """
    loader = AgentLoader()
    return loader.load_single_agent(file_path, **kwargs)


def load_agents_from_markdown(file_paths: Union[str, List[str]], **kwargs) -> List[Agent]:
    """
    Load multiple agents from markdown files with Claude Code YAML frontmatter format.
    
    Args:
        file_paths: Directory path or list of file paths with YAML frontmatter
        **kwargs: Additional configuration overrides
        
    Returns:
        List of configured Agent instances
    """
    loader = AgentLoader()
    return loader.load_multiple_agents(file_paths, **kwargs)