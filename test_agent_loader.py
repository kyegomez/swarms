"""
Test script for the AgentLoader functionality.
This tests the core functionality without requiring external models.
"""

import os
import tempfile
from pathlib import Path
import sys

# Add swarms to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from swarms.utils.agent_loader import AgentLoader, MarkdownAgentConfig

def test_markdown_parsing():
    """Test markdown parsing functionality."""
    print("Testing markdown parsing...")
    
    # Create a sample markdown content
    sample_content = """| name | description | model |
|------|-------------|-------|
| test-agent | A test agent for validation | gpt-4 |

## Focus Areas
- Testing functionality
- Validating implementation
- Ensuring compatibility

## Approach
1. Parse markdown structure
2. Extract configuration data
3. Validate parsed results
4. Create agent instance

## Output
- Test results
- Validation reports
- Configuration summary
"""
    
    # Test parsing functionality
    loader = AgentLoader()
    
    # Test table parsing
    table_data = loader.parse_markdown_table(sample_content)
    assert table_data['name'] == 'test-agent'
    assert table_data['description'] == 'A test agent for validation'
    assert table_data['model_name'] == 'gpt-4'
    print("[OK] Table parsing successful")
    
    # Test section extraction
    sections = loader.extract_sections(sample_content)
    assert 'focus areas' in sections
    assert len(sections['focus areas']) == 3
    assert 'approach' in sections
    assert len(sections['approach']) == 4
    print("[OK] Section extraction successful")
    
    # Test system prompt building
    config_data = {
        'description': table_data['description'],
        'focus_areas': sections['focus areas'],
        'approach': sections['approach'],
        'output': sections.get('output', [])
    }
    system_prompt = loader.build_system_prompt(config_data)
    assert 'Role:' in system_prompt
    assert 'Focus Areas:' in system_prompt
    assert 'Approach:' in system_prompt
    print("[OK] System prompt building successful")
    
    print("Markdown parsing tests passed!")
    return True

def test_file_operations():
    """Test file loading operations."""
    print("\\nTesting file operations...")
    
    # Create temporary markdown file
    sample_content = """| name | description | model |
|------|-------------|-------|
| file-test-agent | Agent created from file | gpt-4 |

## Focus Areas
- File processing
- Configuration validation

## Approach
1. Load from file
2. Parse content
3. Create configuration

## Output
- Loaded agent
- Configuration object
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        temp_file = f.name
    
    try:
        loader = AgentLoader()
        
        # Test file parsing
        config = loader.parse_markdown_file(temp_file)
        assert isinstance(config, MarkdownAgentConfig)
        assert config.name == 'file-test-agent'
        assert config.description == 'Agent created from file'
        print("[OK] File parsing successful")
        
        # Test configuration validation
        assert len(config.focus_areas) == 2
        assert len(config.approach) == 3
        assert config.system_prompt is not None
        print("[OK] Configuration validation successful")
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    print("File operations tests passed!")
    return True

def test_multiple_files():
    """Test loading multiple files."""
    print("\\nTesting multiple file loading...")
    
    # Create multiple temporary files
    files = []
    for i in range(3):
        content = f"""| name | description | model |
|------|-------------|-------|
| agent-{i} | Test agent number {i} | gpt-4 |

## Focus Areas
- Multi-agent testing
- Batch processing

## Approach
1. Process multiple files
2. Create agent configurations
3. Return agent list

## Output
- Multiple agents
- Batch results
"""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False)
        temp_file.write(content)
        temp_file.close()
        files.append(temp_file.name)
    
    try:
        loader = AgentLoader()
        
        # Test parsing multiple files
        configs = []
        for file_path in files:
            config = loader.parse_markdown_file(file_path)
            configs.append(config)
        
        assert len(configs) == 3
        for i, config in enumerate(configs):
            assert config.name == f'agent-{i}'
        
        print("[OK] Multiple file parsing successful")
        
    finally:
        # Cleanup
        for file_path in files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    print("Multiple file tests passed!")
    return True

def test_error_handling():
    """Test error handling scenarios."""
    print("\\nTesting error handling...")
    
    loader = AgentLoader()
    
    # Test non-existent file
    try:
        loader.parse_markdown_file("nonexistent.md")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("[OK] FileNotFoundError handling successful")
    
    # Test invalid markdown
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write("Invalid markdown content without proper structure")
        invalid_file = f.name
    
    try:
        # This should not raise an error, but should handle gracefully
        config = loader.parse_markdown_file(invalid_file)
        # Should have defaults
        assert config.name is not None
        print("[OK] Invalid markdown handling successful")
        
    finally:
        if os.path.exists(invalid_file):
            os.remove(invalid_file)
    
    print("Error handling tests passed!")
    return True

def main():
    """Run all tests."""
    print("=== AgentLoader Test Suite ===")
    
    tests = [
        test_markdown_parsing,
        test_file_operations,
        test_multiple_files,
        test_error_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"[FAIL] Test {test.__name__} failed: {e}")
    
    print(f"\\n=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("All tests passed!")
        return True
    else:
        print("Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)