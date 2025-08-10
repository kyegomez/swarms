"""
Example demonstrating the AgentLoader for loading agents from markdown files.

This example shows:
1. Loading a single agent from a markdown file
2. Loading multiple agents from markdown files
3. Using the convenience functions
4. Error handling and validation
"""

import os
from swarms.utils.agent_loader import AgentLoader, load_agent_from_markdown, load_agents_from_markdown

def main():
    # Initialize the loader
    loader = AgentLoader()
    
    print("=== AgentLoader Demo ===")
    
    # Example 1: Create a sample markdown file for testing
    sample_md = """| name | description | model |
|------|-------------|-------|
| performance-engineer | Optimize application performance and identify bottlenecks | gpt-4 |

## Focus Areas
- Application profiling and performance analysis
- Database optimization and query tuning
- Memory and CPU usage optimization
- Load testing and capacity planning

## Approach
1. Analyze application architecture and identify potential bottlenecks
2. Implement comprehensive monitoring and logging
3. Conduct performance testing under various load conditions
4. Optimize critical paths and resource usage
5. Document findings and provide actionable recommendations

## Output
- Performance analysis reports with specific metrics
- Optimized code recommendations
- Infrastructure scaling suggestions
- Monitoring and alerting setup guidelines
"""
    
    # Create sample markdown file
    sample_file = "sample_agent.md"
    with open(sample_file, 'w') as f:
        f.write(sample_md)
    
    try:
        # Example 2: Load single agent using class method
        print("\\n1. Loading single agent using AgentLoader class:")
        agent = loader.load_single_agent(sample_file)
        print(f"   Loaded agent: {agent.agent_name}")
        print(f"   System prompt preview: {agent.system_prompt[:100]}...")
        
        # Example 3: Load single agent using convenience function
        print("\\n2. Loading single agent using convenience function:")
        agent2 = load_agent_from_markdown(sample_file)
        print(f"   Loaded agent: {agent2.agent_name}")
        
        # Example 4: Load multiple agents (from directory or list)
        print("\\n3. Loading multiple agents:")
        
        # Create another sample file
        sample_md2 = """| name | description | model |
|------|-------------|-------|
| security-analyst | Analyze and improve system security | gpt-4 |

## Focus Areas
- Security vulnerability assessment
- Code security review
- Infrastructure hardening

## Approach
1. Conduct thorough security audits
2. Identify potential vulnerabilities
3. Recommend security improvements

## Output
- Security assessment reports
- Vulnerability remediation plans
- Security best practices documentation
"""
        
        sample_file2 = "security_agent.md"
        with open(sample_file2, 'w') as f:
            f.write(sample_md2)
        
        # Load multiple agents from list
        agents = loader.load_multiple_agents([sample_file, sample_file2])
        print(f"   Loaded {len(agents)} agents:")
        for agent in agents:
            print(f"   - {agent.agent_name}")
        
        # Example 5: Load agents from directory (current directory)
        print("\\n4. Loading agents from current directory:")
        current_dir_agents = load_agents_from_markdown(".")
        print(f"   Found {len(current_dir_agents)} agents in current directory")
        
        # Example 6: Demonstrate error handling
        print("\\n5. Error handling demo:")
        try:
            loader.load_single_agent("nonexistent.md")
        except FileNotFoundError as e:
            print(f"   Caught expected error: {e}")
        
        # Example 7: Test agent functionality
        print("\\n6. Testing loaded agent functionality:")
        test_agent = agents[0]
        response = test_agent.run("What are the key steps for performance optimization?")
        print(f"   Agent response preview: {str(response)[:150]}...")
        
    except Exception as e:
        print(f"Error during demo: {e}")
    
    finally:
        # Cleanup sample files
        for file in [sample_file, sample_file2]:
            if os.path.exists(file):
                os.remove(file)
        print("\\n   Cleaned up sample files")
    
    print("\\n=== Demo Complete ===")

if __name__ == "__main__":
    main()