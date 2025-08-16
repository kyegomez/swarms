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
    
    # Example 1: Create sample markdown files for testing - Claude Code format
    
    # Performance Engineer agent
    performance_md = """---
name: performance-engineer
description: Optimize application performance and identify bottlenecks
model_name: gpt-4
temperature: 0.3
max_loops: 2
mcp_url: http://example.com/mcp
---

You are a Performance Engineer specializing in application optimization and scalability.

Your role involves:
- Analyzing application architecture and identifying potential bottlenecks
- Implementing comprehensive monitoring and logging
- Conducting performance testing under various load conditions
- Optimizing critical paths and resource usage
- Documenting findings and providing actionable recommendations

Expected output:
- Performance analysis reports with specific metrics
- Optimized code recommendations
- Infrastructure scaling suggestions
- Monitoring and alerting setup guidelines
"""
    
    # Data Analyst agent
    data_analyst_md = """---
name: data-analyst
description: Analyze data and provide business insights
model_name: gpt-4
temperature: 0.2
max_loops: 1
---

You are a Data Analyst specializing in extracting insights from complex datasets.

Your responsibilities include:
- Collecting and cleaning data from various sources
- Performing exploratory data analysis and statistical modeling
- Creating compelling visualizations and interactive dashboards
- Applying statistical methods and machine learning techniques
- Presenting findings and actionable business recommendations

Focus on providing data-driven insights that support strategic decision making.
"""
    
    # Create sample markdown files
    performance_file = "performance_engineer.md"  
    data_file = "data_analyst.md"
    
    with open(performance_file, 'w') as f:
        f.write(performance_md)
    
    with open(data_file, 'w') as f:
        f.write(data_analyst_md)
    
    try:
        # Example 2: Load Performance Engineer agent
        print("\\n1. Loading Performance Engineer agent (YAML frontmatter):")
        perf_agent = loader.load_single_agent(performance_file)
        print(f"   Loaded agent: {perf_agent.agent_name}")
        print(f"   Model: {perf_agent.model_name}")
        print(f"   Temperature: {getattr(perf_agent, 'temperature', 'Not set')}")
        print(f"   Max loops: {perf_agent.max_loops}")
        print(f"   System prompt preview: {perf_agent.system_prompt[:100]}...")
        
        # Example 3: Load Data Analyst agent
        print("\\n2. Loading Data Analyst agent:")
        data_agent = loader.load_single_agent(data_file)
        print(f"   Loaded agent: {data_agent.agent_name}")
        print(f"   Temperature: {getattr(data_agent, 'temperature', 'Not set')}")
        print(f"   System prompt preview: {data_agent.system_prompt[:100]}...")
        
        # Example 4: Load single agent using convenience function
        print("\\n3. Loading single agent using convenience function:")
        agent2 = load_agent_from_markdown(performance_file)
        print(f"   Loaded agent: {agent2.agent_name}")
        
        # Example 5: Load multiple agents (from directory or list)
        print("\\n4. Loading multiple agents:")
        
        # Create another sample file - Security Analyst
        security_md = """---
name: security-analyst
description: Analyze and improve system security
model_name: gpt-4
temperature: 0.1
max_loops: 3
---

You are a Security Analyst specializing in cybersecurity assessment and protection.

Your expertise includes:
- Conducting comprehensive security vulnerability assessments
- Performing detailed code security reviews and penetration testing
- Implementing robust infrastructure hardening measures
- Developing incident response and recovery procedures

Key methodology:
1. Conduct thorough security audits across all system components
2. Identify and classify potential vulnerabilities and threats
3. Recommend and implement security improvements and controls
4. Develop comprehensive security policies and best practices
5. Monitor and respond to security incidents

Provide detailed security reports with specific remediation steps and risk assessments.
"""
        
        security_file = "security_analyst.md"
        with open(security_file, 'w') as f:
            f.write(security_md)
        
        # Load multiple agents from list
        agents = loader.load_multiple_agents([performance_file, data_file, security_file])
        print(f"   Loaded {len(agents)} agents:")
        for agent in agents:
            temp_attr = getattr(agent, 'temperature', 'default')
            print(f"   - {agent.agent_name} (temp: {temp_attr})")
        
        # Example 6: Load agents from directory (current directory)
        print("\\n5. Loading agents from current directory:")
        current_dir_agents = load_agents_from_markdown(".")
        print(f"   Found {len(current_dir_agents)} agents in current directory")
        
        # Example 7: Demonstrate error handling
        print("\\n6. Error handling demo:")
        try:
            loader.load_single_agent("nonexistent.md")
        except FileNotFoundError as e:
            print(f"   Caught expected error: {e}")
        
        # Example 8: Test agent functionality  
        print("\\n7. Testing loaded agent functionality:")
        test_agent = agents[0]
        print(f"   Agent: {test_agent.agent_name}")
        print(f"   Temperature: {getattr(test_agent, 'temperature', 'default')}")
        print(f"   Max loops: {test_agent.max_loops}")
        print(f"   Ready for task execution")
        
    except Exception as e:
        print(f"Error during demo: {e}")
    
    finally:
        # Cleanup sample files
        for file in [performance_file, data_file, security_file]:
            if os.path.exists(file):
                os.remove(file)
        print("\\n   Cleaned up sample files")
    
    print("\\n=== Demo Complete ===")

if __name__ == "__main__":
    main()