import sys
from swarms import Agent, ConcurrentWorkflow
import os
import pathlib
from pathlib import Path
from dotenv import load_dotenv
import asyncio
import json
from datetime import datetime
import math

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))  

def create_riemann_hypothesis_agents():
    """Create specialized agents for Riemann Hypothesis proof."""
    
    # Agent 1: Mathematical Analysis Agent
    math_analysis_agent = Agent(
        agent_name="Riemann-Math-Analysis-Agent",
        system_prompt="""You are a specialized mathematical analysis agent focused on the Riemann Hypothesis. Your mission is to:

1. **Understand the Riemann Hypothesis**: ζ(s) = 0 has non-trivial zeros only at s = 1/2 + it for real t
2. **Analyze the Zeta Function**: ζ(s) = Σ(n=1 to ∞) 1/n^s for Re(s) > 1
3. **Calculate Critical Values**: Use MCP tools to compute ζ(1/2 + it) for various t values
4. **Verify Zeros**: Check if ζ(1/2 + it) = 0 for specific t values
5. **Analyze Patterns**: Look for patterns in the distribution of zeros

CRITICAL: You MUST use MCP tools for all mathematical calculations. Focus on:
- Computing ζ function values
- Analyzing the critical line Re(s) = 1/2
- Checking for non-trivial zeros
- Statistical analysis of zero distributions

Use precise mathematical reasoning and provide detailed analysis of each calculation.""",
        model_name="gpt-4o-mini",
        streaming_on=True,
        print_on=True,
        max_loops=10,
        error_handling="continue",
        tool_choice="auto",
        verbose=True,
        mcp_url="stdio://examples/mcp/working_mcp_server.py",
    )
    
    # Agent 2: Computational Verification Agent
    computational_agent = Agent(
        agent_name="Riemann-Computational-Agent",
        system_prompt="""You are a computational verification agent for the Riemann Hypothesis. Your mission is to:

1. **Numerical Verification**: Use MCP tools to compute ζ function values numerically
2. **Zero Detection**: Identify when ζ(s) ≈ 0 within computational precision
3. **Critical Line Analysis**: Focus on s = 1/2 + it values
4. **Statistical Testing**: Analyze the distribution of computed zeros
5. **Error Analysis**: Assess computational accuracy and error bounds

CRITICAL: You MUST use MCP tools for all computations. Focus on:
- High-precision calculations
- Error estimation
- Statistical analysis of results
- Verification of known zeros
- Discovery of new patterns

Provide detailed computational analysis with error bounds and confidence intervals.""",
        model_name="gpt-4o-mini",
        streaming_on=True,
        print_on=True,
        max_loops=10,
        error_handling="continue",
        tool_choice="auto",
        verbose=True,
        mcp_url="stdio://examples/mcp/working_mcp_server.py",
    )
    
    # Agent 3: Proof Strategy Agent
    proof_strategy_agent = Agent(
        agent_name="Riemann-Proof-Strategy-Agent",
        system_prompt="""You are a mathematical proof strategy agent for the Riemann Hypothesis. Your mission is to:

1. **Proof Strategy Development**: Develop systematic approaches to prove RH
2. **Analytic Continuation**: Analyze ζ(s) beyond Re(s) > 1
3. **Functional Equation**: Use ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
4. **Zero-Free Regions**: Identify regions where ζ(s) ≠ 0
5. **Contradiction Methods**: Use proof by contradiction approaches

CRITICAL: You MUST use MCP tools for all mathematical operations. Focus on:
- Functional equation analysis
- Contour integration methods
- Analytic number theory techniques
- Complex analysis applications
- Proof strategy validation

Develop rigorous mathematical arguments and validate each step computationally.""",
        model_name="gpt-4o-mini",
        streaming_on=True,
        print_on=True,
        max_loops=10,
        error_handling="continue",
        tool_choice="auto",
        verbose=True,
        mcp_url="stdio://examples/mcp/working_mcp_server.py",
    )
    
    # Agent 4: Historical Analysis Agent
    historical_agent = Agent(
        agent_name="Riemann-Historical-Analysis-Agent",
        system_prompt="""You are a historical analysis agent for the Riemann Hypothesis. Your mission is to:

1. **Historical Context**: Analyze previous attempts to prove RH
2. **Known Results**: Review verified properties of ζ function
3. **Computational History**: Study previous numerical verifications
4. **Failed Approaches**: Learn from unsuccessful proof attempts
5. **Modern Techniques**: Apply contemporary mathematical methods

CRITICAL: You MUST use MCP tools for all calculations. Focus on:
- Historical verification of known results
- Analysis of previous computational efforts
- Statistical analysis of historical data
- Pattern recognition across different approaches
- Synthesis of historical insights

Provide comprehensive analysis of historical context and its relevance to current proof attempts.""",
        model_name="gpt-4o-mini",
        streaming_on=True,
        print_on=True,
        max_loops=10,
        error_handling="continue",
        tool_choice="auto",
        verbose=True,
        mcp_url="stdio://examples/mcp/working_mcp_server.py",
    )
    
    return [math_analysis_agent, computational_agent, proof_strategy_agent, historical_agent]

def create_riemann_workflow():
    """Create a comprehensive Riemann Hypothesis workflow."""
    
    # Create the specialized agents
    agents = create_riemann_hypothesis_agents()
    
    # Create concurrent workflow
    workflow = ConcurrentWorkflow(
        name="Riemann-Hypothesis-Proof-Attempt",
        agents=agents,
        show_dashboard=True,
        auto_save=True,
        output_type="dict",
        max_loops=5,
        auto_generate_prompts=False,
    )
    
    return workflow

def riemann_hypothesis_proof_attempt():
    """Main function to attempt Riemann Hypothesis proof."""
    
    print(" RIEMANN HYPOTHESIS PROOF ATTEMPT")
    print("=" * 80)
    print("Mission: Calculate and prove the Riemann Hypothesis")
    print("Hypothesis: All non-trivial zeros of ζ(s) have Re(s) = 1/2")
    print("=" * 80)
    
    # Create workflow
    workflow = create_riemann_workflow()
    
    # Comprehensive Riemann Hypothesis task
    riemann_task = """
    MATHEMATICAL MISSION: PROVE THE RIEMANN HYPOTHESIS
    
    The Riemann Hypothesis states that all non-trivial zeros of the Riemann zeta function ζ(s) 
    have real part equal to 1/2. This is one of the most important unsolved problems in mathematics.
    
    TASK BREAKDOWN:
    
    1. **Riemann-Math-Analysis-Agent**: 
       - Define and analyze the Riemann zeta function ζ(s) = Σ(n=1 to ∞) 1/n^s
       - Understand the critical line Re(s) = 1/2
       - Analyze the functional equation: ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s)
       - Use MCP tools to compute ζ function values
       - Identify patterns in zero distribution
    
    2. **Riemann-Computational-Agent**:
       - Perform high-precision numerical calculations of ζ(1/2 + it)
       - Verify known zeros (first few: t ≈ 14.1347, 21.0220, 25.0109, 30.4249, 32.9351)
       - Compute ζ function values for various t values
       - Analyze statistical distribution of computed values
       - Provide error analysis and confidence intervals
    
    3. **Riemann-Proof-Strategy-Agent**:
       - Develop systematic proof strategies
       - Use analytic continuation techniques
       - Apply complex analysis methods
       - Explore proof by contradiction approaches
       - Validate each mathematical step computationally
    
    4. **Riemann-Historical-Analysis-Agent**:
       - Review historical attempts and known results
       - Analyze computational verification history
       - Study failed proof approaches
       - Apply modern mathematical techniques
       - Synthesize historical insights with current methods
    
    MATHEMATICAL REQUIREMENTS:
    - Use MCP tools for ALL calculations
    - Provide rigorous mathematical reasoning
    - Include error analysis and confidence intervals
    - Verify each step computationally
    - Document all assumptions and limitations
    
    EXPECTED DELIVERABLES:
    - Comprehensive analysis of ζ function behavior
    - Numerical verification of known zeros
    - Statistical analysis of zero distribution
    - Proof strategy development
    - Historical context and insights
    - Computational evidence supporting or refuting RH
    
    CRITICAL: This is an attempt to contribute to one of mathematics' greatest unsolved problems.
    Provide thorough, rigorous analysis with full computational validation.
    """
    
    print(" Executing Riemann Hypothesis Proof Attempt...")
    print("=" * 80)
    
    # Execute the workflow
    results = workflow.run(riemann_task)
    
    print("\n" + "=" * 80)
    print("Riemann Hypothesis Proof Attempt Complete!")
    print("=" * 80)
    
    # Display results
    if isinstance(results, dict):
        for agent_name, result in results.items():
            print(f"\n {agent_name}:")
            if isinstance(result, str):
                print(f" Result: {result[:500]}...")
            else:
                print(f" Result: {str(result)[:500]}...")
            print("-" * 60)
    elif isinstance(results, list):
        for i, result in enumerate(results):
            print(f"\n Agent {i+1}:")
            if isinstance(result, str):
                print(f" Result: {result[:500]}...")
            else:
                print(f" Result: {str(result)[:500]}...")
            print("-" * 60)
    else:
        print(f"\n Results: {results}")
    
    return results

def test_individual_riemann_agent():
    """Test a single Riemann Hypothesis agent."""
    
    print(" Testing Individual Riemann Hypothesis Agent...")
    print("=" * 80)
    
    # Create a single agent for focused analysis
    riemann_agent = Agent(
        agent_name="Riemann-Single-Analysis-Agent",
        system_prompt="""You are a specialized Riemann Hypothesis analysis agent. Your mission is to:

1. **Define the Problem**: ζ(s) = Σ(n=1 to ∞) 1/n^s, find all s where ζ(s) = 0
2. **Critical Line Focus**: Analyze s = 1/2 + it (the critical line)
3. **Numerical Verification**: Use MCP tools to compute ζ(1/2 + it) values
4. **Zero Detection**: Identify when |ζ(1/2 + it)| ≈ 0
5. **Statistical Analysis**: Analyze patterns in zero distribution

CRITICAL: You MUST use the specific mathematical MCP tools for ALL calculations:
- Use 'compute_zeta' tool to calculate ζ function values (NOT the calculate tool)
- Use 'find_zeta_zeros' tool to search for zeros of the zeta function
- Use 'complex_math' tool for complex number operations
- Use 'statistical_analysis' tool for data analysis

DO NOT use the simple 'calculate' tool. Use the specialized mathematical tools.
Focus on:
- Computing ζ function values numerically using compute_zeta
- Analyzing the first few known zeros using find_zeta_zeros
- Statistical analysis of results using statistical_analysis
- Error estimation and confidence intervals
- Pattern recognition in zero distribution

Provide detailed mathematical analysis with full computational validation using the proper mathematical tools.""",
        model_name="gpt-4o-mini",
        streaming_on=True,
        print_on=True,
        max_loops=8,
        error_handling="continue",
        tool_choice="auto",
        verbose=True,
        mcp_url="stdio://examples/mcp/working_mcp_server.py",
    )
    
    # Test task focused on Riemann Hypothesis
    test_task = """
    RIEMANN HYPOTHESIS ANALYSIS TASK:
    
    The Riemann Hypothesis states that all non-trivial zeros of ζ(s) have Re(s) = 1/2.
    
    CRITICAL INSTRUCTIONS: You MUST use the EXACT mathematical MCP tools listed below:
    
    TOOL 1: compute_zeta
    - Purpose: Calculate Riemann zeta function values
    - Parameters: real_part (number), imaginary_part (number), precision (integer)
    - Example: compute_zeta with real_part=0.5, imaginary_part=14.1347, precision=1000
    
    TOOL 2: find_zeta_zeros  
    - Purpose: Find zeros of the zeta function
    - Parameters: start_t (number), end_t (number), step_size (number), tolerance (number)
    - Example: find_zeta_zeros with start_t=0.0, end_t=50.0, step_size=0.1, tolerance=0.001
    
    TOOL 3: complex_math
    - Purpose: Perform complex mathematical operations
    - Parameters: operation (string), real1 (number), imag1 (number)
    - Example: complex_math with operation="exp", real1=0.0, imag1=3.14159
    
    TOOL 4: statistical_analysis
    - Purpose: Analyze data statistically
    - Parameters: data (array), analysis_type (string)
    - Example: statistical_analysis with data=[1,2,3,4,5], analysis_type="descriptive"
    
    DO NOT use the 'calculate' tool. Use ONLY the tools listed above.
    
    REQUIRED ACTIONS (execute these in order):
    
    1. Use compute_zeta tool to calculate ζ(1/2 + 14.1347i)
    2. Use compute_zeta tool to calculate ζ(1/2 + 21.0220i)  
    3. Use compute_zeta tool to calculate ζ(1/2 + 25.0109i)
    4. Use find_zeta_zeros tool to search for zeros in range [0, 50]
    5. Use statistical_analysis tool to analyze the results
    
    Execute these actions using the EXACT tool names and parameters shown above.
    """
    
    print(" Executing Riemann Hypothesis Analysis...")
    result = riemann_agent.run(test_task)
    
    print("\n" + "=" * 80)
    print(" Individual Riemann Agent Test Complete!")
    print("=" * 80)
    print(f" Result: {result}")
    
    return result

def riemann_hypothesis_main():
    """Main function to run the Riemann Hypothesis proof attempt."""
    
    print(" RIEMANN HYPOTHESIS PROOF ATTEMPT")
    print("=" * 80)
    print("This is an attempt to contribute to one of mathematics' greatest unsolved problems.")
    print("The Riemann Hypothesis: All non-trivial zeros of ζ(s) have Re(s) = 1/2")
    print("=" * 80)
    
    try:
        # Test individual agent first
        print("\n1️ Testing Individual Riemann Agent...")
        individual_result = test_individual_riemann_agent()
        
        # Then test the full workflow
        print("\n2️ Testing Full Riemann Hypothesis Workflow...")
        workflow_results = riemann_hypothesis_proof_attempt()
        
        # Summary
        print("\n" + "=" * 80)
        print(" Riemann Hypothesis Proof Attempt Summary")
        print("=" * 80)
        print(" Individual Agent Test: COMPLETED")
        print(" Full Workflow Test: COMPLETED")
        print(" Mathematical Analysis: PERFORMED")
        print(" Computational Verification: EXECUTED")
        print(" Proof Strategy: DEVELOPED")
        print(" Historical Analysis: CONDUCTED")
        print("=" * 80)
        print(" Note: This is a computational exploration of the Riemann Hypothesis.")
        print(" The actual proof remains one of mathematics' greatest challenges.")
        print("=" * 80)
        
        return {
            "individual_result": str(individual_result) if individual_result else "No result",
            "workflow_results": str(workflow_results) if workflow_results else "No results",
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "mathematical_mission": "Riemann Hypothesis Analysis",
            "note": "This is a computational exploration, not a formal proof"
        }
        
    except Exception as e:
        print(f"\n Error during Riemann Hypothesis analysis: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            "error": str(e),
            "status": "FAILED",
            "timestamp": datetime.now().isoformat(),
            "mathematical_mission": "Riemann Hypothesis Analysis"
        }

if __name__ == "__main__":
    result = riemann_hypothesis_main()
    print(f"\n Final Results: {json.dumps(result, indent=2)}")
