#!/usr/bin/env python3
"""
Test script to verify Riemann Hypothesis mathematical tools are working.
"""

import subprocess
import json
import sys

def test_mcp_tool(tool_name, arguments):
    """Test a specific MCP tool."""
    # Create the JSON-RPC request
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        }
    }
    
    # Start the MCP server
    process = subprocess.Popen(
        [sys.executable, "examples/mcp/working_mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Send the request
        request_json = json.dumps(request) + "\n"
        stdout, stderr = process.communicate(input=request_json, timeout=10)
        
        # Parse the response
        response = json.loads(stdout.strip())
        
        if "result" in response:
            return response["result"]["content"][0]["text"]
        else:
            return f"Error: {response.get('error', 'Unknown error')}"
            
    except subprocess.TimeoutExpired:
        process.kill()
        return "Error: Timeout"
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("Testing Riemann Hypothesis Mathematical Tools")
    print("=" * 60)
    
    # Test 1: Compute zeta function at known zero
    print("\n1. Testing compute_zeta tool:")
    result1 = test_mcp_tool("compute_zeta", {
        "real_part": 0.5,
        "imaginary_part": 14.1347,
        "precision": 1000
    })
    print(f"ζ(1/2 + 14.1347i) = {result1}")
    
    # Test 2: Find zeros in a range
    print("\n2. Testing find_zeta_zeros tool:")
    result2 = test_mcp_tool("find_zeta_zeros", {
        "start_t": 0.0,
        "end_t": 30.0,
        "step_size": 0.1,
        "tolerance": 0.001
    })
    print(f"Zeros found: {result2}")
    
    # Test 3: Complex math operations
    print("\n3. Testing complex_math tool:")
    result3 = test_mcp_tool("complex_math", {
        "operation": "exp",
        "real1": 0.0,
        "imag1": 3.14159
    })
    print(f"exp(iπ) = {result3}")
    
    # Test 4: Statistical analysis
    print("\n4. Testing statistical_analysis tool:")
    result4 = test_mcp_tool("statistical_analysis", {
        "data": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "analysis_type": "descriptive"
    })
    print(f"Statistical analysis: {result4}")
    
    print("\n" + "=" * 60)
    print("Mathematical tools test complete!")

if __name__ == "__main__":
    main() 
