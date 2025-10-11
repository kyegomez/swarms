# MCP Tools Bug Fix Test Scripts

This directory contains test scripts to verify the fix for the MCP (Model Context Protocol) tools integration bug.

## Bug Description

**Issue**: `TypeError: object of type 'Function' has no len()`

**Location**: `swarms/utils/litellm_wrapper.py` in the `output_for_tools` method

**Root Cause**: The code was incorrectly trying to call `len()` on a `Function` object instead of checking the length of the `tool_calls` array.

## Test Scripts

### 1. `test_mcp_bug_fix.py` - Simple Bug Fix Test

A focused test script that specifically reproduces the exact scenario from the bug report.

**Features**:
- Tests the specific error scenario that was failing
- Verifies the fix handles both single and multiple tool calls
- Provides clear pass/fail results

**Usage**:
```bash
python test_mcp_bug_fix.py
```

### 2. `test_mcp_tools_example.py` - Comprehensive Test Suite

A comprehensive test suite that covers various aspects of MCP tools integration.

**Features**:
- Basic tool fetching test
- Multiple MCP servers test
- Agent execution with MCP tools
- Error handling scenarios
- Performance testing
- Detailed reporting

**Usage**:
```bash
python test_mcp_tools_example.py
```

## Prerequisites

Before running the tests, you need to start the MCP server:

1. **Start the OKX Crypto Server**:
   ```bash
   python examples/mcp/multi_mcp_guide/okx_crypto_server.py
   ```
   
   This will start the server on `http://0.0.0.0:8001/mcp`

2. **Install Required Dependencies**:
   ```bash
   pip install swarms mcp fastmcp requests
   ```

## Expected Results

### Before the Fix
- ❌ `TypeError: object of type 'Function' has no len()`
- ❌ Agent execution fails when using MCP tools
- ❌ MCP tool calls cannot be processed

### After the Fix
- ✅ Tools are fetched successfully
- ✅ Agent can execute tasks using MCP tools
- ✅ Both single and multiple tool calls work correctly
- ✅ No TypeError occurs

## Test Scenarios

### Basic Functionality
1. **Tool Fetching**: Verify MCP tools can be retrieved from the server
2. **Agent Creation**: Verify agents can be created with MCP tool integration
3. **Tool Execution**: Verify agents can execute tasks that use MCP tools

### Error Handling
1. **Invalid Server URL**: Test behavior with non-existent server
2. **Invalid Authentication**: Test behavior with wrong credentials
3. **Network Timeouts**: Test behavior with connection timeouts

### Edge Cases
1. **Single Tool Call**: Verify single tool call processing
2. **Multiple Tool Calls**: Verify multiple tool call processing
3. **Empty Responses**: Test behavior with empty tool responses

## Sample Output

### Successful Test Run
```
🚀 MCP Bug Fix Test
This test verifies the fix for the TypeError in MCP tool usage.
Make sure the OKX crypto server is running on port 8001.

🐛 Testing MCP Bug Fix
========================================
1. Fetching MCP tools...
   ✅ Successfully fetched 2 tools
2. Creating agent with MCP tools...
   ✅ Agent created successfully
3. Running task with MCP tools...
   Task: Get Bitcoin trading volume using get_okx_crypto_volume tool
   ✅ Task completed successfully!
   Result: [Tool execution result with Bitcoin volume data]

========================================
📊 TEST SUMMARY
========================================
✅ Main bug fix: PASSED
   The TypeError: object of type 'Function' has no len() is fixed!
✅ Multiple tool calls: PASSED

🎉 ALL TESTS PASSED!
The MCP tools integration is working correctly.
```

## Troubleshooting

### Common Issues

1. **Server Not Running**:
   ```
   Error: Connection refused
   Solution: Start the OKX crypto server first
   ```

2. **Port Already in Use**:
   ```
   Error: Address already in use
   Solution: Change the port in the server script or kill existing processes
   ```

3. **Authentication Error**:
   ```
   Error: 401 Unauthorized
   Solution: Check the Authorization header in the connection
   ```

### Debug Mode

To get more detailed output, you can modify the test scripts to enable verbose logging:

```python
# In the test scripts, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Code Changes Made

The fix involved modifying the `output_for_tools` method in `swarms/utils/litellm_wrapper.py`:

**Before** (buggy code):
```python
if self.mcp_call is True:
    out = response.choices[0].message.tool_calls[0].function
    if len(out) > 1:  # ❌ Error: Function objects don't have len()
        return out
    else:
        out = out[0]
```

**After** (fixed code):
```python
if self.mcp_call is True:
    tool_calls = response.choices[0].message.tool_calls
    if len(tool_calls) > 1:  # ✅ Correct: Check tool_calls length
        # Handle multiple tool calls
        return [...]
    else:
        # Handle single tool call
        out = tool_calls[0].function
```

## Contributing

If you find any issues with these test scripts or the MCP tools integration, please:

1. Run the test scripts to reproduce the issue
2. Check the server logs for additional error information
3. Report the issue with the test output and error details
4. Include your environment details (Python version, OS, etc.)
