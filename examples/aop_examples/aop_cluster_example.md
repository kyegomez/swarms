# AOP Cluster Example

This example demonstrates how to use AOPCluster to connect to and manage multiple MCP servers running AOP agents.

## Basic Cluster Setup

```python
import json
from swarms.structs.aop import AOPCluster

# Connect to multiple MCP servers
cluster = AOPCluster(
    urls=[
        "http://localhost:8000/mcp",  # Research and Analysis server
        "http://localhost:8001/mcp",  # Writing and Code server
        "http://localhost:8002/mcp"   # Financial server
    ],
    transport="streamable-http"
)

# Get all available tools from all servers
all_tools = cluster.get_tools(output_type="dict")
print(f"Found {len(all_tools)} tools across all servers")

# Pretty print all tools
print(json.dumps(all_tools, indent=2))
```

## Finding Specific Tools

```python
# Find a specific tool by name
research_tool = cluster.find_tool_by_server_name("Research-Agent")
if research_tool:
    print("Found Research-Agent tool:")
    print(json.dumps(research_tool, indent=2))
else:
    print("Research-Agent tool not found")

# Find multiple tools
tool_names = ["Research-Agent", "Analysis-Agent", "Writing-Agent", "Code-Agent"]
found_tools = {}

for tool_name in tool_names:
    tool = cluster.find_tool_by_server_name(tool_name)
    if tool:
        found_tools[tool_name] = tool
        print(f"✓ Found {tool_name}")
    else:
        print(f"✗ {tool_name} not found")

print(f"Found {len(found_tools)} out of {len(tool_names)} tools")
```

## Tool Discovery and Management

```python
# Get tools in different formats
json_tools = cluster.get_tools(output_type="json")
dict_tools = cluster.get_tools(output_type="dict")
str_tools = cluster.get_tools(output_type="str")

print(f"JSON format: {len(json_tools)} tools")
print(f"Dict format: {len(dict_tools)} tools")
print(f"String format: {len(str_tools)} tools")

# Analyze tool distribution across servers
server_tools = {}
for tool in dict_tools:
    server_name = tool.get("server", "unknown")
    if server_name not in server_tools:
        server_tools[server_name] = []
    server_tools[server_name].append(tool.get("function", {}).get("name", "unknown"))

print("\nTools by server:")
for server, tools in server_tools.items():
    print(f"  {server}: {len(tools)} tools - {tools}")
```

## Advanced Cluster Management

```python
class AOPClusterManager:
    def __init__(self, urls, transport="streamable-http"):
        self.cluster = AOPCluster(urls, transport)
        self.tools_cache = {}
        self.last_update = None
    
    def refresh_tools(self):
        """Refresh the tools cache"""
        self.tools_cache = {}
        tools = self.cluster.get_tools(output_type="dict")
        for tool in tools:
            tool_name = tool.get("function", {}).get("name")
            if tool_name:
                self.tools_cache[tool_name] = tool
        self.last_update = time.time()
        return len(self.tools_cache)
    
    def get_tool(self, tool_name):
        """Get a specific tool by name"""
        if not self.tools_cache or time.time() - self.last_update > 300:  # 5 min cache
            self.refresh_tools()
        return self.tools_cache.get(tool_name)
    
    def list_tools_by_category(self):
        """Categorize tools by their names"""
        categories = {
            "research": [],
            "analysis": [],
            "writing": [],
            "code": [],
            "financial": [],
            "other": []
        }
        
        for tool_name in self.tools_cache.keys():
            tool_name_lower = tool_name.lower()
            if "research" in tool_name_lower:
                categories["research"].append(tool_name)
            elif "analysis" in tool_name_lower:
                categories["analysis"].append(tool_name)
            elif "writing" in tool_name_lower:
                categories["writing"].append(tool_name)
            elif "code" in tool_name_lower:
                categories["code"].append(tool_name)
            elif "financial" in tool_name_lower:
                categories["financial"].append(tool_name)
            else:
                categories["other"].append(tool_name)
        
        return categories
    
    def get_available_servers(self):
        """Get list of available servers"""
        servers = set()
        for tool in self.tools_cache.values():
            server = tool.get("server", "unknown")
            servers.add(server)
        return list(servers)

# Usage example
import time

manager = AOPClusterManager([
    "http://localhost:8000/mcp",
    "http://localhost:8001/mcp",
    "http://localhost:8002/mcp"
])

# Refresh and display tools
tool_count = manager.refresh_tools()
print(f"Loaded {tool_count} tools")

# Categorize tools
categories = manager.list_tools_by_category()
for category, tools in categories.items():
    if tools:
        print(f"{category.title()}: {tools}")

# Get available servers
servers = manager.get_available_servers()
print(f"Available servers: {servers}")
```

## Error Handling and Resilience

```python
class ResilientAOPCluster:
    def __init__(self, urls, transport="streamable-http"):
        self.urls = urls
        self.transport = transport
        self.cluster = AOPCluster(urls, transport)
        self.failed_servers = set()
    
    def get_tools_with_fallback(self, output_type="dict"):
        """Get tools with fallback for failed servers"""
        try:
            return self.cluster.get_tools(output_type=output_type)
        except Exception as e:
            print(f"Error getting tools: {e}")
            # Try individual servers
            all_tools = []
            for url in self.urls:
                if url in self.failed_servers:
                    continue
                try:
                    single_cluster = AOPCluster([url], self.transport)
                    tools = single_cluster.get_tools(output_type=output_type)
                    all_tools.extend(tools)
                except Exception as server_error:
                    print(f"Server {url} failed: {server_error}")
                    self.failed_servers.add(url)
            return all_tools
    
    def find_tool_with_retry(self, tool_name, max_retries=3):
        """Find tool with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.cluster.find_tool_by_server_name(tool_name)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        return None

# Usage
resilient_cluster = ResilientAOPCluster([
    "http://localhost:8000/mcp",
    "http://localhost:8001/mcp",
    "http://localhost:8002/mcp"
])

# Get tools with error handling
tools = resilient_cluster.get_tools_with_fallback()
print(f"Retrieved {len(tools)} tools")

# Find tool with retry
research_tool = resilient_cluster.find_tool_with_retry("Research-Agent")
if research_tool:
    print("Found Research-Agent tool")
else:
    print("Research-Agent tool not found after retries")
```

## Monitoring and Health Checks

```python
class AOPClusterMonitor:
    def __init__(self, cluster_manager):
        self.manager = cluster_manager
        self.health_status = {}
    
    def check_server_health(self, url):
        """Check if a server is healthy"""
        try:
            single_cluster = AOPCluster([url], self.manager.cluster.transport)
            tools = single_cluster.get_tools(output_type="dict")
            return {
                "status": "healthy",
                "tool_count": len(tools),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def check_all_servers(self):
        """Check health of all servers"""
        for url in self.manager.cluster.urls:
            health = self.check_server_health(url)
            self.health_status[url] = health
            status_icon = "✓" if health["status"] == "healthy" else "✗"
            print(f"{status_icon} {url}: {health['status']}")
            if health["status"] == "healthy":
                print(f"  Tools available: {health['tool_count']}")
            else:
                print(f"  Error: {health['error']}")
    
    def get_health_summary(self):
        """Get summary of server health"""
        healthy_count = sum(1 for status in self.health_status.values() 
                          if status["status"] == "healthy")
        total_count = len(self.health_status)
        return {
            "healthy_servers": healthy_count,
            "total_servers": total_count,
            "health_percentage": (healthy_count / total_count) * 100 if total_count > 0 else 0
        }

# Usage
monitor = AOPClusterMonitor(manager)
monitor.check_all_servers()

summary = monitor.get_health_summary()
print(f"Health Summary: {summary['healthy_servers']}/{summary['total_servers']} servers healthy ({summary['health_percentage']:.1f}%)")
```

## Complete Example

```python
import json
import time
from swarms.structs.aop import AOPCluster

def main():
    # Initialize cluster
    cluster = AOPCluster(
        urls=[
            "http://localhost:8000/mcp",
            "http://localhost:8001/mcp",
            "http://localhost:8002/mcp"
        ],
        transport="streamable-http"
    )
    
    print("AOP Cluster Management System")
    print("=" * 40)
    
    # Get all tools
    print("\n1. Discovering tools...")
    tools = cluster.get_tools(output_type="dict")
    print(f"Found {len(tools)} tools across all servers")
    
    # List all tool names
    tool_names = [tool.get("function", {}).get("name") for tool in tools]
    print(f"Available tools: {tool_names}")
    
    # Find specific tools
    print("\n2. Finding specific tools...")
    target_tools = ["Research-Agent", "Analysis-Agent", "Writing-Agent", "Code-Agent", "Financial-Agent"]
    
    for tool_name in target_tools:
        tool = cluster.find_tool_by_server_name(tool_name)
        if tool:
            print(f"✓ {tool_name}: Available")
        else:
            print(f"✗ {tool_name}: Not found")
    
    # Display tool details
    print("\n3. Tool details:")
    for tool in tools[:3]:  # Show first 3 tools
        print(f"\nTool: {tool.get('function', {}).get('name')}")
        print(f"Description: {tool.get('function', {}).get('description')}")
        print(f"Parameters: {list(tool.get('function', {}).get('parameters', {}).get('properties', {}).keys())}")
    
    print("\nAOP Cluster setup complete!")

if __name__ == "__main__":
    main()
```

This example demonstrates comprehensive AOP cluster management including tool discovery, error handling, health monitoring, and advanced cluster operations.
