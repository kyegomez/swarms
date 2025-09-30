import json

from swarms.structs.aop import AOPCluster

aop_cluster = AOPCluster(
    urls=["http://localhost:8000/mcp"],
    transport="streamable-http",
)

print(json.dumps(aop_cluster.get_tools(output_type="dict"), indent=4))
print(aop_cluster.find_tool_by_server_name("Research-Agent"))
