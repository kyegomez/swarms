import json
from swarms.schemas.agent_class_schema import AgentConfiguration
from swarms.tools.base_tool import BaseTool
from swarms.schemas.mcp_schemas import MCPConnection


base_tool = BaseTool()

schemas = [AgentConfiguration, MCPConnection]

schema = base_tool.multi_base_models_to_dict(schemas)

print(json.dumps(schema, indent=4))
