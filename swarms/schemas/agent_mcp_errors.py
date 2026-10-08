class AgentMCPError(Exception):
    pass


class AgentMCPConnectionError(AgentMCPError):
    pass


class AgentMCPToolError(AgentMCPError):
    pass
