class AgentMCPError(Exception):
    pass


class AgentMCPConnectionError(AgentMCPError):
    pass


class AgentMCPToolError(AgentMCPError):
    pass


class AgentMCPToolNotFoundError(AgentMCPError):
    pass


class AgentMCPToolInvalidError(AgentMCPError):
    pass
