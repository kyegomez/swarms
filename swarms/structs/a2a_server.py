import uuid

from starlette.responses import JSONResponse

from swarms.structs.mcp_deployer import MCPDeployer


class A2AServer(MCPDeployer):
    def __init__(self, targets, **kwargs):
        super().__init__(targets, **kwargs)
        card_path = "/.well-known/agent-card.json"
        self.public_paths += (card_path,)
        route = self.server.custom_route
        route(card_path, methods=["GET"])(self.agent_card)
        route("/a2a", methods=["POST"])(self.send_message)

    async def agent_card(self, request) -> JSONResponse:
        skill = {
            "id": self.tool_name,
            "name": self.tool_name,
            "description": self.description,
            "tags": [type(self.target).__name__],
        }
        path = request.scope.get("root_path", "") + "/a2a"
        interface = {
            "url": str(request.base_url.replace(path=path)),
            "protocolBinding": "JSONRPC",
            "protocolVersion": "1.0",
        }
        return JSONResponse(
            {
                "name": self.name,
                "description": self.description,
                "supportedInterfaces": [interface],
                "version": "1.0.0",
                "capabilities": {},
                "defaultInputModes": ["text/plain"],
                "defaultOutputModes": ["text/plain"],
                "skills": [skill],
            }
        )

    async def send_message(self, request) -> JSONResponse:
        body = await request.json()
        reply = {"jsonrpc": "2.0", "id": body.get("id")}

        def error(code: int, message: str) -> JSONResponse:
            reply["error"] = {"code": code, "message": message}
            return JSONResponse(reply)

        if request.headers.get("a2a-version") != "1.0":
            return error(-32009, "A2A-Version must be 1.0")
        if body.get("method") != "SendMessage":
            return error(-32601, "Method not found")
        try:
            parts = body["params"]["message"]["parts"]
            task = "\n".join(p["text"] for p in parts if "text" in p)
        except (KeyError, TypeError):
            task = ""
        if not task:
            return error(-32602, "Message needs a text part")
        try:
            output = await self._call(self.tool_name, task)
        except Exception as e:
            return error(-32603, repr(e))
        message = {"role": "ROLE_AGENT", "parts": [{"text": output}]}
        message["messageId"] = str(uuid.uuid4())
        return JSONResponse(reply | {"result": {"message": message}})
