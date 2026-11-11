import uuid
from typing import Any, Dict, List, Optional, Sequence

from starlette.requests import Request
from starlette.responses import JSONResponse

from swarms.structs.a2a_client import (
    BINDING,
    CARD_PATH,
    SEND_MESSAGE,
    VERSION,
    text_of,
)
from swarms.structs.mcp_deployer import MCPDeployer

RPC_PATH = "/a2a"
AGENT_VERSION = "1.0.0"
TEXT_MODES = ("text/plain",)

VERSION_NOT_SUPPORTED = -32009
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class A2AServer(MCPDeployer):
    """An MCP server that also answers as an A2A agent.

    The agent card is public. The message endpoint uses the deployer's auth
    and runs the first registered tool.
    """

    def __init__(
        self,
        targets: Any,
        card_path: str = CARD_PATH,
        rpc_path: str = RPC_PATH,
        protocol_version: str = VERSION,
        agent_version: str = AGENT_VERSION,
        capabilities: Optional[Dict[str, Any]] = None,
        input_modes: Sequence[str] = TEXT_MODES,
        output_modes: Sequence[str] = TEXT_MODES,
        skills: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        """Add the agent card and message routes to the deployer.

        Args:
            targets (Any): One target, a list of targets, or a dict of tool name to target.
            card_path (str): Path the public agent card is served at.
            rpc_path (str): Path the JSON-RPC message endpoint is served at.
            protocol_version (str): A2A protocol version advertised and required from callers.
            agent_version (str): Version of the agent shown on the card.
            capabilities (Optional[Dict[str, Any]]): Capabilities shown on the card; empty when None.
            input_modes (Sequence[str]): Media types the agent accepts.
            output_modes (Sequence[str]): Media types the agent returns.
            skills (Optional[List[Dict[str, Any]]]): Skills shown on the card; the first tool when None.
            **kwargs (Any): Deployer options such as name, description, host, port and api_keys.
        """
        super().__init__(targets, **kwargs)
        self.card_path = card_path
        self.rpc_path = rpc_path
        self.protocol_version = protocol_version
        self.agent_version = agent_version
        self.capabilities = dict(capabilities or {})
        self.input_modes = list(input_modes)
        self.output_modes = list(output_modes)
        self.skills = skills

        self.public_paths += (card_path,)
        route = self.server.custom_route
        route(card_path, methods=["GET"])(self.agent_card)
        route(rpc_path, methods=["POST"])(self.send_message)

    async def agent_card(self, request: Request) -> JSONResponse:
        """Return this server's A2A agent card.

        Args:
            request (Request): The incoming request, whose base URL goes on the card.

        Returns:
            JSONResponse: The card, with one JSON-RPC interface and its skills.
        """
        card = {
            "name": self.name,
            "description": self.description,
            "version": self.agent_version,
            "capabilities": self.capabilities,
            "defaultInputModes": self.input_modes,
            "defaultOutputModes": self.output_modes,
            "supportedInterfaces": [self._card_interface(request)],
            "skills": self.skills or [self._card_skill()],
        }
        return JSONResponse(card)

    async def send_message(self, request: Request) -> JSONResponse:
        """Run the first tool on a SendMessage request.

        Args:
            request (Request): A JSON-RPC request with the A2A-Version header.

        Returns:
            JSONResponse: The tool output as an agent message, or a JSON-RPC error.
        """
        body = await request.json()
        rpc_id = body.get("id")

        version = self.protocol_version
        if request.headers.get("a2a-version") != version:
            return self._rpc_error(
                rpc_id,
                VERSION_NOT_SUPPORTED,
                f"A2A-Version must be {version}",
            )
        if body.get("method") != SEND_MESSAGE:
            return self._rpc_error(
                rpc_id, METHOD_NOT_FOUND, "Method not found"
            )

        task = self._message_task(body)
        if not task:
            return self._rpc_error(
                rpc_id, INVALID_PARAMS, "Message needs a text part"
            )

        try:
            output = await self._call(self.tool_name, task)
        except Exception as e:
            return self._rpc_error(rpc_id, INTERNAL_ERROR, repr(e))
        return self._rpc_result(rpc_id, output)

    def _card_interface(self, request: Request) -> dict:
        """Describe the JSON-RPC endpoint for the agent card.

        Args:
            request (Request): The incoming request, used for the public base URL.

        Returns:
            dict: The interface entry with its URL, binding and version.
        """
        path = request.scope.get("root_path", "") + self.rpc_path
        return {
            "url": str(request.base_url.replace(path=path)),
            "protocolBinding": BINDING,
            "protocolVersion": self.protocol_version,
        }

    def _card_skill(self) -> dict:
        """Describe the first tool as the card's skill.

        Returns:
            dict: The skill entry, tagged with the target's type.
        """
        return {
            "id": self.tool_name,
            "name": self.tool_name,
            "description": self.description,
            "tags": [type(self.target).__name__],
        }

    @staticmethod
    def _message_task(body: dict) -> str:
        """Read the task text from a SendMessage request body.

        Args:
            body (dict): The JSON-RPC request body.

        Returns:
            str: The message's text parts, or an empty string when there are none.
        """
        try:
            return text_of(body["params"]["message"])
        except (KeyError, TypeError, AttributeError):
            return ""

    @staticmethod
    def _rpc_error(
        rpc_id: Any, code: int, message: str
    ) -> JSONResponse:
        """Build a JSON-RPC error response.

        Args:
            rpc_id (Any): The id of the request being answered.
            code (int): The JSON-RPC error code.
            message (str): What went wrong.

        Returns:
            JSONResponse: The error response.
        """
        error = {"code": code, "message": message}
        return JSONResponse(
            {"jsonrpc": "2.0", "id": rpc_id, "error": error}
        )

    @staticmethod
    def _rpc_result(rpc_id: Any, output: str) -> JSONResponse:
        """Build a JSON-RPC response carrying the agent's reply.

        Args:
            rpc_id (Any): The id of the request being answered.
            output (str): The tool's output.

        Returns:
            JSONResponse: The reply as an agent message.
        """
        message = {
            "role": "ROLE_AGENT",
            "messageId": str(uuid.uuid4()),
            "parts": [{"text": output}],
        }
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {"message": message},
            }
        )
