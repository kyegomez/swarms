import fastapi
import uvicorn
from .registry import build_tool, list_tools
from .retriever import Retriever
from typing import List, Dict
from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    topk: int = 3


def _bind_tool_server(t: "ToolServer"):
    """Add property API to ToolServer.
    t.api is a FastAPI object
    """

    @t.api.get("/")
    def health():
        return {
            "status": "ok",
        }

    @t.api.get("/list")
    def get_tools_list():
        return {
            "tools": t.list_tools(),
        }

    @t.api.get("/loaded")
    def get_loaded_tools():
        return {
            "tools": list(t.loaded_tools),
        }

    @t.api.get("/.well-known/ai-plugin.json", include_in_schema=False)
    def get_api_info():
        return {
            "schema_version": "v1",
            "name_for_human": "SwarmTools",
            "name_for_model": "SwarmTools",
            "description_for_human": "tools for swarms",
            "description_for_model": "tools for swarms",
            "auth": {
                "type": "none",
            },
            "api": {
                "type": "openapi",
                "url": "/openapi.json",
                "is_user_authenticated": False,
            },
            "logo_url": None,
            "contact_email": "",
            "legal_info_url": "",
        }

    @t.api.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        tool_list = t.retrieve(request.query, request.topk)
        return {"tools": tool_list}


class ToolServer:
    """This class host your own API backend."""

    def __init__(self) -> None:
        # define the root API server
        self.api = fastapi.FastAPI(
            title="Swarm tools",
            description="Tools for swarms",
        )
        self.loaded_tools = dict()
        self.retriever = Retriever()
        _bind_tool_server(self)

    def load_tool(self, name: str, config: Dict = {}):
        if self.is_loaded(name):
            raise ValueError(f"Tool {name} is already loaded")
        try:
            tool = build_tool(name, config)
        except BaseException as e:
            print(f"Cannot load tool {name}: {repr(e)}")
            return
        self.loaded_tools[name] = tool.api_info
        self.retriever.add_tool(name, tool.api_info)

        # mount sub API server to the root API server, thus can mount all urls of sub API server to /tools/{name} route
        self.api.mount(f"/tools/{name}", tool, name)
        return

    def is_loaded(self, name: str):
        return name in self.loaded_tools

    def serve(self, host: str = "0.0.0.0", port: int = 8079):
        uvicorn.run(self.api, host=host, port=port)

    def list_tools(self) -> List[str]:
        return list_tools()

    def retrieve(self, query: str, topk: int = 3) -> List[str]:
        return self.retriever.query(query, topk)
