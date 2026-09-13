"""
Serve one agent, or any swarm, as an MCP server.

``MCPDeployer`` wraps one or more targets, each an ``Agent`` or any callable
structure (a ``SequentialWorkflow``, a ``SwarmRouter``, a plain function), and
exposes each as its own MCP tool over streamable HTTP, SSE or stdio. Every HTTP request passes
through an auth layer before it reaches the MCP transport, so the deployed
server can require an API key, a bearer token checked by your own code, or an
OAuth-style ``TokenVerifier`` from the ``mcp`` package.

Example:
    >>> from swarms import Agent, MCPDeployer
    >>> agent = Agent(agent_name="Researcher", model_name="gpt-5.4", max_loops=1)
    >>> MCPDeployer(agent, api_keys=["sk-local-dev"], port=8000).run()

    Several agents and swarms become several tools on one server:

    >>> MCPDeployer(
    ...     {"research": researcher, "write": writer, "review": pipeline},
    ...     api_keys=["sk-local-dev"],
    ... ).run()

    Another agent can then use it with
    ``Agent(mcp_url="http://127.0.0.1:8000/mcp", ...)`` and the key in
    ``MCPConnection(api_key="sk-local-dev")``.
"""

import hmac
import inspect
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

import anyio
from loguru import logger
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse

AuthCallable = Callable[
    [Optional[str], Headers],
    Union[bool, Dict[str, Any], None, Awaitable[Any]],
]

# Paths any client may hit without credentials.
DEFAULT_PUBLIC_PATHS = ("/health",)

# Banner palette: the CLI's alien in three shades of purple instead of red.
BANNER_DEEP = "#7B2CFF"
BANNER_PURPLE = "#9B5CFF"
BANNER_VIOLET = "#C77DFF"


@dataclass
class AuthResult:
    """What the auth layer learned about a request it let through."""

    method: str
    subject: Optional[str] = None
    scopes: List[str] = field(default_factory=list)
    claims: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServedTool:
    """One target exposed as one MCP tool."""

    name: str
    target: Any
    description: str

    @property
    def target_type(self) -> str:
        return type(self.target).__name__


def _is_servable(target: Any) -> bool:
    return callable(target) or callable(getattr(target, "run", None))


def _default_tool_name(target: Any) -> str:
    """A snake_case MCP tool name derived from the target."""
    raw = (
        getattr(target, "agent_name", None)
        or getattr(target, "name", None)
        or getattr(target, "__name__", None)
        or type(target).__name__
    )
    name = re.sub(r"[^0-9a-zA-Z]+", "_", str(raw)).strip("_").lower()
    return name or "run"


def _default_description(target: Any) -> str:
    for attr in ("agent_description", "description"):
        value = getattr(target, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    doc = inspect.getdoc(target)
    if doc:
        return doc.strip().splitlines()[0]
    return f"Run {_default_tool_name(target)} on a task."


def _to_text(result: Any) -> str:
    """Render whatever the target returned as the tool's text result."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, indent=2, default=str)
    except Exception:
        return str(result)


def credential_from_headers(
    headers: Headers, api_key_header: str = "x-api-key"
) -> Optional[str]:
    """
    Pull the credential a client sent, from either supported header.

    ``api_key_header`` is read first, with an optional ``Bearer`` prefix
    stripped. ``Authorization: Bearer <token>`` is the fallback, which is
    what ``MCPManager`` sends by default.

    Args:
        headers: The request headers.
        api_key_header: The dedicated key header to look at first.

    Returns:
        The credential, or None when neither header carries one.
    """
    raw = headers.get(api_key_header)
    if raw and raw.strip():
        value = raw.strip()
        if value.lower().startswith("bearer "):
            value = value[7:].strip()
        return value or None

    authorization = headers.get("authorization", "")
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


class _AuthMiddleware:
    """ASGI middleware that asks the deployer to admit each HTTP request."""

    def __init__(self, app: Any, deployer: "MCPDeployer"):
        self.app = app
        self.deployer = deployer

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in self.deployer.public_paths:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        try:
            result = await self.deployer.authenticate(headers)
        except Exception as e:
            logger.warning(
                f"MCPDeployer auth check raised on {path}: {e}"
            )
            result = None

        if result is None:
            logger.warning(f"MCPDeployer rejected request to {path}")
            response = JSONResponse(
                {"error": "unauthorized"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        scope.setdefault("state", {})["mcp_auth"] = result
        await self.app(scope, receive, send)


class MCPDeployer:
    """
    Expose an agent or swarm as an authenticated MCP server.

    Args:
        targets: What to serve. One target, a list of targets, or a dict
            of tool name to target. A target is an ``Agent``, any structure
            with a ``run(task, ...)`` method, or a plain callable taking the
            task string. Each target becomes one MCP tool with a
            ``(task, img)`` schema. More can be added with :meth:`add_tool`.
        name: Server name advertised to MCP clients. Defaults to the first
            tool's name.
        description: Tool description for a single target. Defaults to the
            target's ``agent_description`` / ``description`` / docstring.
            With several targets use a dict or :meth:`add_tool` instead.
        tool_name: Tool name for a single target. Defaults to a snake_case
            form of the target's name. With several targets use a dict.
        host: Bind address. Defaults to ``127.0.0.1``.
        port: Bind port. Defaults to ``8000``.
        transport: ``"streamable-http"`` (default), ``"sse"`` or
            ``"stdio"``. Auth applies to the two HTTP transports only.
        path: URL path for the MCP endpoint. Defaults to ``/mcp`` (or
            ``/sse`` for SSE).
        api_keys: Static keys accepted in ``api_key_header`` or as a
            bearer token. Compared in constant time.
        api_key_env: Name of an environment variable holding more keys,
            comma-separated. Read at construction.
        api_key_header: Header clients may use for a raw key. Defaults to
            ``x-api-key``. ``Authorization: Bearer`` always works too.
        auth: Your own check, ``(credential, headers) -> bool | dict |
            None``, sync or async. A truthy return admits the request; a
            dict is stored as the request's claims. Takes precedence over
            ``api_keys`` and ``token_verifier``.
        token_verifier: An ``mcp.server.auth.provider.TokenVerifier``.
            Used when ``auth`` is not given; the token must carry every
            scope in ``required_scopes``.
        required_scopes: Scopes a verified token must include.
        allow_anonymous: Serve without any auth. Off by default, so a
            server with no credentials configured refuses to start.
        public_paths: Paths exempt from auth. Defaults to ``/health``.
        extra_tools: Additional plain functions to expose as MCP tools.
            Each needs a docstring; the signature becomes the schema.
        timeout: Seconds a single tool call may run before it fails.
        json_response: Streamable HTTP only. Return plain JSON instead of
            an event stream.
        stateless_http: Streamable HTTP only. Do not keep per-session
            state; the right choice behind a load balancer.
        verbose: Log every admitted call.
        show_banner: Print the startup banner from ``run`` and ``start``.
    """

    def __init__(
        self,
        targets: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
        tool_name: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        transport: str = "streamable-http",
        path: Optional[str] = None,
        api_keys: Optional[Iterable[str]] = None,
        api_key_env: Optional[str] = None,
        api_key_header: str = "x-api-key",
        auth: Optional[AuthCallable] = None,
        token_verifier: Optional[Any] = None,
        required_scopes: Optional[Iterable[str]] = None,
        allow_anonymous: bool = False,
        public_paths: Optional[Iterable[str]] = None,
        extra_tools: Optional[Iterable[Callable]] = None,
        timeout: Optional[float] = None,
        json_response: bool = False,
        stateless_http: bool = True,
        verbose: bool = False,
        show_banner: bool = True,
    ):
        if targets is None:
            raise ValueError("MCPDeployer needs a target to serve.")
        if transport not in ("streamable-http", "sse", "stdio"):
            raise ValueError(
                "transport must be 'streamable-http', 'sse' or 'stdio', "
                f"got {transport!r}."
            )

        self.tools: Dict[str, ServedTool] = {}
        self._register_targets(targets, tool_name, description)
        self.name = name or self.tool_name
        self.host = host
        self.port = int(port)
        self.transport = transport
        self.path = path or ("/sse" if transport == "sse" else "/mcp")
        self.api_key_header = api_key_header.lower()
        self.auth = auth
        self.token_verifier = token_verifier
        self.required_scopes = list(required_scopes or [])
        self.allow_anonymous = allow_anonymous
        self.public_paths = tuple(
            public_paths or DEFAULT_PUBLIC_PATHS
        )
        self.extra_tools = list(extra_tools or [])
        self.timeout = timeout
        self.json_response = json_response
        self.stateless_http = stateless_http
        self.verbose = verbose
        self.show_banner = show_banner

        self.api_keys = self._collect_api_keys(api_keys, api_key_env)

        if (
            not self.allow_anonymous
            and self.auth is None
            and self.token_verifier is None
            and not self.api_keys
        ):
            raise ValueError(
                "No auth configured. Pass api_keys, api_key_env, auth or "
                "token_verifier, or set allow_anonymous=True."
            )

        self.server = self._build_server()
        for tool in self.tools.values():
            self._register_on_server(tool)
        self._app = None
        self._uvicorn = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _register_targets(
        self,
        targets: Any,
        tool_name: Optional[str],
        description: Optional[str],
    ) -> None:
        if isinstance(targets, dict):
            if tool_name or description:
                raise ValueError(
                    "tool_name and description apply to a single target; "
                    "with a dict the keys are the tool names."
                )
            for key, target in targets.items():
                self._add(target, key, None)
            return

        if isinstance(targets, (list, tuple)):
            if tool_name or description:
                raise ValueError(
                    "tool_name and description apply to a single target; "
                    "pass a dict to name several."
                )
            for target in targets:
                self._add(target, None, None)
            return

        self._add(targets, tool_name, description)

        if not self.tools:
            raise ValueError("MCPDeployer needs at least one target.")

    def _add(
        self,
        target: Any,
        name: Optional[str],
        description: Optional[str],
    ) -> ServedTool:
        if not _is_servable(target):
            raise ValueError(
                f"{target!r} must be callable or expose a run() method."
            )
        tool = ServedTool(
            name=name or _default_tool_name(target),
            target=target,
            description=description or _default_description(target),
        )
        if tool.name in self.tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered; "
                "pass a dict to give each target its own name."
            )
        self.tools[tool.name] = tool
        return tool

    def add_tool(
        self,
        target: Any,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ServedTool:
        """
        Register one more agent, swarm or callable as a tool.

        Call before :meth:`run` or :meth:`start`.

        Args:
            target: The agent, swarm or callable to serve.
            name: Tool name. Defaults to a snake_case form of the target's
                name.
            description: Tool description. Defaults to the target's own.

        Returns:
            The registered ``ServedTool``.
        """
        if self._thread is not None:
            raise RuntimeError(
                "add_tool must be called before start()."
            )
        tool = self._add(target, name, description)
        self._register_on_server(tool)
        return tool

    @property
    def tool_names(self) -> List[str]:
        return list(self.tools)

    @property
    def tool_name(self) -> str:
        """The first registered tool's name."""
        return next(iter(self.tools))

    @property
    def target(self) -> Any:
        """The first registered target."""
        return self.tools[self.tool_name].target

    @property
    def description(self) -> str:
        """The first registered tool's description."""
        return self.tools[self.tool_name].description

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_api_keys(
        api_keys: Optional[Iterable[str]], api_key_env: Optional[str]
    ) -> List[str]:
        keys = [
            k.strip() for k in (api_keys or []) if k and k.strip()
        ]
        if api_key_env:
            raw = os.getenv(api_key_env, "")
            keys += [k.strip() for k in raw.split(",") if k.strip()]
        return list(dict.fromkeys(keys))

    def _key_matches(self, credential: str) -> bool:
        return any(
            hmac.compare_digest(credential, key)
            for key in self.api_keys
        )

    async def authenticate(
        self, headers: Headers
    ) -> Optional[AuthResult]:
        """
        Decide whether a request may proceed.

        Checks, in order: ``allow_anonymous``, the custom ``auth``
        callable, ``token_verifier``, then the static key set.

        Args:
            headers: The request headers.

        Returns:
            An ``AuthResult`` when admitted, None when refused.
        """
        if self.allow_anonymous:
            return AuthResult(method="anonymous")

        credential = credential_from_headers(
            headers, self.api_key_header
        )

        if self.auth is not None:
            verdict = self.auth(credential, headers)
            if inspect.isawaitable(verdict):
                verdict = await verdict
            if not verdict:
                return None
            claims = verdict if isinstance(verdict, dict) else {}
            return AuthResult(
                method="custom",
                subject=claims.get("subject") or claims.get("sub"),
                scopes=list(claims.get("scopes", [])),
                claims=claims,
            )

        if credential is None:
            return None

        if self.token_verifier is not None:
            token = await self.token_verifier.verify_token(credential)
            if token is None:
                return None
            if token.expires_at and token.expires_at < int(
                time.time()
            ):
                return None
            scopes = list(token.scopes or [])
            if any(s not in scopes for s in self.required_scopes):
                return None
            return AuthResult(
                method="token",
                subject=getattr(token, "subject", None)
                or token.client_id,
                scopes=scopes,
                claims=dict(getattr(token, "claims", None) or {}),
            )

        if self._key_matches(credential):
            return AuthResult(method="api_key")
        return None

    # ------------------------------------------------------------------
    # Tool
    # ------------------------------------------------------------------

    def _invoke(
        self, name: str, task: str, img: Optional[str] = None
    ) -> str:
        """Run one tool's target synchronously and render its output."""
        target = self.tools[name].target
        run = getattr(target, "run", None)
        if callable(run):
            if img:
                try:
                    return _to_text(run(task, img=img))
                except TypeError:
                    pass
            return _to_text(run(task))
        return _to_text(target(task))

    async def _call(
        self, name: str, task: str, img: Optional[str] = None
    ) -> str:
        if self.verbose:
            logger.info(
                f"MCPDeployer[{self.name}] {name}: {task[:80]!r}"
            )
        fn = partial(self._invoke, name, task, img)
        if self.timeout:
            with anyio.fail_after(self.timeout):
                return await anyio.to_thread.run_sync(fn)
        return await anyio.to_thread.run_sync(fn)

    def _build_server(self):
        from mcp.server.mcpserver import MCPServer

        instructions = "\n".join(
            f"{t.name}: {t.description}" for t in self.tools.values()
        )
        server = MCPServer(name=self.name, instructions=instructions)

        for fn in self.extra_tools:
            server.tool()(fn)

        deployer = self

        @server.custom_route("/health", methods=["GET"])
        async def health(request: Request) -> JSONResponse:
            return JSONResponse(
                {
                    "status": "ok",
                    "name": deployer.name,
                    "tool": deployer.tool_name,
                    "tools": deployer.tool_names,
                    "transport": deployer.transport,
                }
            )

        return server

    def _register_on_server(self, tool: ServedTool) -> None:
        name = tool.name

        # The closure carries no `self` in its signature so the schema is
        # just (task, img).
        async def run(task: str, img: Optional[str] = None) -> str:
            return await self._call(name, task, img)

        run.__name__ = name
        run.__doc__ = tool.description
        self.server.tool(name=name, description=tool.description)(run)

    # ------------------------------------------------------------------
    # App and lifecycle
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}{self.path}"

    def _transport_security(self):
        from mcp.server.transport_security import (
            TransportSecuritySettings,
        )

        if self.host in ("127.0.0.1", "localhost"):
            return None
        # The default rebinding guard only admits localhost Host headers,
        # which would reject every request to a server bound elsewhere.
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=False
        )

    def build_app(self):
        """The ASGI app: the MCP transport wrapped in the auth layer."""
        if self.transport == "stdio":
            raise ValueError("stdio transport has no ASGI app.")

        if self.transport == "sse":
            inner = self.server.sse_app(
                sse_path=self.path,
                transport_security=self._transport_security(),
                host=self.host,
            )
        else:
            inner = self.server.streamable_http_app(
                streamable_http_path=self.path,
                json_response=self.json_response,
                stateless_http=self.stateless_http,
                transport_security=self._transport_security(),
                host=self.host,
            )
        return _AuthMiddleware(inner, self)

    @property
    def app(self):
        if self._app is None:
            self._app = self.build_app()
        return self._app

    def _auth_label(self) -> str:
        if self.allow_anonymous:
            return "anonymous (open)"
        if self.auth is not None:
            return "custom auth callable"
        if self.token_verifier is not None:
            scopes = ", ".join(self.required_scopes) or "any"
            return f"token verifier · scopes: {scopes}"
        n = len(self.api_keys)
        return f"{n} api key{'s' if n != 1 else ''} · {self.api_key_header} or Bearer"

    def print_banner(self, console: Optional[Console] = None) -> None:
        """Print the startup banner: the swarms alien in purple."""
        console = console or Console()

        icon = Text()
        icon.append("▄     ▄\n", style=f"bold {BANNER_DEEP}")
        icon.append("▀█████▀\n", style=f"bold {BANNER_DEEP}")
        icon.append("█▀███▀█\n", style=f"bold {BANNER_PURPLE}")
        icon.append("███████\n", style=f"bold {BANNER_PURPLE}")
        icon.append("▀█   █▀", style=f"bold {BANNER_VIOLET}")

        endpoint = "stdio" if self.transport == "stdio" else self.url
        info = Text()
        info.append("MCPDeployer", style="bold white")
        info.append(f"  {self.name}\n", style=f"bold {BANNER_VIOLET}")
        tools = list(self.tools.values())
        shown, hidden = tools[:4], tools[4:]
        label = "tool      " if len(tools) == 1 else "tools     "
        for i, tool in enumerate(shown):
            info.append(
                label if i == 0 else "          ", style="dim white"
            )
            info.append(tool.name, style=f"{BANNER_DEEP}")
            info.append(
                f"  ({tool.target_type})\n", style="dim white"
            )
        if hidden:
            info.append("          ", style="dim white")
            info.append(f"+{len(hidden)} more\n", style="dim white")
        info.append("endpoint  ", style="dim white")
        info.append(f"{endpoint}\n", style=f"{BANNER_DEEP}")
        info.append("transport ", style="dim white")
        info.append(f"{self.transport}\n", style="white")
        info.append("auth      ", style="dim white")
        info.append(self._auth_label(), style="white")

        header = Table.grid(padding=(0, 1))
        header.add_column(width=9, vertical="top")
        header.add_column(vertical="top")
        header.add_row(icon, info)

        hint = Text()
        if self.transport == "stdio":
            hint.append(
                "Launched by an MCP host over stdin/stdout.",
                style="dim white",
            )
        else:
            hint.append("Connect with  ", style="dim white")
            hint.append(
                f'Agent(mcp_url=MCPConnection(url="{self.url}", api_key=...))',
                style=BANNER_PURPLE,
            )
            hint.append(
                f"\nHealth check  GET http://{self.host}:{self.port}/health",
                style="dim white",
            )

        console.print(
            Panel(
                Group(
                    header,
                    Text(""),
                    Rule(style=f"dim {BANNER_PURPLE}"),
                    hint,
                ),
                border_style=BANNER_DEEP,
                title=f"[bold {BANNER_VIOLET}] 👾 Swarms MCP [/bold {BANNER_VIOLET}]",
                title_align="left",
                subtitle="[dim white] ctrl+c to stop [/dim white]",
                subtitle_align="right",
                padding=(0, 2),
            )
        )

    def run(self, log_level: str = "info") -> None:
        """Serve until interrupted. Blocks."""
        if self.show_banner:
            self.print_banner()
        if self.transport == "stdio":
            if not self.allow_anonymous:
                logger.warning(
                    "stdio transport carries no headers; auth settings "
                    "are ignored."
                )
            self.server.run("stdio")
            return

        import uvicorn

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=log_level,
        )

    def start(self, wait: float = 10.0) -> "MCPDeployer":
        """
        Serve in a background thread and return once the socket is open.

        Args:
            wait: Seconds to wait for startup before raising.
        """
        if self.transport == "stdio":
            raise ValueError("start() needs an HTTP transport.")
        if self._thread is not None:
            return self
        if self.show_banner:
            self.print_banner()

        import uvicorn

        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
        )
        self._uvicorn = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._uvicorn.run, daemon=True
        )
        self._thread.start()

        deadline = time.monotonic() + wait
        while not self._uvicorn.started:
            if (
                time.monotonic() > deadline
                or not self._thread.is_alive()
            ):
                self.stop()
                raise RuntimeError(
                    f"MCPDeployer did not start on {self.host}:{self.port}"
                )
            time.sleep(0.05)
        return self

    def stop(self, wait: float = 10.0) -> None:
        """Stop a server started with :meth:`start`."""
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=wait)
        self._uvicorn = None
        self._thread = None

    def __enter__(self) -> "MCPDeployer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def deploy_as_mcp(target: Any, **kwargs) -> MCPDeployer:
    """Build an :class:`MCPDeployer` and serve it. Blocks."""
    deployer = MCPDeployer(target, **kwargs)
    deployer.run()
    return deployer
