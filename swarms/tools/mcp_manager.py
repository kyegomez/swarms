"""
MCP (Model Context Protocol) client manager for Swarms agents.

This module owns every piece of MCP behaviour that used to be inlined in
``swarms/structs/agent.py``:

* connection/config normalization (a single URL, many URLs, ``MCPConnection``
  objects, or plain dicts)
* authentication — API keys, bearer tokens, arbitrary headers, and full
  OAuth 2.1 (PKCE authorization-code with RFC 7591 dynamic client
  registration, plus headless client-credentials and pre-issued tokens)
* transport selection — streamable HTTP, SSE, and stdio, with auto detection
* tool discovery, returned as OpenAI function-calling schemas
* tool execution, routing each tool call to the server that owns the tool

``MCPManager`` is the only object an ``Agent`` needs to hold.
"""

import asyncio
import json
import os
import re
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

from loguru import logger
from mcp import ClientSession
from mcp.types import Tool as MCPTool

from swarms.schemas.agent_mcp_errors import (
    AgentMCPConnectionError,
    AgentMCPToolError,
)
from swarms.schemas.mcp_schemas import (
    MCPConnection,
    MCPOAuthConfig,
)

DEFAULT_TOKEN_CACHE_DIR = Path.home() / ".swarms" / "mcp_auth"

# Servers advertising these paths are still commonly SSE-only.
_SSE_URL_HINTS = ("/sse", "/sse/")

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


########################################################
# Helpers
########################################################


def _resolve_secret(value: Optional[str]) -> Optional[str]:
    """
    Resolve a possibly indirected secret.

    ``"env:MY_TOKEN"`` and ``"${MY_TOKEN}"`` are read from the environment;
    anything else is returned untouched.
    """
    if not isinstance(value, str) or not value:
        return value

    if value.startswith("env:"):
        name = value[4:].strip()
        resolved = os.getenv(name)
        if resolved is None:
            logger.warning(
                f"MCP: environment variable '{name}' is not set."
            )
        return resolved

    match = _ENV_PATTERN.match(value)
    if match:
        name = match.group(1)
        resolved = os.getenv(name)
        if resolved is None:
            logger.warning(
                f"MCP: environment variable '{name}' is not set."
            )
        return resolved

    return value


def _describe_exception(error: Optional[BaseException]) -> str:
    """
    Render an exception as an actionable one-liner.

    anyio wraps transport failures in ``(Base)ExceptionGroup``s whose ``str()``
    is empty, which is how a 401 used to reach the user as a blank message.
    This flattens groups and always includes the exception type.
    """
    if error is None:
        return "unknown error"

    parts: List[str] = []

    def _walk(exc: BaseException, depth: int = 0) -> None:
        if depth > 5:
            return
        nested = getattr(exc, "exceptions", None)
        if nested:
            for sub in nested:
                _walk(sub, depth + 1)
            return
        message = str(exc).strip()
        parts.append(
            f"{type(exc).__name__}: {message}"
            if message
            else type(exc).__name__
        )

    _walk(error)

    seen = set()
    unique = [p for p in parts if not (p in seen or seen.add(p))]
    return " | ".join(unique) or type(error).__name__


def _slugify_url(url: str) -> str:
    """Turn a server URL into a filesystem-safe slug."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", url).strip("_")[:150]


def _server_origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _mcp_is_v2() -> bool:
    """
    True when the installed mcp is 2.x.

    Detected by the factory rename rather than a version string, so it tracks
    the actual API rather than packaging. 2.x also changed timeout types from
    ``timedelta`` to plain seconds.
    """
    try:
        from mcp.client.streamable_http import (  # noqa: F401
            streamablehttp_client,
        )

        return False
    except ImportError:
        return True


MCP_IS_V2 = _mcp_is_v2()


def _read_timeout(seconds: float):
    """Session read timeout in the type the installed mcp expects."""
    return seconds if MCP_IS_V2 else timedelta(seconds=seconds)


def _http_timeout(connect: float, read: float):
    """
    Build a timeout object for mcp 2.x's ``create_mcp_http_client``.

    It expects the httpx flavour that the installed mcp bundles, which is
    ``httpx2`` on 2.x and ``httpx`` on 1.x. Falling back to ``None`` simply
    accepts the library defaults rather than failing the connection.
    """
    for module_name in ("httpx2", "httpx"):
        try:
            module = __import__(module_name)
            return module.Timeout(connect, read=read)
        except Exception:
            continue
    return None


def run_async(coro: Any) -> Any:
    """
    Run a coroutine from sync code, whether or not a loop is already running.

    When called from inside a running event loop the coroutine is executed on
    a dedicated loop in a worker thread, which avoids the
    "asyncio.run() cannot be called from a running event loop" failure the
    previous implementation hit inside async agents.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


########################################################
# OAuth support
########################################################


class MCPFileTokenStorage:
    """
    Disk-backed :class:`mcp.client.auth.TokenStorage` implementation.

    Tokens and dynamically registered client credentials are cached per server
    so an interactive OAuth flow only has to run once. The cache file is
    written with ``0600`` permissions.
    """

    def __init__(self, server_url: str, path: Optional[str] = None):
        self.server_url = server_url
        if path:
            self.path = Path(path).expanduser()
        else:
            self.path = (
                DEFAULT_TOKEN_CACHE_DIR
                / f"{_slugify_url(server_url)}.json"
            )
        self._lock = threading.Lock()

    # --- internal io -------------------------------------------------
    def _read(self) -> Dict[str, Any]:
        with self._lock:
            try:
                if not self.path.exists():
                    return {}
                return json.loads(self.path.read_text())
            except Exception as e:
                logger.warning(
                    f"MCP: could not read OAuth token cache {self.path}: {e}"
                )
                return {}

    def _write(self, data: Dict[str, Any]) -> None:
        with self._lock:
            try:
                self.path.parent.mkdir(
                    parents=True, exist_ok=True, mode=0o700
                )
                # Create the file 0600 from the start. write_text() creates it
                # honoring the umask (typically 0644), leaving a window in which
                # another local user could read the OAuth token before a later
                # chmod lands. os.open with an explicit mode closes that window.
                fd = os.open(
                    self.path,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                try:
                    os.write(fd, json.dumps(data, indent=2).encode())
                finally:
                    os.close(fd)
                # os.open's mode only applies on creation; tighten a file that
                # already existed with looser permissions.
                os.chmod(self.path, 0o600)
            except Exception as e:
                logger.warning(
                    f"MCP: could not write OAuth token cache {self.path}: {e}"
                )

    # --- TokenStorage protocol ---------------------------------------
    async def get_tokens(self):
        from mcp.shared.auth import OAuthToken

        data = self._read().get("tokens")
        if not data:
            return None
        try:
            return OAuthToken.model_validate(data)
        except Exception:
            return None

    async def set_tokens(self, tokens) -> None:
        data = self._read()
        data["tokens"] = json.loads(tokens.model_dump_json())
        data["server_url"] = self.server_url
        self._write(data)

    async def get_client_info(self):
        from mcp.shared.auth import OAuthClientInformationFull

        data = self._read().get("client_info")
        if not data:
            return None
        try:
            return OAuthClientInformationFull.model_validate(data)
        except Exception:
            return None

    async def set_client_info(self, client_info) -> None:
        data = self._read()
        data["client_info"] = json.loads(
            client_info.model_dump_json()
        )
        data["server_url"] = self.server_url
        self._write(data)

    def clear(self) -> None:
        """Delete the cached tokens, forcing a fresh authorization."""
        with self._lock:
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception as e:
                logger.warning(
                    f"MCP: could not clear token cache {self.path}: {e}"
                )


class MCPInMemoryTokenStorage:
    """Non-persistent token storage — used when ``use_token_cache=False``."""

    def __init__(self):
        self._tokens = None
        self._client_info = None

    async def get_tokens(self):
        return self._tokens

    async def set_tokens(self, tokens) -> None:
        self._tokens = tokens

    async def get_client_info(self):
        return self._client_info

    async def set_client_info(self, client_info) -> None:
        self._client_info = client_info

    def clear(self) -> None:
        self._tokens = None
        self._client_info = None


class _CallbackRequestHandler(BaseHTTPRequestHandler):
    """Captures the ``?code=...&state=...`` redirect from the OAuth server."""

    def do_GET(self):  # noqa: N802 - required name
        params = parse_qs(urlparse(self.path).query)
        self.server.auth_code = params.get("code", [None])[0]
        self.server.auth_state = params.get("state", [None])[0]
        self.server.auth_error = params.get("error", [None])[0]
        self.server.auth_error_description = params.get(
            "error_description", [None]
        )[0]

        if self.server.auth_error:
            body = (
                "<h2>Authorization failed</h2>"
                f"<p>{self.server.auth_error}</p>"
            )
            status = 400
        else:
            body = (
                "<h2>Authorization complete</h2>"
                "<p>You can close this tab and return to your agent.</p>"
            )
            status = 200

        payload = f"<html><body style='font-family:sans-serif'>{body}</body></html>".encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.server.callback_received.set()

    def log_message(self, *args, **kwargs):  # silence stdlib logging
        return


class _OAuthCallbackServer:
    """Short-lived loopback HTTP server for the authorization-code redirect."""

    def __init__(self, redirect_uri: str):
        parsed = urlparse(redirect_uri)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (
            443 if parsed.scheme == "https" else 80
        )
        self.path = parsed.path or "/callback"
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        try:
            server = HTTPServer(
                (self.host, self.port), _CallbackRequestHandler
            )
        except OSError as e:
            raise AgentMCPConnectionError(
                f"Could not bind the OAuth redirect listener on "
                f"{self.host}:{self.port}: {e}. Choose a free port via "
                f"MCPOAuthConfig(redirect_uri=...) or supply a pre-issued "
                f"access_token instead."
            )
        server.auth_code = None
        server.auth_state = None
        server.auth_error = None
        server.auth_error_description = None
        server.callback_received = threading.Event()
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="swarms-mcp-oauth-callback",
            daemon=True,
        )
        self._thread.start()

    def wait(self, timeout: float) -> Tuple[str, Optional[str]]:
        if self._server is None:
            raise AgentMCPConnectionError(
                "OAuth callback server was never started."
            )
        received = self._server.callback_received.wait(timeout)
        try:
            if not received:
                raise AgentMCPConnectionError(
                    f"Timed out after {timeout}s waiting for the OAuth "
                    f"redirect on {self.host}:{self.port}."
                )
            if self._server.auth_error:
                raise AgentMCPConnectionError(
                    f"OAuth authorization failed: {self._server.auth_error} "
                    f"{self._server.auth_error_description or ''}".strip()
                )
            if not self._server.auth_code:
                raise AgentMCPConnectionError(
                    "OAuth redirect did not include an authorization code."
                )
            return self._server.auth_code, self._server.auth_state
        finally:
            self.stop()

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        self._server = None
        self._thread = None


async def _discover_token_endpoint(
    server_url: str, timeout: float
) -> Optional[str]:
    """
    Look up the token endpoint from the server's authorization metadata.

    Tries the MCP/OAuth 2.0 protected-resource and authorization-server
    metadata documents in the order described by the MCP authorization spec.
    """
    import httpx

    origin = _server_origin(server_url)
    candidates = [
        f"{origin}/.well-known/oauth-authorization-server",
        f"{origin}/.well-known/openid-configuration",
    ]

    async with httpx.AsyncClient(
        timeout=timeout, follow_redirects=True
    ) as client:
        # The protected resource document points at its authorization servers.
        try:
            pr = await client.get(
                f"{origin}/.well-known/oauth-protected-resource"
            )
            if pr.status_code == 200:
                servers = pr.json().get("authorization_servers") or []
                for auth_server in servers:
                    auth_server = str(auth_server).rstrip("/")
                    candidates.insert(
                        0,
                        f"{auth_server}/.well-known/oauth-authorization-server",
                    )
        except Exception:
            pass

        for url in candidates:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    endpoint = resp.json().get("token_endpoint")
                    if endpoint:
                        return endpoint
            except Exception:
                continue

    return None


class _ClientCredentialsToken:
    """Caches a client-credentials access token and refreshes it on expiry."""

    def __init__(self, oauth: MCPOAuthConfig, server_url: str):
        self.oauth = oauth
        self.server_url = server_url
        self._token: Optional[str] = None
        self._expires_at: float = 0.0
        self._lock = asyncio.Lock()

    async def get(self, timeout: float) -> str:
        async with self._lock:
            if self._token and time.time() < self._expires_at - 30:
                return self._token

            import httpx

            token_url = _resolve_secret(
                self.oauth.token_url
            ) or await _discover_token_endpoint(
                self.server_url, timeout
            )
            if not token_url:
                raise AgentMCPConnectionError(
                    "client_credentials OAuth requires a token endpoint. Set "
                    "MCPOAuthConfig(token_url=...) — discovery via "
                    "/.well-known/oauth-authorization-server found nothing."
                )

            client_id = _resolve_secret(self.oauth.client_id)
            client_secret = _resolve_secret(self.oauth.client_secret)
            if not client_id:
                raise AgentMCPConnectionError(
                    "client_credentials OAuth requires client_id."
                )

            data = {
                "grant_type": "client_credentials",
                "client_id": client_id,
            }
            if client_secret:
                data["client_secret"] = client_secret
            if self.oauth.scopes:
                data["scope"] = " ".join(self.oauth.scopes)

            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(token_url, data=data)
                if resp.status_code >= 400:
                    raise AgentMCPConnectionError(
                        f"client_credentials token request failed "
                        f"({resp.status_code}): {resp.text[:300]}"
                    )
                payload = resp.json()

            token = payload.get("access_token")
            if not token:
                raise AgentMCPConnectionError(
                    "Token endpoint response contained no access_token."
                )
            self._token = token
            self._expires_at = time.time() + float(
                payload.get("expires_in", 3600)
            )
            return token


########################################################
# Manager
########################################################


class MCPManager:
    """
    Owns every MCP server an agent talks to.

    Responsibilities:

    * normalize whatever the caller passed (``mcp_url``, ``mcp_urls``,
      ``mcp_config``, ``mcp_configs``) into a list of :class:`MCPConnection`
    * resolve authentication per server (API key, bearer token, OAuth 2.1)
    * discover tools and expose them as OpenAI function-calling schemas
    * execute LLM tool calls against the server that owns each tool

    Args:
        mcp_url: A single server URL or ``MCPConnection``.
        mcp_urls: Several server URLs (or ``MCPConnection`` objects).
        mcp_config: A single ``MCPConnection``/dict.
        mcp_configs: Several ``MCPConnection``/dict objects.
        api_key: Default API key applied to servers that do not define one.
        authorization_token: Default bearer token for servers without one.
        oauth: Default OAuth config applied to servers without one.
        headers: Default extra headers merged into every request.
        transport: Force a transport for every server ("streamable_http",
            "sse", "stdio", or "auto").
        timeout: Default request timeout in seconds.
        agent_name: Used in log lines only.
        verbose: Emit per-call debug logging.
        retry_attempts: Retries per network operation, with backoff.
    """

    def __init__(
        self,
        mcp_url: Optional[Union[str, MCPConnection, Dict]] = None,
        mcp_urls: Optional[
            List[Union[str, MCPConnection, Dict]]
        ] = None,
        mcp_config: Optional[Union[MCPConnection, Dict]] = None,
        mcp_configs: Optional[
            List[Union[MCPConnection, Dict]]
        ] = None,
        api_key: Optional[str] = None,
        authorization_token: Optional[str] = None,
        oauth: Optional[Union[MCPOAuthConfig, Dict]] = None,
        headers: Optional[Dict[str, str]] = None,
        transport: Optional[str] = None,
        timeout: Optional[int] = None,
        agent_name: str = "agent",
        verbose: bool = False,
        retry_attempts: int = 3,
    ):
        self.agent_name = agent_name
        self.verbose = verbose
        self.retry_attempts = max(1, int(retry_attempts or 1))

        self.default_api_key = api_key
        self.default_authorization_token = authorization_token
        self.default_oauth = self._coerce_oauth(oauth)
        self.default_headers = headers
        self.default_transport = transport
        self.default_timeout = timeout

        self.connections: List[MCPConnection] = (
            self._build_connections(
                mcp_url=mcp_url,
                mcp_urls=mcp_urls,
                mcp_config=mcp_config,
                mcp_configs=mcp_configs,
            )
        )

        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._tool_routes: Dict[str, MCPConnection] = {}
        self._oauth_providers: Dict[str, Any] = {}
        self._client_credentials: Dict[
            str, _ClientCredentialsToken
        ] = {}

    ####################################################
    # Config normalization
    ####################################################

    @staticmethod
    def _coerce_oauth(
        oauth: Optional[Union[MCPOAuthConfig, Dict]],
    ) -> Optional[MCPOAuthConfig]:
        if oauth is None:
            return None
        if isinstance(oauth, MCPOAuthConfig):
            return oauth
        if isinstance(oauth, dict):
            return MCPOAuthConfig(**oauth)
        raise AgentMCPConnectionError(
            f"oauth must be an MCPOAuthConfig or dict, got {type(oauth).__name__}"
        )

    def _coerce_connection(
        self, item: Union[str, MCPConnection, Dict]
    ) -> MCPConnection:
        if isinstance(item, MCPConnection):
            connection = item
        elif isinstance(item, dict):
            connection = MCPConnection(**item)
        elif isinstance(item, str):
            connection = MCPConnection(url=item)
        else:
            raise AgentMCPConnectionError(
                "MCP servers must be given as a URL string, a dict, or an "
                f"MCPConnection — got {type(item).__name__}."
            )

        # Apply manager-level defaults where the connection is silent.
        if (
            connection.oauth is None
            and self.default_oauth is not None
        ):
            connection.oauth = self.default_oauth
        elif isinstance(connection.oauth, dict):
            connection.oauth = MCPOAuthConfig(**connection.oauth)

        if connection.api_key is None and self.default_api_key:
            connection.api_key = self.default_api_key

        if (
            connection.authorization_token is None
            and self.default_authorization_token
        ):
            connection.authorization_token = (
                self.default_authorization_token
            )

        if self.default_headers:
            merged = dict(self.default_headers)
            merged.update(connection.headers or {})
            connection.headers = merged

        if self.default_transport:
            connection.transport = self.default_transport

        if self.default_timeout is not None:
            connection.timeout = self.default_timeout

        return connection

    def _build_connections(
        self,
        mcp_url=None,
        mcp_urls=None,
        mcp_config=None,
        mcp_configs=None,
    ) -> List[MCPConnection]:
        raw: List[Union[str, MCPConnection, Dict]] = []

        for item in (mcp_url, mcp_config):
            if item is not None:
                raw.append(item)

        for group in (mcp_urls, mcp_configs):
            if group:
                if isinstance(group, (str, dict, MCPConnection)):
                    raw.append(group)
                else:
                    raw.extend(group)

        connections: List[MCPConnection] = []
        seen: set = set()
        for item in raw:
            connection = self._coerce_connection(item)
            key = self._connection_key(connection)
            if key in seen:
                continue
            seen.add(key)
            connections.append(connection)

        return connections

    @staticmethod
    def _connection_key(connection: MCPConnection) -> str:
        if connection.command:
            return f"stdio::{connection.command}::{' '.join(connection.args or [])}"
        return f"http::{connection.url}"

    def label(self, connection: MCPConnection) -> str:
        return (
            connection.name
            or connection.url
            or connection.command
            or "mcp-server"
        )

    ####################################################
    # Public surface
    ####################################################

    @property
    def enabled(self) -> bool:
        """True when at least one MCP server is configured."""
        return len(self.connections) > 0

    def __bool__(self) -> bool:
        return self.enabled

    def __len__(self) -> int:
        return len(self.connections)

    def __repr__(self) -> str:
        return (
            f"MCPManager(agent={self.agent_name!r}, "
            f"servers={[self.label(c) for c in self.connections]})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable, secret-redacted view of the configuration."""
        servers = []
        for connection in self.connections:
            servers.append(
                {
                    "name": self.label(connection),
                    "url": connection.url,
                    "transport": self._resolve_transport(connection),
                    "auth_type": self._resolve_auth_type(connection),
                    "timeout": connection.timeout,
                }
            )
        return {"enabled": self.enabled, "servers": servers}

    def add_server(
        self, server: Union[str, MCPConnection, Dict]
    ) -> None:
        """Register another MCP server and invalidate the tool cache."""
        connection = self._coerce_connection(server)
        key = self._connection_key(connection)
        if key in {self._connection_key(c) for c in self.connections}:
            return
        self.connections.append(connection)
        self.clear_cache()

    def clear_cache(self) -> None:
        """Drop cached tool schemas and routing information."""
        self._tools_cache = None
        self._tool_routes = {}

    def clear_auth_cache(self) -> None:
        """Forget in-process OAuth providers and cached client tokens."""
        self._oauth_providers = {}
        self._client_credentials = {}

    ####################################################
    # Tool discovery
    ####################################################

    def get_tools(
        self,
        format: Literal["openai", "mcp"] = "openai",
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Fetch tools from every configured server (synchronous).

        Returns OpenAI function-calling schemas by default, ready to be handed
        to the LLM. Results are cached; pass ``force_refresh=True`` to re-fetch.
        """
        if (
            format == "openai"
            and self._tools_cache is not None
            and not force_refresh
        ):
            return self._tools_cache

        return run_async(
            self.aget_tools(
                format=format, force_refresh=force_refresh
            )
        )

    async def aget_tools(
        self,
        format: Literal["openai", "mcp"] = "openai",
        force_refresh: bool = False,
    ) -> List[Dict[str, Any]]:
        """Async version of :meth:`get_tools`."""
        if not self.enabled:
            return []

        if (
            format == "openai"
            and self._tools_cache is not None
            and not force_refresh
        ):
            return self._tools_cache

        results = await asyncio.gather(
            *[
                self._alist_tools(connection, format=format)
                for connection in self.connections
            ],
            return_exceptions=True,
        )

        tools: List[Dict[str, Any]] = []
        routes: Dict[str, MCPConnection] = {}
        failures: List[str] = []

        for connection, result in zip(self.connections, results):
            label = self.label(connection)
            if isinstance(result, BaseException):
                failures.append(f"{label}: {result}")
                logger.error(
                    f"MCP [{self.agent_name}]: failed to load tools from {label}: {result}"
                )
                continue

            for tool in result:
                name = self._tool_name(tool)
                if name and name in routes:
                    logger.warning(
                        f"MCP [{self.agent_name}]: duplicate tool '{name}' from "
                        f"{label}; keeping the one from "
                        f"{self.label(routes[name])}."
                    )
                    continue
                if name:
                    routes[name] = connection
                tools.append(tool)

        if not tools and failures:
            raise AgentMCPConnectionError(
                "Could not load tools from any MCP server -> "
                + "; ".join(failures)
            )

        if format == "openai":
            self._tools_cache = tools
            self._tool_routes = routes

        if self.verbose:
            logger.info(
                f"MCP [{self.agent_name}]: loaded {len(tools)} tools from "
                f"{len(self.connections)} server(s)."
            )

        return tools

    def list_tool_names(self) -> List[str]:
        """Names of every tool exposed by the configured servers."""
        if self._tools_cache is None:
            self.get_tools()
        return [
            name for name in self._tool_routes.keys() if name
        ] or [self._tool_name(t) for t in (self._tools_cache or [])]

    @staticmethod
    def _tool_name(tool: Any) -> Optional[str]:
        if isinstance(tool, dict):
            function = tool.get("function") or {}
            return function.get("name") or tool.get("name")
        if isinstance(tool, MCPTool):
            return tool.name
        return getattr(tool, "name", None)

    async def _alist_tools(
        self, connection: MCPConnection, format: str = "openai"
    ) -> List[Any]:
        async def _op():
            async with self._session(connection) as session:
                result = await session.list_tools()
                if format == "openai":
                    return [
                        self._mcp_tool_to_openai(tool)
                        for tool in result.tools
                    ]
                return list(result.tools)

        return await self._with_retries(
            _op, description=f"list_tools({self.label(connection)})"
        )

    @staticmethod
    def _mcp_tool_to_openai(tool: MCPTool) -> Dict[str, Any]:
        """Convert an MCP tool definition into an OpenAI tool schema."""
        # mcp 2.x renamed `inputSchema` to `input_schema`.
        parameters = (
            getattr(tool, "inputSchema", None)
            or getattr(tool, "input_schema", None)
            or {
                "type": "object",
                "properties": {},
            }
        )
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": parameters,
                "strict": False,
            },
        }

    ####################################################
    # Tool execution
    ####################################################

    def execute_tool_calls(
        self,
        response: Any,
        output_type: Literal["json", "dict", "str"] = "dict",
    ) -> Union[List[Dict[str, Any]], str]:
        """
        Execute the tool calls contained in an LLM response (synchronous).

        Each call is routed to the server that advertised the tool. Results are
        returned in the order the calls appeared.
        """
        return run_async(
            self.aexecute_tool_calls(
                response, output_type=output_type
            )
        )

    async def aexecute_tool_calls(
        self,
        response: Any,
        output_type: Literal["json", "dict", "str"] = "dict",
    ) -> Union[List[Dict[str, Any]], str]:
        """Async version of :meth:`execute_tool_calls`."""
        if not self.enabled:
            raise AgentMCPConnectionError(
                "No MCP servers are configured for this agent."
            )

        calls = self._normalize_tool_calls(response)
        if not calls:
            if self.verbose:
                logger.info(
                    f"MCP [{self.agent_name}]: response contained no tool calls."
                )
            return [] if output_type != "json" else "[]"

        if not self._tool_routes:
            await self.aget_tools()

        # Group calls per server so one session serves all of its calls.
        grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
        results: List[Optional[Dict[str, Any]]] = [None] * len(calls)

        for index, call in enumerate(calls):
            name = call["name"]
            connection = self._route(name)
            if connection is None:
                results[index] = {
                    "tool": name,
                    "server": None,
                    "arguments": call["arguments"],
                    "is_error": True,
                    "error": (
                        f"No configured MCP server exposes a tool named '{name}'. "
                        f"Available tools: {', '.join(self._tool_routes) or 'none'}"
                    ),
                }
                continue
            grouped.setdefault(
                self._connection_key(connection), []
            ).append((index, call))

        async def _run_group(items):
            connection = self._route(items[0][1]["name"])
            try:
                async with self._session(connection) as session:
                    for index, call in items:
                        results[index] = await self._acall_tool(
                            session, connection, call
                        )
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except BaseException as e:
                message = _describe_exception(e)
                logger.error(
                    f"MCP [{self.agent_name}]: session error on "
                    f"{self.label(connection)}: {message}\n{traceback.format_exc()}"
                )
                for index, call in items:
                    if results[index] is None:
                        results[index] = {
                            "tool": call["name"],
                            "server": self.label(connection),
                            "arguments": call["arguments"],
                            "is_error": True,
                            "error": message,
                        }

        await asyncio.gather(
            *[_run_group(items) for items in grouped.values()]
        )

        final = [r for r in results if r is not None]
        return self.format_results(final, output_type=output_type)

    async def _acall_tool(
        self,
        session: ClientSession,
        connection: MCPConnection,
        call: Dict[str, Any],
    ) -> Dict[str, Any]:
        name = call["name"]
        arguments = call["arguments"]

        if self.verbose:
            logger.info(
                f"MCP [{self.agent_name}]: calling '{name}' on "
                f"{self.label(connection)} with {arguments}"
            )

        try:
            result = await session.call_tool(
                name=name, arguments=arguments
            )
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException as e:
            message = _describe_exception(e)
            logger.error(
                f"MCP [{self.agent_name}]: tool '{name}' failed on "
                f"{self.label(connection)}: {message}"
            )
            return {
                "tool": name,
                "server": self.label(connection),
                "arguments": arguments,
                "is_error": True,
                "error": message,
            }

        return {
            "tool": name,
            "server": self.label(connection),
            "arguments": arguments,
            "is_error": bool(getattr(result, "isError", False)),
            "result": self._extract_result(result),
        }

    def call_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a single MCP tool by name (synchronous convenience helper)."""
        return run_async(self.acall_tool(name, arguments))

    async def acall_tool(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Call a single MCP tool by name."""
        if not self._tool_routes:
            await self.aget_tools()

        connection = self._route(name)
        if connection is None:
            raise AgentMCPToolError(
                f"No configured MCP server exposes a tool named '{name}'."
            )

        call = {"name": name, "arguments": arguments or {}}
        async with self._session(connection) as session:
            return await self._acall_tool(session, connection, call)

    def _route(self, name: str) -> Optional[MCPConnection]:
        if name in self._tool_routes:
            return self._tool_routes[name]
        # Single-server setups do not need a discovery round-trip.
        if len(self.connections) == 1:
            return self.connections[0]
        return None

    @staticmethod
    def _extract_result(result: Any) -> Any:
        """
        Pull the useful payload out of an MCP ``CallToolResult``.

        Text blocks win over ``structuredContent`` because servers serialize
        the same value into both, and the text form is what an LLM reads best
        (FastMCP, for instance, wraps scalars as ``{"result": 42}``).
        """
        texts: List[str] = []
        others: List[Any] = []
        for block in getattr(result, "content", None) or []:
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
            else:
                try:
                    others.append(block.model_dump(mode="json"))
                except Exception:
                    others.append(str(block))

        if texts and not others:
            return texts[0] if len(texts) == 1 else "\n".join(texts)
        if texts or others:
            return {"text": "\n".join(texts), "content": others}

        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured

        try:
            return result.model_dump(mode="json")
        except Exception:
            return str(result)

    @staticmethod
    def format_results(
        results: List[Dict[str, Any]],
        output_type: Literal["json", "dict", "str"] = "dict",
    ) -> Union[List[Dict[str, Any]], str]:
        """Render tool results as raw dicts, a JSON string, or plain text."""
        if output_type == "json":
            return json.dumps(results, indent=2, default=str)
        if output_type == "str":
            lines = []
            for item in results:
                header = f"{item.get('tool')} @ {item.get('server')}"
                if item.get("is_error"):
                    lines.append(
                        f"{header} -> ERROR: {item.get('error')}"
                    )
                else:
                    payload = item.get("result")
                    if not isinstance(payload, str):
                        payload = json.dumps(payload, default=str)
                    lines.append(f"{header} -> {payload}")
            return "\n\n".join(lines)
        return results

    @staticmethod
    def _normalize_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """
        Flatten whatever the LLM returned into ``[{"name", "arguments"}, ...]``.

        Accepts a JSON string, a single tool call, a list of tool calls, a full
        chat-completion message, and both dict and pydantic/OpenAI object forms.
        """
        if response is None:
            return []

        if isinstance(response, str):
            try:
                response = json.loads(response)
            except json.JSONDecodeError:
                return []

        if not isinstance(response, (list, tuple)):
            response = [response]

        calls: List[Dict[str, Any]] = []

        def _add(item: Any) -> None:
            if item is None:
                return

            # Pydantic / OpenAI SDK objects -> dict
            if not isinstance(item, dict):
                if hasattr(item, "model_dump"):
                    try:
                        item = item.model_dump()
                    except Exception:
                        item = getattr(item, "__dict__", {})
                elif hasattr(item, "__dict__"):
                    item = dict(item.__dict__)
                else:
                    return

            # A whole assistant message
            nested = item.get("tool_calls") or item.get("toolCalls")
            if nested:
                for call in nested:
                    _add(call)
                return

            function = item.get("function") or item
            name = function.get("name") or item.get("name")
            if not name:
                return

            arguments = function.get(
                "arguments", function.get("parameters", {})
            )
            if isinstance(arguments, str):
                try:
                    arguments = (
                        json.loads(arguments) if arguments else {}
                    )
                except json.JSONDecodeError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            calls.append({"name": name, "arguments": arguments})

        for item in response:
            _add(item)

        return calls

    ####################################################
    # Transport + auth
    ####################################################

    def _resolve_transport(self, connection: MCPConnection) -> str:
        transport = (connection.transport or "auto").replace("-", "_")

        if connection.command:
            return "stdio"

        if transport not in ("auto", "", None):
            return transport

        url = connection.url or ""
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            if any(
                url.rstrip("/").endswith(hint.rstrip("/"))
                for hint in _SSE_URL_HINTS
            ):
                return "sse"
            return "streamable_http"
        if not parsed.scheme:
            return "stdio"
        return "streamable_http"

    def _resolve_auth_type(self, connection: MCPConnection) -> str:
        if connection.auth_type:
            return connection.auth_type
        if connection.oauth is not None:
            return "oauth"
        if connection.api_key:
            return "api_key"
        if connection.authorization_token:
            return "bearer"
        if connection.headers:
            return "custom"
        return "none"

    async def _build_headers(
        self, connection: MCPConnection
    ) -> Dict[str, str]:
        """Assemble request headers, resolving env-indirected secrets."""
        headers: Dict[str, str] = dict(connection.headers or {})
        auth_type = self._resolve_auth_type(connection)

        if auth_type != "oauth":
            api_key = _resolve_secret(connection.api_key)
            if api_key:
                prefix = connection.api_key_prefix or ""
                header = connection.api_key_header or "Authorization"
                headers[header] = (
                    f"{prefix} {api_key}".strip()
                    if prefix
                    else api_key
                )

            token = _resolve_secret(connection.authorization_token)
            if token and "Authorization" not in headers:
                headers["Authorization"] = f"Bearer {token}"

        if auth_type == "oauth" and connection.oauth is not None:
            oauth = connection.oauth

            static_token = _resolve_secret(oauth.access_token)
            if static_token:
                headers["Authorization"] = f"Bearer {static_token}"
            elif oauth.grant_type == "client_credentials":
                key = self._connection_key(connection)
                if key not in self._client_credentials:
                    self._client_credentials[key] = (
                        _ClientCredentialsToken(
                            oauth, connection.url or ""
                        )
                    )
                token = await self._client_credentials[key].get(
                    float(connection.timeout or 30)
                )
                headers["Authorization"] = f"Bearer {token}"

        return headers

    def _build_oauth_provider(
        self, connection: MCPConnection
    ) -> Optional[Any]:
        """
        Build an ``httpx.Auth`` that runs the OAuth 2.1 authorization-code flow.

        Returns ``None`` when the connection does not need the interactive flow
        (no OAuth configured, a static token was supplied, or the headless
        client-credentials grant is in use — both of those become headers).
        """
        oauth = connection.oauth
        if oauth is None:
            return None
        if _resolve_secret(oauth.access_token):
            return None
        if oauth.grant_type != "authorization_code":
            return None

        key = self._connection_key(connection)
        if key in self._oauth_providers:
            return self._oauth_providers[key]

        try:
            from mcp.client.auth import OAuthClientProvider
            from mcp.shared.auth import OAuthClientMetadata
        except ImportError as e:
            raise AgentMCPConnectionError(
                "OAuth support requires a recent MCP SDK "
                f"(pip install -U mcp): {e}"
            )

        storage = (
            MCPFileTokenStorage(
                connection.url or key, oauth.token_storage_path
            )
            if oauth.use_token_cache
            else MCPInMemoryTokenStorage()
        )

        metadata_kwargs: Dict[str, Any] = {
            "client_name": oauth.client_name,
            "redirect_uris": [oauth.redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": (
                "client_secret_post"
                if _resolve_secret(oauth.client_secret)
                else "none"
            ),
        }
        if oauth.scopes:
            metadata_kwargs["scope"] = " ".join(oauth.scopes)
        if oauth.client_uri:
            metadata_kwargs["client_uri"] = oauth.client_uri

        client_metadata = OAuthClientMetadata(**metadata_kwargs)

        callback_server = _OAuthCallbackServer(oauth.redirect_uri)
        label = self.label(connection)

        async def redirect_handler(authorization_url: str) -> None:
            callback_server.start()
            logger.info(
                f"MCP [{self.agent_name}]: authorize access to {label} at:\n"
                f"{authorization_url}"
            )
            if oauth.open_browser:
                try:
                    import webbrowser

                    webbrowser.open(authorization_url)
                except Exception as e:  # headless environments
                    logger.warning(
                        f"MCP: could not open a browser ({e}). Open the URL above manually."
                    )

        async def callback_handler() -> Tuple[str, Optional[str]]:
            return await asyncio.to_thread(
                callback_server.wait, float(oauth.callback_timeout)
            )

        provider = OAuthClientProvider(
            server_url=_server_origin(connection.url or ""),
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=redirect_handler,
            callback_handler=callback_handler,
            timeout=float(oauth.callback_timeout),
        )

        self._oauth_providers[key] = provider
        return provider

    def _client_context(
        self, connection: MCPConnection, headers: Dict[str, str], auth
    ):
        """Return the transport-specific async client context manager."""
        transport = self._resolve_transport(connection)
        timeout = float(connection.timeout or 30)
        sse_read_timeout = float(connection.sse_read_timeout or 300)

        if transport == "stdio":
            try:
                from mcp.client.stdio import (
                    StdioServerParameters,
                    stdio_client,
                )
            except ImportError as e:
                raise AgentMCPConnectionError(
                    f"stdio transport is unavailable in this MCP SDK: {e}"
                )

            command = connection.command
            args = list(connection.args or [])
            if not command:
                # Allow "python server.py" style strings in `url`.
                parts = (connection.url or "").split()
                if not parts:
                    raise AgentMCPConnectionError(
                        "stdio transport requires `command` (and optionally `args`)."
                    )
                command, args = parts[0], parts[1:]

            env = dict(os.environ)
            for name, value in (connection.env or {}).items():
                resolved = _resolve_secret(value)
                if resolved is not None:
                    env[name] = resolved

            return stdio_client(
                StdioServerParameters(
                    command=command, args=args, env=env
                )
            )

        if not connection.url:
            raise AgentMCPConnectionError(
                "An MCP server URL is required for HTTP transports."
            )

        if transport == "sse":
            try:
                from mcp.client.sse import sse_client
            except ImportError as e:
                raise AgentMCPConnectionError(
                    f"SSE transport is unavailable in this MCP SDK: {e}"
                )
            return sse_client(
                url=connection.url,
                headers=headers or None,
                timeout=timeout,
                sse_read_timeout=sse_read_timeout,
                auth=auth,
            )

        try:
            # mcp 1.x
            from mcp.client.streamable_http import (
                streamablehttp_client,
            )
        except ImportError:
            # mcp 2.x renamed the factory and changed its signature: headers,
            # timeout and auth now go into an http client that is passed in,
            # rather than being arguments of the transport itself.
            from mcp.client.streamable_http import (
                create_mcp_http_client,
                streamable_http_client,
            )

            return streamable_http_client(
                url=connection.url,
                http_client=create_mcp_http_client(
                    headers=headers or None,
                    timeout=_http_timeout(timeout, sse_read_timeout),
                    auth=auth,
                ),
            )

        return streamablehttp_client(
            url=connection.url,
            headers=headers or None,
            timeout=timedelta(seconds=timeout),
            sse_read_timeout=timedelta(seconds=sse_read_timeout),
            auth=auth,
        )

    @asynccontextmanager
    async def _session(self, connection: MCPConnection):
        """
        Open an authenticated, initialized MCP session for one server.

        Entering and exiting happen inside a single ``async with`` so anyio's
        task groups unwind correctly; connection failures surface as
        :class:`AgentMCPConnectionError` with the underlying cause attached.
        """
        label = self.label(connection)
        detail = (
            f"transport '{self._resolve_transport(connection)}', "
            f"auth '{self._resolve_auth_type(connection)}'"
        )

        try:
            headers = await self._build_headers(connection)
            auth = self._build_oauth_provider(connection)
            client_cm = self._client_context(
                connection, headers, auth
            )
        except AgentMCPConnectionError:
            raise
        except Exception as e:
            raise AgentMCPConnectionError(
                f"Could not prepare a connection to MCP server "
                f"'{label}' ({detail}): {_describe_exception(e)}"
            )

        try:
            async with client_cm as ctx:
                read, write = ctx[0], ctx[1]
                async with ClientSession(
                    read,
                    write,
                    read_timeout_seconds=_read_timeout(
                        float(
                            connection.tool_timeout
                            or connection.timeout
                            or 120
                        )
                    ),
                ) as session:
                    await session.initialize()
                    yield session
        except (AgentMCPConnectionError, AgentMCPToolError):
            raise
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except BaseException as e:
            # anyio surfaces transport failures as (Base)ExceptionGroups whose
            # str() is empty, so flatten them into something actionable.
            raise AgentMCPConnectionError(
                f"Failed to connect to MCP server '{label}' ({detail}): "
                f"{_describe_exception(e)}"
            ) from e

    ####################################################
    # Retries
    ####################################################

    async def _with_retries(self, op, description: str = "") -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(self.retry_attempts):
            try:
                return await op()
            except AgentMCPToolError:
                raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except BaseException as e:
                last_error = e
                if attempt == self.retry_attempts - 1:
                    break
                delay = 2**attempt
                logger.warning(
                    f"MCP [{self.agent_name}]: {description} failed "
                    f"(attempt {attempt + 1}/{self.retry_attempts}): "
                    f"{_describe_exception(e)}. Retrying in {delay}s."
                )
                await asyncio.sleep(delay)

        raise AgentMCPConnectionError(
            f"MCP operation {description} failed after "
            f"{self.retry_attempts} attempt(s): "
            f"{_describe_exception(last_error)}"
        )
