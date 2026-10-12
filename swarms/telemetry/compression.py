import gzip
import threading
from typing import Any, Optional

import requests

# A 32 MB window reaches back over the repeated conversation history that consecutive spans carry.
ZSTD_LEVEL = 3
ZSTD_WINDOW_LOG = 25


def _zstd_compressor() -> Optional[Any]:
    """Build the zstd compressor used for exports.

    Returns:
        Optional[Any]: A zstandard compressor, or None when zstandard is not installed.
    """
    try:
        import zstandard
    except ImportError:
        return None
    params = zstandard.ZstdCompressionParameters.from_level(
        ZSTD_LEVEL, enable_ldm=True, window_log=ZSTD_WINDOW_LOG
    )
    return zstandard.ZstdCompressor(compression_params=params)


class CompressingSession(requests.Session):
    """HTTP session that zstd-compresses export bodies and falls back to gzip.

    Attributes:
        encoding (str): The content encoding the next export is sent with.
    """

    def __init__(self) -> None:
        """Start with zstd when zstandard is installed, otherwise gzip.

        Returns:
            None
        """
        super().__init__()
        self._zstd = _zstd_compressor()
        self._lock = threading.Lock()
        self.encoding = "zstd" if self._zstd is not None else "gzip"

    def request(
        self, method: str, url: str, **kwargs: Any
    ) -> requests.Response:
        """Compress byte POST bodies from either exporter request path.

        Args:
            method (str): The HTTP request method.
            url (str): The export endpoint.
            **kwargs (Any): Request arguments, including the body in data.

        Returns:
            requests.Response: The collector's response.
        """
        data = kwargs.get("data")
        if method.upper() != "POST" or not isinstance(data, bytes):
            return super().request(method, url, **kwargs)
        kwargs.pop("data")
        if self.encoding == "zstd":
            with self._lock:
                body = self._zstd.compress(data)
            response = self._send(method, url, body, "zstd", **kwargs)
            # A collector that cannot read zstd fails to parse the body and answers 400 or 415.
            if response.status_code not in (400, 415):
                return response
            fallback = self._send(
                method, url, gzip.compress(data), "gzip", **kwargs
            )
            if fallback.ok:
                self.encoding = "gzip"
            return fallback
        return self._send(
            method, url, gzip.compress(data), "gzip", **kwargs
        )

    def _send(
        self,
        method: str,
        url: str,
        body: bytes,
        encoding: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Post an already compressed body with its content encoding.

        Args:
            method (str): The HTTP request method.
            url (str): The export endpoint.
            body (bytes): The compressed body.
            encoding (str): The content encoding of the body.
            **kwargs (Any): Passed through to the request.

        Returns:
            requests.Response: The collector's response.
        """
        headers = dict(kwargs.pop("headers", None) or {})
        headers["Content-Encoding"] = encoding
        return super().request(
            method, url, data=body, headers=headers, **kwargs
        )
