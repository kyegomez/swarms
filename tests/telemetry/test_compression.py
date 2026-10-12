import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
import requests
import zstandard

from swarms.telemetry import compression
from swarms.telemetry.compression import CompressingSession


@pytest.fixture
def collector():
    """A local endpoint that records each request and answers per encoding."""
    state = {"seen": [], "bodies": [], "accept": {"zstd", "gzip"}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(
                int(self.headers["Content-Length"])
            )
            encoding = self.headers.get("Content-Encoding", "")
            state["seen"].append(encoding)
            if encoding not in state["accept"]:
                self.send_response(400)
                self.end_headers()
                return
            if encoding == "zstd":
                body = (
                    zstandard.ZstdDecompressor()
                    .stream_reader(body)
                    .read()
                )
            elif encoding == "gzip":
                body = gzip.decompress(body)
            state["bodies"].append(body)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1/traces"
    try:
        yield state
    finally:
        server.shutdown()


@pytest.mark.parametrize("method", ["POST", "post"])
def test_direct_request_body_arrives_intact(collector, method):
    """Exporter requests compress bytes without changing caller headers."""
    session = CompressingSession()
    payload = b"span data " * 1000
    headers = {"Content-Type": "application/x-protobuf"}

    assert session.request(
        method, collector["url"], data=payload, headers=headers
    ).ok
    assert collector["seen"] == ["zstd"]
    assert collector["bodies"] == [payload]
    assert headers == {"Content-Type": "application/x-protobuf"}


@pytest.mark.parametrize(
    "method,data",
    [
        ("POST", None),
        ("POST", "text"),
        ("POST", {"key": "value"}),
        ("PUT", b"bytes"),
        ("GET", b"bytes"),
    ],
)
def test_non_export_requests_pass_through(monkeypatch, method, data):
    """Only POST requests with byte bodies are compressed."""
    sent = []

    def request(self, method, url, **kwargs):
        sent.append((method, url, kwargs))
        return requests.Response()

    monkeypatch.setattr(requests.Session, "request", request)
    session = CompressingSession()
    headers = {"X-Test": "value"}

    session.request(
        method, "http://localhost/", data=data, headers=headers
    )

    assert sent == [
        (
            method,
            "http://localhost/",
            {"data": data, "headers": headers},
        )
    ]


def test_zstd_body_arrives_intact(collector):
    """A body is sent zstd-compressed and decodes to the original bytes."""
    session = CompressingSession()
    payload = b"span data " * 1000

    assert session.post(collector["url"], data=payload).ok
    assert collector["seen"] == ["zstd"]
    assert collector["bodies"] == [payload]


def test_falls_back_to_gzip_when_zstd_is_rejected(collector):
    """A collector without zstd gets the same batch in gzip, and gzip from then on."""
    collector["accept"] = {"gzip"}
    session = CompressingSession()

    assert session.post(collector["url"], data=b"first").ok
    assert session.post(collector["url"], data=b"second").ok
    assert collector["seen"] == ["zstd", "gzip", "gzip"]
    assert collector["bodies"] == [b"first", b"second"]
    assert session.encoding == "gzip"


def test_stays_on_zstd_when_gzip_fails_too(collector):
    """A rejection that gzip does not fix is not blamed on zstd."""
    collector["accept"] = set()
    session = CompressingSession()

    assert (
        session.post(collector["url"], data=b"x").status_code == 400
    )
    session.post(collector["url"], data=b"y")
    assert collector["seen"] == ["zstd", "gzip", "zstd", "gzip"]
    assert session.encoding == "zstd"


def test_uses_gzip_without_zstandard(collector, monkeypatch):
    """Without zstandard installed, every body is sent gzip-compressed."""
    monkeypatch.setattr(compression, "_zstd_compressor", lambda: None)
    session = CompressingSession()

    assert session.encoding == "gzip"
    assert session.post(collector["url"], data=b"payload").ok
    assert collector["seen"] == ["gzip"]
    assert collector["bodies"] == [b"payload"]


def test_compresses_with_a_long_window():
    """Exports use long-distance matching with a 32 MB window."""
    frame = compression._zstd_compressor().compress(b"x" * 64_000_000)
    params = zstandard.get_frame_parameters(frame)

    assert params.window_size == 32 * 1024 * 1024
