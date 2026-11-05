import gzip
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
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


def test_zstd_body_arrives_intact(collector):
    """A body is sent zstd-compressed and decodes to the original bytes."""
    session = CompressingSession()
    payload = b"span data " * 1000

    assert session.post(collector["url"], data=payload).ok
    assert collector["seen"] == ["zstd"]
    assert collector["bodies"] == [payload]


def test_compresses_bodies_sent_through_request(collector):
    session = CompressingSession()
    payload = b"span data " * 1000

    response = session.request(
        method="POST",
        url=collector["url"],
        headers={"content-type": "application/x-protobuf"},
        data=payload,
        timeout=10,
    )

    assert response.ok
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
