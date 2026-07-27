"""
Regression tests for the SSRF guard on remote image/audio fetching.

``_is_safe_url`` gates every URL that ``get_image_base64`` (and the audio path
in ``litellm_wrapper``) will fetch. Those fetchers are reachable from
``agent.run(img=<url>)`` with an end-user-supplied value, so a weak guard is a
server-side request forgery hole into cloud metadata and internal services.

The original guard only inspected *literal* IPs and passed every hostname
straight through, and it never resolved names — so ``2130706433`` (decimal
127.0.0.1), ``metadata.google.internal``, and wildcard-DNS names like
``127.0.0.1.nip.io`` all sailed past. These tests lock the closures shut.

Run:
    pytest tests/utils/test_ssrf_url_guard.py -v
"""

import ipaddress

import pytest

from swarms.utils.image_file_b64 import _ip_is_blocked, _is_safe_url


########################################################
# Must be blocked
########################################################

BLOCKED = [
    # Cloud metadata endpoints — the crown jewels of SSRF.
    "http://169.254.169.254/latest/meta-data/",
    "http://[fd00:ec2::254]/latest/meta-data/",
    # Loopback in every notation.
    "http://127.0.0.1:8000/admin",
    "http://localhost/",
    "http://LocalHost/",
    "http://2130706433/",  # decimal 127.0.0.1
    "http://0x7f000001/",  # hex 127.0.0.1
    "http://[::1]/",  # IPv6 loopback
    # IPv4 embedded in IPv6.
    "http://[::ffff:169.254.169.254]/",
    "http://[::ffff:127.0.0.1]/",
    # Private ranges.
    "http://10.0.0.5/",
    "http://192.168.1.1/",
    "http://172.16.0.1/",
    # Link-local and unspecified.
    "http://169.254.1.1/",
    "http://0.0.0.0/",
    # Wildcard-DNS names that resolve into blocked space.
    "http://127.0.0.1.nip.io/",
    # Non-HTTP schemes must never be fetched.
    "file:///etc/passwd",
    "ftp://internal/secret",
    "gopher://127.0.0.1:6379/_INFO",
    "http:///nohost",
]


@pytest.mark.parametrize("url", BLOCKED)
def test_blocked_urls_are_rejected(url):
    assert _is_safe_url(url) is False, f"SSRF guard allowed {url!r}"


########################################################
# Must be allowed
########################################################

ALLOWED = [
    "http://example.com/image.png",
    "https://example.com/image.png",
    "https://raw.githubusercontent.com/org/repo/main/x.png",
    "https://upload.wikimedia.org/wikipedia/commons/x.jpg",
]


@pytest.mark.parametrize("url", ALLOWED)
def test_public_urls_are_allowed(url):
    assert (
        _is_safe_url(url) is True
    ), f"SSRF guard blocked public {url!r}"


########################################################
# The address classifier
########################################################


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "169.254.169.254",
        "10.1.2.3",
        "192.168.0.1",
        "172.31.255.255",
        "0.0.0.0",
        "::1",
        "fe80::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
    ],
)
def test_ip_is_blocked_true(ip):
    assert _ip_is_blocked(ipaddress.ip_address(ip)) is True


@pytest.mark.parametrize(
    "ip", ["8.8.8.8", "1.1.1.1", "93.184.216.34"]
)
def test_ip_is_blocked_false(ip):
    assert _ip_is_blocked(ipaddress.ip_address(ip)) is False


########################################################
# Malformed input degrades to "unsafe", never raises
########################################################


@pytest.mark.parametrize(
    "value", ["", "not a url", "http://", "://x", None]
)
def test_malformed_is_unsafe(value):
    # None would raise inside urlparse; the guard must swallow and reject.
    assert _is_safe_url(value if value is not None else "") is False
