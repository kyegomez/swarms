"""
Guard for outbound URLs supplied by a caller.

Anything the framework fetches server-side from a user-controlled string is a
server-side request forgery surface: agent image inputs (``agent.run(img=...)``)
and marketplace skill URLs (``Agent(skill_urls=[...])``) both take one. The
guard used to live inside ``image_file_b64``; it is here so a second fetcher
can reuse it rather than import a private name across modules.
"""

import ipaddress
import socket
from urllib.parse import urlparse


def ip_is_blocked(addr: ipaddress._BaseAddress) -> bool:
    """Return True for any address that must never be reached over the network.

    Covers loopback, private (RFC 1918), link-local (incl. the 169.254.169.254
    cloud-metadata range), reserved, unspecified, and multicast space. IPv4
    addresses embedded in IPv6 (``::ffff:a.b.c.d`` and 6to4) are unwrapped and
    re-checked, so a mapped metadata address cannot slip through.
    """
    # Unwrap IPv4-in-IPv6 so ::ffff:169.254.169.254 is judged as the v4 address.
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    sixtofour = getattr(addr, "sixtofour", None)
    if sixtofour is not None:
        addr = sixtofour

    return (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_unspecified
        or addr.is_multicast
    )


def is_safe_url(url: str) -> bool:
    """
    Reject URLs that target private/link-local networks (SSRF prevention).

    Only ``http``/``https`` are permitted, and the host is **resolved** before
    the verdict: a hostname that maps to a private or cloud-metadata address is
    blocked, and *every* address it resolves to must be public (so a name with
    one public and one internal A-record cannot be used to pivot). Numeric host
    forms (decimal, hex, octal) and IPv4-in-IPv6 are normalized rather than
    trusted as opaque strings.

    Note: this checks the addresses known at call time. A determined attacker
    controlling DNS can still rebind between this check and the subsequent
    request (TOCTOU); eliminating that requires pinning the resolved IP for the
    actual connection, which the calling HTTP client does not currently do.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        host = (parsed.hostname or "").strip()
        if not host or host.lower() == "localhost":
            return False

        # If the host is already a literal IP (in any notation ipaddress
        # accepts — decimal, hex, dotted), judge it directly.
        try:
            return not ip_is_blocked(ipaddress.ip_address(host))
        except ValueError:
            pass  # a hostname — resolve it below.

        # Resolve the hostname and require EVERY answer to be public.
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return False  # cannot resolve — do not fetch.

        resolved = {info[4][0] for info in infos}
        if not resolved:
            return False

        for ip in resolved:
            try:
                if ip_is_blocked(ipaddress.ip_address(ip)):
                    return False
            except ValueError:
                return False

        return True
    except Exception:
        return False
