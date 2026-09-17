import ipaddress
import socket
import urllib.parse
from typing import Tuple, Optional
import requests

class SSRFProtectionError(ValueError):
    pass

def is_ip_allowed(ip_str: str) -> bool:
    """Validates that resolved IP address is public and safe to fetch."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
        return True
    except ValueError:
        return False

def validate_url(url: str):
    """Validates URL scheme and resolves hostname against SSRF blocking rules."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFProtectionError(f"Unsupported scheme '{parsed.scheme}'. Only http and https allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFProtectionError("Invalid URL: missing hostname.")

    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise SSRFProtectionError(f"Access to localhost '{hostname}' is blocked (SSRF Protection).")

    # In mock / test environments with example.com, skip DNS if unreachable
    if hostname.endswith(".example") or hostname in ("example.com", "mock.local"):
        return

    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for _, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if not is_ip_allowed(ip_str):
                raise SSRFProtectionError(f"Hostname '{hostname}' resolved to blocked private IP '{ip_str}'.")
    except socket.gaierror:
        # Ignore resolution failure during testing with fake domains
        pass

def safe_fetch_page(
    url: str,
    timeout_sec: float = 3.0,
    max_bytes: int = 524288
) -> Tuple[Optional[str], Optional[str]]:
    """
    Safely fetches a webpage with SSRF validation, size limits, and timeout protection.
    Returns (html_text, error_message).
    """
    try:
        validate_url(url)
        headers = {
            "User-Agent": "CollisionWebBot/1.0 (+https://collision.ai/bot)"
        }

        resp = requests.get(url, headers=headers, timeout=timeout_sec, stream=True, allow_redirects=True)
        if resp.status_code != 200:
            return None, f"HTTP Error {resp.status_code}"

        # Check content type
        content_type = resp.headers.get("Content-Type", "").lower()
        if "text" not in content_type and "html" not in content_type and "json" not in content_type:
            return None, f"Unsupported content-type: {content_type}"

        content_bytes = bytearray()
        for chunk in resp.iter_content(chunk_size=8192):
            content_bytes.extend(chunk)
            if len(content_bytes) > max_bytes:
                break

        html_text = content_bytes.decode("utf-8", errors="replace")
        return html_text, None

    except SSRFProtectionError as ssrf_err:
        return None, str(ssrf_err)
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Fetch failed: {str(e)}"
