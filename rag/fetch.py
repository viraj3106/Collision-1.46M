import ipaddress
import socket
import urllib.parse
import requests
from typing import Tuple, Optional

class SSRFProtectionError(ValueError):
    pass

def is_ip_allowed(ip_str: str) -> bool:
    """
    Validates if an IP address is public and safe to fetch (blocks loopback, private, link-local, multicast).
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
        return True
    except ValueError:
        return False

def validate_url(url: str) -> Tuple[str, str]:
    """
    Validates URL scheme and checks resolved IP address against SSRF rules.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFProtectionError(f"Unsupported scheme '{parsed.scheme}'. Only http and https are allowed.")
    
    hostname = parsed.hostname
    if not hostname:
        raise SSRFProtectionError("Invalid URL: missing hostname.")
    
    # Check explicitly for localhost or internal names
    if hostname.lower() in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        raise SSRFProtectionError(f"Access to local hostname '{hostname}' is forbidden (SSRF Protection).")
    
    try:
        # Resolve hostname to IP addresses
        addr_info = socket.getaddrinfo(hostname, None)
        for family, socktype, proto, canonname, sockaddr in addr_info:
            ip_str = sockaddr[0]
            if not is_ip_allowed(ip_str):
                raise SSRFProtectionError(f"URL hostname '{hostname}' resolved to blocked private IP '{ip_str}' (SSRF Protection).")
    except socket.gaierror as e:
        raise SSRFProtectionError(f"Could not resolve hostname '{hostname}': {str(e)}")

    return parsed.scheme, hostname

def safe_fetch_webpage(url: str, timeout_sec: float = 3.0, max_bytes: int = 524288) -> Tuple[Optional[str], Optional[str]]:
    """
    Safely fetches a webpage HTML content with strict SSRF protection, timeout, and max byte size.
    Returns (html_content, error_message).
    """
    try:
        validate_url(url)
        headers = {
            "User-Agent": "CollisionBot/1.0 (+https://collision-ai.org/bot)"
        }
        
        # Use stream=True to enforce max_bytes size limit
        response = requests.get(url, headers=headers, timeout=timeout_sec, stream=True, allow_redirects=True)
        
        # Validate final URL after redirects for SSRF
        if response.url != url:
            validate_url(response.url)
            
        if response.status_code != 200:
            return None, f"HTTP Error {response.status_code}"
            
        content_bytes = bytearray()
        for chunk in response.iter_content(chunk_size=8192):
            content_bytes.extend(chunk)
            if len(content_bytes) > max_bytes:
                break
                
        html_str = content_bytes.decode("utf-8", errors="replace")
        return html_str, None
    except SSRFProtectionError as ssrf_err:
        return None, str(ssrf_err)
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, f"Fetch error: {str(e)}"
