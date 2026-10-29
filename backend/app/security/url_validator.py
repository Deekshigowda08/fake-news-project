"""URL security, SSRF defense, and safe content extraction."""

import ipaddress
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse
import logging
import httpx

logger = logging.getLogger(__name__)

MAX_ARTICLE_RESPONSE_BYTES = 2 * 1024 * 1024  # 2 MB limit
REQUEST_TIMEOUT_SECONDS = 5.0


class SecurityValidationError(Exception):
    """Raised when a URL violates SSRF or security policies."""
    pass


def is_ip_private_or_loopback(ip_str: str) -> bool:
    """Checks if an IP address belongs to loopback, private, link-local, or multicast blocks."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        return True


def validate_safe_url(url: str) -> str:
    """Validates that a URL is safe to fetch and protects against SSRF.
    
    Rules:
    - Must be http or https scheme.
    - Resolves host DNS to ensure it does not map to localhost, 127.0.0.1, 10.0.0.0/8,
      192.168.0.0/16, 172.16.0.0/12, or cloud metadata services (169.254.169.254).
    """
    if not url:
        raise SecurityValidationError("URL cannot be empty")

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise SecurityValidationError(f"Invalid URL protocol '{parsed.scheme}'. Only HTTP and HTTPS are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SecurityValidationError("Missing hostname in URL")

    # Reject localhost names
    if hostname.lower() in ("localhost", "127.0.0.1", "::1", "metadata.google.internal"):
        raise SecurityValidationError("Access to localhost or internal services is forbidden.")

    # DNS Resolution check
    try:
        addr_info = socket.getaddrinfo(hostname, None)
        for entry in addr_info:
            ip_candidate = entry[4][0]
            if is_ip_private_or_loopback(ip_candidate):
                raise SecurityValidationError(
                    f"Destination IP '{ip_candidate}' resolved from '{hostname}' is on a private or restricted network."
                )
    except socket.gaierror as e:
        raise SecurityValidationError(f"Could not resolve domain '{hostname}': {e}")

    return url.strip()


async def fetch_article_text_safely(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Safely fetches and extracts text from an external news URL with timeout and byte limits."""
    try:
        safe_url = validate_safe_url(url)
    except SecurityValidationError as e:
        return None, str(e)

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT_SECONDS,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=5),
        ) as client:
            # First check HEAD or stream GET
            response = await client.get(
                safe_url,
                headers={"User-Agent": "MultimodalFakeNewsResearchBot/1.0 (+research-eval)"},
            )

            # Re-check redirect target URL against SSRF
            final_url = str(response.url)
            validate_safe_url(final_url)

            if response.status_code >= 400:
                return None, f"HTTP error {response.status_code} fetching article"

            content_bytes = response.content
            if len(content_bytes) > MAX_ARTICLE_RESPONSE_BYTES:
                content_bytes = content_bytes[:MAX_ARTICLE_RESPONSE_BYTES]

            raw_text = content_bytes.decode("utf-8", errors="replace")

            # Extract basic paragraph text using regex to avoid heavy parser dependency
            from ml.preprocessing.text import clean_text
            import re

            # Strip script and style
            no_scripts = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_text, flags=re.DOTALL | re.IGNORECASE)
            # Strip tags
            plain = re.sub(r"<[^>]+>", " ", no_scripts)
            cleaned = clean_text(plain)

            return cleaned[:5000], None  # Limit to 5000 characters of article body

    except httpx.TimeoutException:
        return None, "Connection timed out fetching URL"
    except Exception as e:
        logger.warning("Safe article fetch failed for %s: %s", url, e)
        return None, f"Failed to retrieve article content: {e}"
