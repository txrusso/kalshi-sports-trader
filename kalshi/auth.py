"""Kalshi request signing (RSA-PSS over timestamp+method+path).

Kalshi authenticates each request with three headers:
    KALSHI-ACCESS-KEY        : the API key id
    KALSHI-ACCESS-TIMESTAMP  : current time in milliseconds (string)
    KALSHI-ACCESS-SIGNATURE  : base64( RSA-PSS-SHA256( ts + METHOD + path ) )

The signed 'path' is the URL path INCLUDING the '/trade-api/v2' prefix but
EXCLUDING the query string and host.
"""
from __future__ import annotations

import base64
import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey


def _sign(private_key: RSAPrivateKey, message: str) -> str:
    signature = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def build_headers(key_id: str, private_key: RSAPrivateKey, method: str, path: str) -> dict[str, str]:
    """Build the signed auth headers for a request.

    `path` must be the path only, e.g. '/trade-api/v2/portfolio/balance'
    (no scheme, host, or query string).
    """
    ts_ms = str(int(time.time() * 1000))
    method = method.upper()
    # Strip any query string defensively; Kalshi signs the path only.
    path_only = path.split("?", 1)[0]
    msg = ts_ms + method + path_only
    signature = _sign(private_key, msg)
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
