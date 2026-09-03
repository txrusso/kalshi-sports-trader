"""Load Kalshi API credentials from the local key file.

Point KALSHI_KEY_FILE at a file of the form:

    APIkeyid = <uuid>
    Privatekey = -----BEGIN RSA PRIVATE KEY-----
    <base64 lines>
    -----END RSA PRIVATE KEY-----

Keep it outside the repo. Secrets are read at runtime and never logged or printed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

# Kept outside the repo so it can never be committed: a sibling of the project
# directory. Override with the KALSHI_KEY_FILE environment variable.
DEFAULT_KEY_FILE = str(Path(__file__).resolve().parents[2] / "Kal_API.txt")


@dataclass
class KalshiCredentials:
    key_id: str
    private_key: RSAPrivateKey

    def __repr__(self) -> str:  # never leak the key material in logs
        return f"KalshiCredentials(key_id={self.key_id[:8]}..., private_key=<hidden>)"


def _extract_pem(raw: str) -> str:
    """Pull the PEM block out of the key file, tolerating the 'Privatekey =' prefix."""
    begin = raw.find("-----BEGIN")
    end_marker = "-----END RSA PRIVATE KEY-----"
    end = raw.find(end_marker)
    if begin == -1 or end == -1:
        # Also accept generic PKCS#8 header
        end_marker = "-----END PRIVATE KEY-----"
        end = raw.find(end_marker)
    if begin == -1 or end == -1:
        raise ValueError("Could not find an RSA PRIVATE KEY block in the key file.")
    return raw[begin : end + len(end_marker)] + "\n"


def load_credentials(key_file: str | None = None) -> KalshiCredentials:
    path = Path(key_file or os.environ.get("KALSHI_KEY_FILE", DEFAULT_KEY_FILE))
    if not path.exists():
        raise FileNotFoundError(f"Kalshi key file not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")

    key_id = None
    for line in raw.splitlines():
        low = line.lower()
        if low.strip().startswith("apikeyid"):
            key_id = line.split("=", 1)[1].strip()
            break
    if not key_id:
        raise ValueError("Could not find 'APIkeyid = ...' in the key file.")

    pem = _extract_pem(raw)
    private_key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    if not isinstance(private_key, RSAPrivateKey):
        raise TypeError("Loaded key is not an RSA private key.")

    return KalshiCredentials(key_id=key_id, private_key=private_key)
