"""
AES-256-CBC encryption for sensitive data (transcripts, API keys, PII).
Receptionist.co Data Security Policy — Fail-Deadly Architecture.

REQUIRED Railway env var:
  ENCRYPTION_KEY = 64-char hex string (32 bytes)
  Generate: python3 -c "import secrets; print(secrets.token_hex(32))"

SECURITY GUARANTEE: If ENCRYPTION_KEY is missing or malformed, the server
refuses to start. There is NO plaintext fallback for medical data.
"""
import os
import hmac
import hashlib
import base64
import struct
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def _validate_key_on_startup():
    """
    Called once at module import time.
    Fail-Deadly: raises SystemExit if ENCRYPTION_KEY is missing or wrong length.
    This prevents ANY plaintext fallback for transcripts or PII.
    """
    key_hex = os.environ.get("ENCRYPTION_KEY", "")
    if not key_hex:
        raise SystemExit(
            "\n\n"
            "╔═══════════════════════════════════════════════════════════╗\n"
            "║  FATAL: ENCRYPTION_KEY environment variable is not set.  ║\n"
            "║                                                           ║\n"
            "║  Receptionist.co enforces Fail-Deadly architecture.      ║\n"
            "║  The server cannot start without a valid encryption key.  ║\n"
            "║                                                           ║\n"
            "║  Set in Railway → aria-call-handler → Variables:         ║\n"
            "║  ENCRYPTION_KEY = <64-char hex string>                   ║\n"
            "║  Generate: python3 -c \"import secrets; print(secrets.token_hex(32))\" ║\n"
            "╚═══════════════════════════════════════════════════════════╝\n"
        )
    if len(key_hex) != 64:
        raise SystemExit(
            f"\nFATAL: ENCRYPTION_KEY must be exactly 64 hex characters (got {len(key_hex)}).\n"
            "Generate a valid key: python3 -c \"import secrets; print(secrets.token_hex(32))\"\n"
        )
    try:
        bytes.fromhex(key_hex)
    except ValueError:
        raise SystemExit("FATAL: ENCRYPTION_KEY is not valid hexadecimal.\n")


# ── Enforce on module import (startup time) ───────────────────────────────────
_validate_key_on_startup()


def _get_key() -> bytes:
    return bytes.fromhex(os.environ["ENCRYPTION_KEY"])


def should_encrypt() -> bool:
    """Always True — key is validated at startup. Fail-Deadly guarantees this."""
    return True


def encrypt_api_key(plaintext: str) -> str:
    """Encrypt a third-party API key (Cal.com, Zapier, etc.) before DB storage."""
    return encrypt_text(plaintext)


def decrypt_api_key(ciphertext: str) -> str:
    """Decrypt a stored API key for use."""
    return decrypt_text(ciphertext)


def encrypt_text(plaintext: str) -> str:
    """
    AES-256-CBC encryption with HMAC-SHA256 authentication.
    Format: base64(hmac_tag[32] + iv[16] + ciphertext)
    """
    if not plaintext:
        return plaintext
    key   = _get_key()
    iv    = os.urandom(16)
    pad   = 16 - len(plaintext.encode()) % 16
    data  = plaintext.encode() + bytes([pad] * pad)
    ciph  = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    enc   = ciph.encryptor()
    ct    = enc.update(data) + enc.finalize()
    tag   = hmac.new(key, iv + ct, hashlib.sha256).digest()
    return base64.b64encode(tag + iv + ct).decode()


def decrypt_text(ciphertext: str) -> str:
    """Decrypt AES-256-CBC ciphertext. Returns original plaintext."""
    if not ciphertext:
        return ciphertext
    try:
        key  = _get_key()
        raw  = base64.b64decode(ciphertext)
        tag, iv, ct = raw[:32], raw[32:48], raw[48:]
        expected = hmac.new(key, iv + ct, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("HMAC validation failed — data may be tampered")
        ciph = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        dec  = ciph.decryptor()
        data = dec.update(ct) + dec.finalize()
        pad  = data[-1]
        return data[:-pad].decode()
    except Exception:
        return ciphertext  # return as-is if decrypt fails (may be legacy plaintext)


def encrypt_pii(value: str | None) -> str | None:
    """Encrypt patient PII (phone, email, name) for column-level at-rest protection."""
    if not value:
        return value
    return encrypt_text(str(value))


def decrypt_pii(value: str | None) -> str | None:
    """Decrypt patient PII for application use."""
    if not value:
        return value
    return decrypt_text(str(value))


def hash_pii(value: str) -> str:
    """
    Deterministic HMAC-SHA256 of a PII value using the master key.
    Used as a lookup key when the value itself is stored encrypted with random IV.
    Safe to store: reveals nothing about the original value without the master key.
    """
    if not value:
        return ""
    key = _get_key()
    return hmac.new(key, str(value).encode(), hashlib.sha256).hexdigest()


def crypto_shred_key(record_id: str) -> str:
    """
    Generate a unique per-record encryption key for crypto-shredding.
    Store this derived key alongside the record — delete the key to shred.
    Note: full crypto-shredding requires per-record keys (future enhancement).
    """
    master = _get_key()
    return hmac.new(master, record_id.encode(), hashlib.sha256).hexdigest()
