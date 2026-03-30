"""
AES-256-CBC encryption for sensitive third-party API keys.
Used to encrypt Cal.com API keys before storing in Supabase.

Required Railway env var:
  ENCRYPTION_KEY = 64-char hex string (32 bytes)
  Generate with: python3 -c "import secrets; print(secrets.token_hex(32))"

NEVER store raw API keys in the database. Encrypt on write, decrypt on read.
"""
import os
import hmac
import hashlib
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


def _get_key() -> bytes:
    key_hex = os.environ.get("ENCRYPTION_KEY", "")
    if not key_hex or len(key_hex) != 64:
        raise ValueError("ENCRYPTION_KEY must be a 64-char hex string (32 bytes)")
    return bytes.fromhex(key_hex)


def encrypt_api_key(plaintext: str) -> str | None:
    """Encrypt a raw API key. Returns 'iv_hex:ciphertext_hex' string."""
    if not plaintext:
        return None
    key = _get_key()
    import secrets
    iv = secrets.token_bytes(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    # Pad to 16-byte boundary (PKCS7)
    pad_len = 16 - (len(plaintext) % 16)
    padded = plaintext.encode() + bytes([pad_len] * pad_len)
    ciphertext = encryptor.update(padded) + encryptor.finalize()
    return f"{iv.hex()}:{ciphertext.hex()}"


def decrypt_api_key(encrypted: str) -> str | None:
    """Decrypt an encrypted API key from Supabase."""
    if not encrypted:
        return None
    try:
        key = _get_key()
        iv_hex, ct_hex = encrypted.split(":", 1)
        iv = bytes.fromhex(iv_hex)
        ciphertext = bytes.fromhex(ct_hex)
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()
        # Remove PKCS7 padding
        pad_len = padded[-1]
        return padded[:-pad_len].decode()
    except Exception:
        return None


# Aliases for encrypting conversational data (transcripts, summaries, notes)
def encrypt_text(text: str) -> str | None:
    """Encrypt sensitive text (transcripts, call summaries, AI notes) for HIPAA compliance."""
    return encrypt_api_key(text)

def decrypt_text(encrypted: str) -> str | None:
    """Decrypt a field encrypted with encrypt_text."""
    return decrypt_api_key(encrypted)

def should_encrypt() -> bool:
    """Only encrypt if ENCRYPTION_KEY is configured (graceful degradation)."""
    return bool(os.environ.get("ENCRYPTION_KEY", ""))
