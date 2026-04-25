"""
contact_pii.py — Phase 2a per-field encryption for contact PII.

Receptionist.co Data Security Policy — Phase 2a (READ-only deployment).

This module ADDS per-field encryption capability using HKDF-derived keys.
It does NOT replace encryption.py — transcripts continue to use the
single-key scheme via encryption.py's encrypt_text/decrypt_text.

────────────────────────────────────────────────────────────────────────────
DESIGN
────────────────────────────────────────────────────────────────────────────
Per-field encryption derives a unique key for each (business, contact, field)
tuple via HKDF-SHA256. If one derived key is compromised (e.g. via memory
dump on a single request), it decrypts only that one field for that one
contact — not the entire database.

KEY DERIVATION:
  derived_key = HKDF-SHA256(
    ikm    = ENCRYPTION_KEY    (32 bytes from env)
    salt   = ENCRYPTION_PEPPER (32 bytes from env)
    info   = utf8(business_id + "|" + contact_id + "|" + field_name)
    length = 32
  )

WIRE FORMAT:
  base64( 0x01 || tag[32] || iv[16] || ciphertext )

  version_byte = 0x01  (distinguishes from legacy single-key format)
  tag          = HMAC-SHA256(derived_key, version_byte || iv || ciphertext)
  iv           = 16 random bytes (fresh per encryption)
  ciphertext   = AES-256-CBC(derived_key, iv, PKCS#7-padded plaintext)

  Note: HMAC scope INCLUDES the version byte. This cryptographically binds
  the format version to the integrity tag — preventing version-strip attacks
  where an attacker reframes a v1 ciphertext as legacy and vice versa.

────────────────────────────────────────────────────────────────────────────
PHASE 2a SCOPE
────────────────────────────────────────────────────────────────────────────
This module is deployed in a READ-CAPABLE state:
  - decrypt_field() can read v1 (HKDF-derived) AND legacy (master-key) AND
    plaintext fallback. Production reads continue to use encryption.py for
    now; nothing currently calls decrypt_field() from production code.
  - encrypt_field() exists and is correct, but no production code calls it
    yet. Phase 2b will wire write paths in CRM and Railway.

────────────────────────────────────────────────────────────────────────────
REQUIRED ENV VARS
────────────────────────────────────────────────────────────────────────────
  ENCRYPTION_KEY    = 64-char hex (32 bytes)  — already validated by encryption.py
  ENCRYPTION_PEPPER = 64-char hex (32 bytes)  — validated here at import time

If ENCRYPTION_PEPPER is missing or malformed, this module fails at import
(SystemExit). Same fail-deadly pattern as encryption.py.
"""

import os
import hmac
import hashlib
import base64
import secrets
from typing import Optional
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend

# Import the master-key primitives for legacy fallback
from encryption import decrypt_text as _decrypt_text_legacy

# Wire format constants
VERSION_V1 = 0x01
TAG_LEN    = 32
IV_LEN     = 16
MIN_V1_RAW_LEN = 1 + TAG_LEN + IV_LEN + 16  # version + tag + iv + at least one block of ct


# ────────────────────────────────────────────────────────────────────────────
# Pepper validation (fail-deadly at module import)
# ────────────────────────────────────────────────────────────────────────────

def _validate_pepper_on_startup() -> None:
    pepper_hex = os.environ.get("ENCRYPTION_PEPPER", "")
    if not pepper_hex:
        raise SystemExit(
            "\n\n"
            "FATAL: ENCRYPTION_PEPPER env var is not set.\n"
            "Phase 2a per-field encryption requires both ENCRYPTION_KEY and\n"
            "ENCRYPTION_PEPPER. Generate: python3 -c \"import secrets; print(secrets.token_hex(32))\"\n"
            "Set in Railway → Variables. Must be identical across CRM and Railway.\n"
        )
    if len(pepper_hex) != 64:
        raise SystemExit(
            f"\nFATAL: ENCRYPTION_PEPPER must be exactly 64 hex characters (got {len(pepper_hex)}).\n"
        )
    try:
        bytes.fromhex(pepper_hex)
    except ValueError:
        raise SystemExit("FATAL: ENCRYPTION_PEPPER is not valid hexadecimal.\n")


_validate_pepper_on_startup()


def _get_master_key() -> bytes:
    return bytes.fromhex(os.environ["ENCRYPTION_KEY"])


def _get_pepper() -> bytes:
    return bytes.fromhex(os.environ["ENCRYPTION_PEPPER"])


# ────────────────────────────────────────────────────────────────────────────
# Key derivation
# ────────────────────────────────────────────────────────────────────────────

def derive_field_key(business_id: str, contact_id: str, field_name: str) -> bytes:
    """
    HKDF-SHA256 derivation of a 32-byte key bound to (business, contact, field).

    Inputs are concatenated as `business_id|contact_id|field_name` and
    UTF-8 encoded. UUIDs MUST be lowercase with dashes (Postgres default).
    Field names MUST be the bare column name (e.g. "phone_normalized",
    not "contacts.phone_normalized").

    Pure function — same inputs always yield same output. The CRM
    TypeScript implementation produces byte-identical output.
    """
    info = (business_id + "|" + contact_id + "|" + field_name).encode("utf-8")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_get_pepper(),
        info=info,
        backend=default_backend(),
    ).derive(_get_master_key())


# ────────────────────────────────────────────────────────────────────────────
# Encryption — v1 wire format
# ────────────────────────────────────────────────────────────────────────────

def encrypt_field(
    business_id: str,
    contact_id: str,
    field_name: str,
    plaintext: Optional[str],
) -> Optional[str]:
    """
    Encrypt plaintext for a specific (business, contact, field) tuple.

    Returns base64 string in the v1 wire format:
      base64( 0x01 || HMAC[32] || IV[16] || AES_CBC(plaintext) )

    Null/empty handling matches encryption.py's encrypt_text:
      - None    → None
      - ""      → ""
      - else    → ciphertext string

    NOT CALLED FROM PRODUCTION CODE IN PHASE 2a. Phase 2b will wire this
    into Railway's call_handler.py contact write paths.
    """
    if plaintext is None:
        return None
    if plaintext == "":
        return ""

    key = derive_field_key(business_id, contact_id, field_name)
    iv = secrets.token_bytes(IV_LEN)

    # PKCS#7 padding (matches encryption.py exactly)
    pad_len = 16 - len(plaintext.encode("utf-8")) % 16
    padded = plaintext.encode("utf-8") + bytes([pad_len] * pad_len)

    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()).encryptor()
    ct = encryptor.update(padded) + encryptor.finalize()

    # HMAC scope INCLUDES version byte — binds format version to tag
    version_byte = bytes([VERSION_V1])
    tag = hmac.new(key, version_byte + iv + ct, hashlib.sha256).digest()

    return base64.b64encode(version_byte + tag + iv + ct).decode("ascii")


# ────────────────────────────────────────────────────────────────────────────
# Decryption — three-way dispatch (v1 / legacy / plaintext fallback)
# ────────────────────────────────────────────────────────────────────────────

def decrypt_field(
    business_id: str,
    contact_id: str,
    field_name: str,
    value: Optional[str],
) -> Optional[str]:
    """
    Decrypt a field value, dispatching across format versions.

    Dispatch order:
      1. If value parses as v1 (starts with 0x01, length sane, HMAC verifies
         under derived key): return decrypted plaintext.
      2. If value parses as legacy single-key format (encryption.py
         encrypt_text output): return decrypted plaintext.
      3. Otherwise: return value unchanged (legacy plaintext rows).

    The fallback chain handles the migration window: existing rows may be
    plaintext (never encrypted), legacy-encrypted (master key), or v1
    (per-field key). Phase 2d will remove the plaintext fallback after
    backfill is complete.

    Note on collisions: a legacy ciphertext could coincidentally start
    with byte 0x01 (~0.4% probability). When that happens, the v1 HMAC
    verification fails (wrong key), and we fall through to legacy decrypt
    which succeeds. No data loss.
    """
    if value is None:
        return None
    if value == "":
        return ""

    # Try v1 format first
    try:
        raw = base64.b64decode(value)
        if len(raw) >= MIN_V1_RAW_LEN and raw[0] == VERSION_V1:
            version_byte = raw[0:1]
            tag = raw[1:1 + TAG_LEN]
            iv  = raw[1 + TAG_LEN:1 + TAG_LEN + IV_LEN]
            ct  = raw[1 + TAG_LEN + IV_LEN:]
            # Ciphertext must be a multiple of AES block size
            if len(ct) > 0 and len(ct) % 16 == 0:
                key = derive_field_key(business_id, contact_id, field_name)
                expected = hmac.new(key, version_byte + iv + ct, hashlib.sha256).digest()
                if hmac.compare_digest(tag, expected):
                    decryptor = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend()).decryptor()
                    padded = decryptor.update(ct) + decryptor.finalize()
                    pad_len = padded[-1]
                    if 1 <= pad_len <= 16 and padded[-pad_len:] == bytes([pad_len] * pad_len):
                        return padded[:-pad_len].decode("utf-8")
    except Exception:
        pass

    # Fall through to legacy single-key decrypt (encryption.py)
    # That function returns the input unchanged on failure, which gives us
    # the plaintext-fallback behavior automatically.
    return _decrypt_text_legacy(value)
