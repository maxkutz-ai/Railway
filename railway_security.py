# ─────────────────────────────────────────────────────────────────────────────
# railway_security.py  —  Drop this into your Railway services
# Implements Rule 6: Webhook Signature Verification for Twilio + LiveKit
# ─────────────────────────────────────────────────────────────────────────────

import hashlib
import hmac
import base64
import os
from urllib.parse import urlencode


# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: TWILIO VERIFICATION  (for aria-call-handler)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Usage in your FastAPI/aiohttp handler:
#
#   from railway_security import verify_twilio_signature
#
#   @app.post("/twilio-incoming")
#   async def twilio_incoming(request: Request):
#       body_bytes = await request.body()
#       signature  = request.headers.get("X-Twilio-Signature", "")
#       url        = str(request.url)          # full URL Twilio called
#       params     = dict(await request.form()) # POST body params
#
#       if not verify_twilio_signature(signature, url, params):
#           return Response("Unauthorized", status_code=401)
#
#       # Safe to process — this is genuinely from Twilio
#       ...

def verify_twilio_signature(
    signature: str,
    url: str,
    params: dict,
    auth_token: str | None = None
) -> bool:
    """
    Verify Twilio's X-Twilio-Signature header.
    Returns True if the request is genuine, False if it should be rejected.
    """
    token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN", "")
    if not token:
        print("WARNING: TWILIO_AUTH_TOKEN not set — skipping signature check")
        return True  # dev fallback — set token in prod

    # Twilio signs: URL + alphabetically sorted params concatenated
    sorted_params = sorted(params.items())
    signing_str = url + "".join(k + v for k, v in sorted_params)

    expected = base64.b64encode(
        hmac.new(
            token.encode("utf-8"),
            signing_str.encode("utf-8"),
            hashlib.sha1
        ).digest()
    ).decode("utf-8")

    # Timing-safe comparison to prevent timing attacks
    return hmac.compare_digest(signature, expected)


# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: LIVEKIT WEBHOOK VERIFICATION  (for aria_agent.py)
# ═══════════════════════════════════════════════════════════════════════════════
#
# LiveKit Cloud fires a webhook to your agent when a participant joins/leaves.
# It signs the request body with HMAC-SHA256 using your API Secret.
#
# Usage in your LiveKit webhook handler:
#
#   from railway_security import verify_livekit_webhook
#
#   @app.post("/livekit-webhook")
#   async def livekit_webhook(request: Request):
#       body_bytes = await request.body()
#       auth_header = request.headers.get("Authorization", "")
#
#       if not verify_livekit_webhook(auth_header, body_bytes):
#           return Response("Unauthorized", status_code=401)
#
#       event = json.loads(body_bytes)
#       # Safe to process
#       ...

def verify_livekit_webhook(
    auth_header: str,
    body_bytes: bytes,
    api_secret: str | None = None
) -> bool:
    """
    Verify LiveKit's Authorization header on webhook requests.
    LiveKit signs: HMAC-SHA256 of request body using your API Secret.
    Returns True if genuine, False if should be rejected.
    """
    secret = api_secret or os.environ.get("LIVEKIT_API_SECRET", "")
    if not secret:
        print("WARNING: LIVEKIT_API_SECRET not set — skipping webhook verification")
        return True  # dev fallback

    # LiveKit sends: Authorization: <base64(hmac-sha256(body))>
    try:
        expected = base64.b64encode(
            hmac.new(
                secret.encode("utf-8"),
                body_bytes,
                hashlib.sha256
            ).digest()
        ).decode("utf-8")

        return hmac.compare_digest(auth_header, expected)
    except Exception as e:
        print(f"LiveKit signature verification error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: COMPLETE FASTAPI INTEGRATION EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

FASTAPI_TWILIO_EXAMPLE = '''
# ── Add to aria-call-handler's main.py ───────────────────────────────────────

from fastapi import FastAPI, Request, Response
from railway_security import verify_twilio_signature

app = FastAPI()

@app.post("/twilio-incoming")
async def twilio_incoming(request: Request):
    """Twilio calls this when a call connects. Must verify signature first."""
    body_bytes = await request.body()
    params     = dict(await request.form())
    signature  = request.headers.get("X-Twilio-Signature", "")
    url        = str(request.url)

    if not verify_twilio_signature(signature, url, params):
        return Response("Unauthorized", status_code=401)

    # Genuinely from Twilio — safe to process
    call_sid    = params.get("CallSid", "")
    from_number = params.get("From", "")
    business_id = params.get("business_id", "")

    # ... your existing call handling logic

@app.post("/twilio-status")  
async def twilio_status(request: Request):
    """Twilio calls this when a call ends — also needs verification."""
    body_bytes = await request.body()
    params     = dict(await request.form())
    signature  = request.headers.get("X-Twilio-Signature", "")

    if not verify_twilio_signature(signature, str(request.url), params):
        return Response("Unauthorized", status_code=401)

    call_status   = params.get("CallStatus", "")
    call_duration = params.get("CallDuration", "0")
    # ... update Supabase with final call status
'''

FASTAPI_LIVEKIT_EXAMPLE = '''
# ── Add to aria_agent.py ─────────────────────────────────────────────────────

from fastapi import FastAPI, Request, Response
from railway_security import verify_livekit_webhook
import json

app = FastAPI()

@app.post("/livekit-webhook")
async def livekit_webhook(request: Request):
    """LiveKit fires this when a participant joins/leaves a room."""
    body_bytes  = await request.body()
    auth_header = request.headers.get("Authorization", "")

    if not verify_livekit_webhook(auth_header, body_bytes):
        return Response("Unauthorized", status_code=401)

    event = json.loads(body_bytes)
    event_type = event.get("event", "")

    if event_type == "participant_joined":
        room_name   = event["room"]["name"]
        participant = event["participant"]["identity"]

        # Only join if it's a real user (not the agent itself)
        if not participant.startswith("aria-agent"):
            await join_room_as_agent(room_name, event["room"].get("metadata", "{}"))

    elif event_type == "participant_left":
        # Clean up session if needed
        pass

    return Response("OK", status_code=200)
'''


if __name__ == "__main__":
    print("railway_security.py — Webhook verification utilities")
    print("Add TWILIO_AUTH_TOKEN and LIVEKIT_API_SECRET to your Railway env vars")
    print("\nTwilio verification: verify_twilio_signature(signature, url, params)")
    print("LiveKit verification: verify_livekit_webhook(auth_header, body_bytes)")
