"""
aria_webhook.py — Minimal HTTP server for the aria_agent Railway service.
Runs alongside aria_agent.py worker (see Procfile.aria_agent).

Provides:
  GET  /health          — Railway health check
  POST /livekit-webhook — LiveKit room events with signature verification (Rule 6)

LiveKit fires webhook events when participants join/leave rooms.
We verify the signature then let the agent SDK handle the room connection.
"""
import os
import json
import logging
import asyncio
from datetime import datetime

from fastapi import FastAPI, Request, Response

# ── Security: Rule 6 ─────────────────────────────────────────────────────────
from railway_security import verify_livekit_webhook

logger = logging.getLogger("aria-webhook")
app    = FastAPI()

LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")


@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "service": "aria-agent",
        "livekit_secret_set": bool(LIVEKIT_API_SECRET),
    }


@app.post("/livekit-webhook")
async def livekit_webhook(request: Request):
    """
    LiveKit Cloud fires this when a participant joins or leaves a room.
    The agent SDK dispatches aria_agent.py to the room automatically via
    the 'aria-agent' agent_name registration.

    This endpoint exists for:
    - Logging room events to Supabase
    - Future: custom dispatch logic
    - Security audit trail

    LiveKit signs requests with HMAC-SHA256 using your LIVEKIT_API_SECRET.
    """
    body_bytes  = await request.body()
    auth_header = request.headers.get("Authorization", "")

    # ── SECURITY: Verify this is genuinely from LiveKit Cloud ────────────────
    if LIVEKIT_API_SECRET:
        if not verify_livekit_webhook(auth_header, body_bytes, LIVEKIT_API_SECRET):
            logger.warning(
                f"[livekit-webhook] Invalid signature from {request.client.host} — rejecting"
            )
            return Response("Unauthorized", status_code=401)
    else:
        logger.warning("[livekit-webhook] LIVEKIT_API_SECRET not set — skipping verification")

    # ── Parse event ───────────────────────────────────────────────────────────
    try:
        event = json.loads(body_bytes)
    except Exception:
        return Response("Bad Request", status_code=400)

    event_type  = event.get("event", "")
    room        = event.get("room", {})
    participant = event.get("participant", {})

    room_name   = room.get("name", "")
    identity    = participant.get("identity", "")

    logger.info(f"LiveKit event: {event_type} | room={room_name} | participant={identity}")

    # ── Log to Supabase (fire and forget) ─────────────────────────────────────
    if event_type in ("participant_joined", "participant_left", "room_finished"):
        try:
            from supabase import create_client
            sb = create_client(
                os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", ""),
                os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
            )

            # Extract business_id from room name (format: aria-{uuid}-{timestamp})
            parts = room_name.split("-")
            import re
            business_id = None
            if len(parts) >= 6:
                candidate = "-".join(parts[1:6])
                if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', candidate, re.I):
                    business_id = candidate

            if business_id and event_type == "room_finished":
                # Session ended — could trigger summary generation here
                logger.info(f"Room finished for business {business_id}")

        except Exception as e:
            logger.warning(f"Supabase log failed: {e}")

    return Response("OK", status_code=200)
