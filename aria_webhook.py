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


# ── Glass Box: Barge-In Endpoint ─────────────────────────────────────────────
from livekit import api as livekit_api

LIVEKIT_URL        = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY    = os.environ.get("LIVEKIT_API_KEY", "")

@app.post("/barge-in")
async def barge_in(request: Request):
    """
    Owner clicks "Take Over" in the Glass Box dashboard.
    1. Marks Aria as muted in active_calls
    2. Returns a LiveKit token for the owner's browser mic to join the room
    
    The owner's browser then joins the LiveKit room directly — the SIP caller
    (already a SIPParticipant in the room) hears the owner instead of Aria.
    """
    body = await request.json()
    room_name      = body.get("room_name")
    active_call_id = body.get("active_call_id")
    business_id    = body.get("business_id")

    if not room_name or not business_id:
        return {"error": "room_name and business_id required"}, 400

    # ── Mark Aria as muted in active_calls ───────────────────────────────────
    try:
        from supabase import create_client
        sb_url = os.environ.get("SUPABASE_URL", "")
        sb_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if sb_url and sb_key and active_call_id:
            sb = create_client(sb_url, sb_key)
            sb.from_("active_calls").update({
                "ai_muted":     True,
                "owner_joined": True,
                "status":       "human-handled",
            }).eq("id", active_call_id).execute()
    except Exception as e:
        logger.warning(f"barge-in: supabase update failed: {e}")

    # ── Generate LiveKit token for owner's browser ────────────────────────────
    try:
        token = livekit_api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_URL) \
            .with_identity(f"owner-{business_id[:8]}") \
            .with_name("Business Owner") \
            .with_grants(livekit_api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,       # owner can speak
                can_subscribe=True,     # owner can hear caller
                can_publish_data=True,
            )) \
            .to_jwt()

        return {
            "ok":        True,
            "token":     token,
            "ws_url":    LIVEKIT_URL,
            "room_name": room_name,
        }
    except Exception as e:
        logger.error(f"barge-in: LiveKit token generation failed: {e}")
        return {"error": str(e)}, 500


@app.post("/restore-ai")
async def restore_ai(request: Request):
    """Owner hands control back to Aria."""
    body           = await request.json()
    active_call_id = body.get("active_call_id")

    if not active_call_id:
        return {"error": "active_call_id required"}, 400

    try:
        from supabase import create_client
        sb_url = os.environ.get("SUPABASE_URL", "")
        sb_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
        if sb_url and sb_key:
            sb = create_client(sb_url, sb_key)
            sb.from_("active_calls").update({
                "ai_muted":     False,
                "owner_joined": False,
                "status":       "in-progress",
            }).eq("id", active_call_id).execute()
    except Exception as e:
        logger.warning(f"restore-ai: supabase update failed: {e}")

    return {"ok": True, "message": "AI restored"}


# ── Twilio Subaccount Provisioner ─────────────────────────────────────────────
@app.post("/provision")
async def provision_business(request: Request):
    """
    Called when a new business signs up on Receptionist.co.
    
    1. Creates a dedicated Twilio subaccount for the business
    2. Purchases a local number in their area code
    3. Configures the number webhook back to this call handler
    4. Stores credentials in Supabase (encrypted)
    
    This isolates each business's SMS/call traffic — if one goes rogue,
    only their subaccount is suspended, not the entire platform.
    """
    body        = await request.json()
    business_id = body.get("business_id")
    business_name = body.get("business_name", "Receptionist Business")
    area_code   = body.get("area_code", "720")  # Default: Denver

    if not business_id:
        return {"error": "business_id required"}, 400

    TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
    HANDLER_URL  = os.environ.get("CALL_HANDLER_URL", 
                   "https://aria-call-handler-production.up.railway.app")

    if not TWILIO_SID or not TWILIO_TOKEN:
        return {"error": "Twilio credentials not configured"}, 500

    try:
        from twilio.rest import Client
        master_client = Client(TWILIO_SID, TWILIO_TOKEN)

        # 1. Create subaccount
        subaccount = master_client.api.accounts.create(
            friendly_name=f"Receptionist - {business_name}"
        )
        sub_sid   = subaccount.sid
        sub_token = subaccount.auth_token

        logger.info(f"Created Twilio subaccount {sub_sid} for business {business_id}")

        # 2. Purchase a local number in the business's area code
        sub_client = Client(sub_sid, sub_token)
        available  = sub_client.available_phone_numbers("US").local.list(
            area_code=area_code, sms_enabled=True, voice_enabled=True, limit=1
        )
        if not available:
            available = sub_client.available_phone_numbers("US").local.list(
                sms_enabled=True, voice_enabled=True, limit=1
            )

        if not available:
            return {"error": f"No numbers available in area code {area_code}"}, 404

        purchased = sub_client.incoming_phone_numbers.create(
            phone_number=available[0].phone_number,
            voice_url=f"{HANDLER_URL}/voice?business_id={business_id}",
            sms_url=f"{HANDLER_URL}/sms?business_id={business_id}",
            friendly_name=f"{business_name} - Aria"
        )

        phone_number = purchased.phone_number
        logger.info(f"Purchased {phone_number} for business {business_id}")

        # 3. Store in Supabase
        try:
            from supabase import create_client
            sb_url = os.environ.get("SUPABASE_URL", "")
            sb_key = os.environ.get("SUPABASE_SERVICE_KEY", "")
            if sb_url and sb_key:
                sb = create_client(sb_url, sb_key)
                # Store subaccount creds (encrypt in production via Supabase Vault)
                sb.from_("businesses").update({
                    "twilio_subaccount_sid":   sub_sid,
                    "twilio_subaccount_token": sub_token,  # TODO: encrypt via Vault
                    "provisioned_phone":       phone_number,
                }).eq("id", business_id).execute()

                sb.from_("settings_business").upsert({
                    "business_id":      business_id,
                    "provisioned_phone": phone_number,
                }, on_conflict="business_id").execute()

                logger.info(f"Saved provisioning data for business {business_id}")
        except Exception as db_err:
            logger.warning(f"DB save failed (number still provisioned): {db_err}")

        return {
            "ok":             True,
            "phone_number":   phone_number,
            "subaccount_sid": sub_sid,
            "area_code":      area_code,
        }

    except Exception as e:
        logger.error(f"Provision failed for business {business_id}: {e}")
        return {"error": str(e)}, 500
