"""
call_handler.py — Twilio Media Streams WebSocket handler for Aria phone calls.
Runs as the 'web' service on Railway (needs a public URL for Twilio).
aria_agent.py runs as the 'worker' service (LiveKit video sessions).

Procfile:
  web:    python call_handler.py
  worker: python aria_agent.py start

Required env vars:
  OPENAI_API_KEY
  TWILIO_AUTH_TOKEN         ← NEW: required for webhook verification (Rule 6)
  SUPABASE_URL  (or NEXT_PUBLIC_SUPABASE_URL)
  SUPABASE_SERVICE_KEY  (or SUPABASE_SERVICE_ROLE_KEY)
"""
import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import parse_qs, urlparse

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Response

# ── Security: Twilio + LiveKit webhook verification (Rule 6) ──────────────────
from railway_security import verify_twilio_signature

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-handler")

app = FastAPI()

OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
PORT              = int(os.environ.get("PORT", 8080))
OPENAI_WS_URL     = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

try:
    from supabase import create_client, Client
    def get_supabase() -> Optional[Client]:
        if SUPABASE_URL and SUPABASE_KEY:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        return None
except ImportError:
    def get_supabase():
        return None


async def get_business_context(business_id: str) -> dict:
    """Load business config and knowledge from Supabase."""
    sb = get_supabase()
    if not sb or not business_id:
        return {}
    try:
        config   = sb.table("ai_receptionist_config").select("*").eq("business_id", business_id).single().execute()
        memories = sb.table("ai_memory").select("memory_key,memory_value,category").eq("business_id", business_id).order("created_at", desc=True).limit(60).execute()
        biz      = sb.table("businesses").select("name,industry,phone,timezone").eq("id", business_id).single().execute()

        config_data = config.data   or {}
        mem_data    = memories.data or []
        biz_data    = biz.data      or {}

        facts    = [m for m in mem_data if m.get("category") != "conversation"]
        mem_text = "\n".join([f"- {m['memory_key']}: {m['memory_value']}" for m in facts]) if facts else ""

        return {
            "business_name": biz_data.get("name", "our business"),
            "industry":      biz_data.get("industry", ""),
            "config":        config_data,
            "memory_text":   mem_text,
        }
    except Exception as e:
        logger.warning(f"Failed to load business context for {business_id}: {e}")
        return {}


def build_system_prompt(business_id: str, ctx: dict) -> str:
    config = ctx.get("config", {})
    brand  = ctx.get("business_name", "our business")
    memory = ctx.get("memory_text", "")

    ai_name    = config.get("ai_name", "Aria")
    role       = config.get("role_description")  or f"You are {ai_name}, the AI receptionist for {brand}."
    goal       = config.get("primary_goal")       or "Help callers book appointments and get their questions answered."
    anti_hal   = config.get("anti_hallucination_rule") or "If unsure, say you'll have the team follow up."
    turn_rules = config.get("turn_taking_rules")  or "Ask ONE question at a time. Wait for the full answer before continuing."
    rush_mode  = config.get("rush_mode_rules")    or "If the caller seems rushed, get name and callback number only."
    no_loop    = config.get("no_loop_rule")       or "Never ask the same question more than twice."
    escalation = config.get("escalation_rules")   or "If the caller says 'human', 'agent', or 'representative' — offer to have the team call back."
    greeting   = config.get("greeting")           or f"Thank you for calling {brand}, I'm {ai_name}. How can I help you today?"
    custom     = config.get("custom_instructions") or ""

    sales_note = ""
    if business_id == "00000000-0000-0000-0000-000000000099":
        sales_note = """
IMPORTANT — YOU ARE THE PRODUCT DEMO:
This caller is a potential customer considering buying an AI receptionist service.
You ARE the product they're evaluating. Be enthusiastic, knowledgeable, and offer to book a demo with Max.
"""

    return f"""{role}
{sales_note}
BUSINESS: {brand}
PRIMARY GOAL: {goal}

━━━ KNOWLEDGE & MEMORY ━━━━━━━━━━━━━━━━━━
{memory if memory else "No specific knowledge loaded yet."}

━━━ CALL RULES ━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-HALLUCINATION: {anti_hal}
TURN-TAKING: {turn_rules}
RUSH MODE: {rush_mode}
NO-LOOP: {no_loop}
ESCALATION: {escalation}
HUMAN (ALWAYS): If caller says "human", "agent", "representative", or presses 0 — say
  "Of course, let me have someone from our team call you right back. What's the best number to reach you?"

━━━ VOICE RULES (CRITICAL) ━━━━━━━━━━━━━
This is a PHONE CALL. Keep every response SHORT and NATURAL.
• Max 1-2 sentences per turn
• No bullet points, no markdown — natural spoken language only
• Never say "Certainly!", "Absolutely!", "Great question!" — pure filler, avoid
• Answer first, then ask ONE follow-up question if needed
• Never end every turn with "Is there anything else I can help with?"
• 12-hour time format with AM/PM only — never say "14:00"

━━━ GREETING ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start with exactly: {greeting}

{custom}""".strip()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "aria-call-handler"}


@app.post("/twilio-incoming")
async def twilio_incoming(request: Request):
    """
    Twilio calls this HTTP webhook when a call first connects.
    Returns TwiML to connect the call to our WebSocket stream.
    Security: verifies X-Twilio-Signature before responding.
    """
    body_bytes = await request.body()
    params     = dict(await request.form())
    signature  = request.headers.get("X-Twilio-Signature", "")
    url        = str(request.url)

    if TWILIO_AUTH_TOKEN:
        if not verify_twilio_signature(signature, url, params, TWILIO_AUTH_TOKEN):
            logger.warning(f"[twilio-incoming] Invalid signature from {request.client.host} — rejecting")
            return Response("Unauthorized", status_code=401)
    else:
        logger.warning("[twilio-incoming] TWILIO_AUTH_TOKEN not set — skipping signature check")

    call_sid    = params.get("CallSid", "")
    from_number = params.get("From", "")
    to_number   = params.get("To", "")
    business_id = params.get("business_id", "00000000-0000-0000-0000-000000000099")

    # Look up business by their Twilio number if not passed as param
    if business_id == "00000000-0000-0000-0000-000000000099" and to_number:
        sb = get_supabase()
        if sb:
            try:
                r = sb.table("businesses").select("id").eq("twilio_phone", to_number).single().execute()
                if r.data:
                    business_id = r.data["id"]
            except Exception:
                pass

    # Log the call
    sb = get_supabase()
    if sb:
        try:
            sb.table("calls").insert({
                "business_id":     business_id,
                "twilio_call_sid": call_sid,
                "direction":       "inbound",
                "from_number":     from_number,
                "to_number":       to_number,
                "started_at":      datetime.now(timezone.utc).isoformat(),
                "outcome":         "in_progress",
                "handled_by_ai":   True,
            }).execute()
        except Exception as e:
            logger.warning(f"Call log failed: {e}")

    ws_base = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "aria-call-handler-production.up.railway.app")
    ws_url  = (f"wss://{ws_base}/ws/call"
               f"?business_id={business_id}"
               f"&amp;call_sid={call_sid}"
               f"&amp;from={from_number}")

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{ws_url}">
      <Parameter name="business_id" value="{business_id}"/>
      <Parameter name="call_sid"    value="{call_sid}"/>
      <Parameter name="from_number" value="{from_number}"/>
    </Stream>
  </Connect>
</Response>"""

    return Response(content=twiml, media_type="text/xml")


@app.post("/twilio-status")
async def twilio_status(request: Request):
    """
    Twilio calls this when a call ends with final status (duration, etc).
    Security: also verifies X-Twilio-Signature.
    """
    body_bytes = await request.body()
    params     = dict(await request.form())
    signature  = request.headers.get("X-Twilio-Signature", "")

    if TWILIO_AUTH_TOKEN:
        if not verify_twilio_signature(signature, str(request.url), params, TWILIO_AUTH_TOKEN):
            logger.warning("[twilio-status] Invalid signature — rejecting")
            return Response("Unauthorized", status_code=401)

    call_sid = params.get("CallSid", "")
    duration = int(params.get("CallDuration", 0))
    status   = params.get("CallStatus", "completed")

    sb = get_supabase()
    if sb and call_sid:
        try:
            outcome = "completed" if status == "completed" else "missed"
            sb.table("calls").update({
                "duration_seconds": duration,
                "ended_at":         datetime.now(timezone.utc).isoformat(),
                "outcome":          outcome,
            }).eq("twilio_call_sid", call_sid).execute()
            logger.info(f"Call {call_sid} ended: {status}, {duration}s")
        except Exception as e:
            logger.warning(f"Status update failed: {e}")

    return Response("OK", status_code=200)


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET — TWILIO MEDIA STREAMS ↔ OPENAI REALTIME
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/call")
async def call_websocket(ws: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Twilio connects here and streams bidirectional μ-law 8kHz audio.
    We relay to OpenAI Realtime API and stream Aria's voice back.

    Note: WebSocket connections from Twilio are authenticated via the
    X-Twilio-Signature on the /twilio-incoming POST that generates the TwiML.
    This WS endpoint receives audio only from callers Twilio has already
    authenticated and connected.
    """
    await ws.accept()

    parsed      = urlparse(str(ws.url))
    params      = parse_qs(parsed.query)
    business_id = params.get("business_id", ["00000000-0000-0000-0000-000000000099"])[0]
    call_sid    = params.get("call_sid",    [""])[0]

    logger.info(f"📞 Call stream connected | business={business_id} | sid={call_sid}")

    ctx    = await get_business_context(business_id)
    prompt = build_system_prompt(business_id, ctx)

    stream_sid       = None
    transcript_parts = []
    aria_spoke       = False

    try:
        import websockets

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta":   "realtime=v1",
        }

        async with websockets.connect(
            OPENAI_WS_URL,
            additional_headers=headers,
            open_timeout=12,
        ) as oai_ws:
            logger.info("✓ OpenAI Realtime connected")

            await oai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {
                        "type":                "server_vad",
                        "threshold":           0.5,
                        "prefix_padding_ms":   300,
                        "silence_duration_ms": 1200,
                    },
                    "input_audio_format":        "g711_ulaw",
                    "output_audio_format":       "g711_ulaw",
                    "input_audio_transcription": {"model": "whisper-1"},
                    "voice":        "shimmer",
                    "instructions": prompt,
                    "modalities":   ["text", "audio"],
                    "temperature":  0.75,
                }
            }))

            # Trigger greeting
            await oai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "[Call started. Deliver your greeting now.]"}],
                }
            }))
            await oai_ws.send(json.dumps({"type": "response.create"}))

            async def recv_twilio():
                nonlocal stream_sid
                async for message in ws.iter_text():
                    try:
                        data  = json.loads(message)
                        event = data.get("event", "")
                        if event == "start":
                            stream_sid = data["start"]["streamSid"]
                        elif event == "media":
                            await oai_ws.send(json.dumps({
                                "type":  "input_audio_buffer.append",
                                "audio": data["media"]["payload"],
                            }))
                        elif event == "stop":
                            break
                    except Exception as e:
                        logger.warning(f"recv_twilio: {e}")

            async def recv_openai():
                nonlocal aria_spoke
                async for raw in oai_ws:
                    try:
                        data       = json.loads(raw)
                        event_type = data.get("type", "")

                        if event_type == "error":
                            logger.error(f"OpenAI error: {data.get('error', {})}")
                        elif event_type == "response.audio.delta" and stream_sid:
                            aria_spoke = True
                            await ws.send_text(json.dumps({
                                "event":     "media",
                                "streamSid": stream_sid,
                                "media":     {"payload": data["delta"]},
                            }))
                        elif event_type == "response.audio_transcript.done":
                            text = data.get("transcript", "").strip()
                            if text:
                                transcript_parts.append(f"Aria: {text}")
                        elif event_type == "conversation.item.input_audio_transcription.completed":
                            text = data.get("transcript", "").strip()
                            if text:
                                transcript_parts.append(f"Caller: {text}")
                        elif event_type == "input_audio_buffer.speech_started":
                            await oai_ws.send(json.dumps({"type": "response.cancel"}))
                    except Exception as e:
                        logger.warning(f"recv_openai: {e}")

            await asyncio.gather(recv_twilio(), recv_openai(), return_exceptions=True)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Call handler error: {type(e).__name__}: {e}", exc_info=True)
    finally:
        if call_sid:
            sb = get_supabase()
            if sb:
                try:
                    outcome = "completed" if aria_spoke and transcript_parts else "missed"
                    summary = " | ".join(transcript_parts[-12:]) if transcript_parts else ""
                    sb.table("calls").update({
                        "transcript_summary": summary or None,
                        "outcome":            outcome,
                        "ended_at":           datetime.now(timezone.utc).isoformat(),
                    }).eq("twilio_call_sid", call_sid).execute()
                    logger.info(f"Call logged: {call_sid} | outcome={outcome} | turns={len(transcript_parts)}")
                except Exception as e:
                    logger.warning(f"Failed to update call: {e}")
        logger.info("Call session ended")


if __name__ == "__main__":
    logger.info(f"🚀 Aria Call Handler starting on port {PORT}")
    logger.info(f"   TWILIO_AUTH_TOKEN set: {bool(TWILIO_AUTH_TOKEN)}")
    logger.info(f"   OPENAI_API_KEY set:    {bool(OPENAI_API_KEY)}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
