"""
call_handler.py — Twilio Media Streams WebSocket handler for Aria phone calls.

Runs as a separate FastAPI server on Railway alongside aria_agent.py.
Receives Twilio audio via WebSocket, sends to OpenAI Realtime API,
streams Aria's voice response back to the caller.

Logs all calls to Supabase.

Configure Railway:
  - Start command: python call_handler.py
  - PORT env var: auto-set by Railway
  - OPENAI_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY required

Twilio webhook:
  POST https://your-railway-url.up.railway.app/api/calls/inbound
  (or use app.receptionist.co/api/calls/inbound which returns TwiML pointing here)
"""

import os
import json
import asyncio
import base64
import logging
import httpx
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import PlainTextResponse
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-handler")

app = FastAPI()

OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL     = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY     = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
PORT             = int(os.environ.get("PORT", 8080))

OPENAI_WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"


def get_supabase() -> Optional[Client]:
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None


async def get_business_context(business_id: str) -> dict:
    """Load business config and knowledge from Supabase."""
    sb = get_supabase()
    if not sb:
        return {}
    try:
        biz = sb.table("settings_business").select("*").eq("business_id", business_id).single().execute()
        config = sb.table("ai_receptionist_config").select("*").eq("business_id", business_id).single().execute()
        memories = sb.table("ai_memory").select("memory_key,memory_value,category").eq("business_id", business_id).limit(50).execute()

        biz_data    = biz.data    or {}
        config_data = config.data or {}
        mem_data    = memories.data or []

        mem_text = "\n".join([f"- {m['memory_key']}: {m['memory_value']}" for m in mem_data]) if mem_data else ""

        return {
            "brand_name":   biz_data.get("brand_name", "our business"),
            "config":       config_data,
            "memory_text":  mem_text,
        }
    except Exception as e:
        logger.warning(f"Failed to load business context: {e}")
        return {}


def build_system_prompt(business_id: str, ctx: dict) -> str:
    """Build Aria's system prompt for phone calls."""
    config = ctx.get("config", {})
    brand  = ctx.get("brand_name", "our business")
    memory = ctx.get("memory_text", "")

    ai_name    = config.get("ai_name", "Aria")
    owner      = config.get("owner_name", "")
    role       = config.get("role_description", f"You are {ai_name}, the AI receptionist for {brand}.")
    goal       = config.get("primary_goal", "Help callers with bookings and questions.")
    anti_hal   = config.get("anti_hallucination_rule", "If unsure, say you'll have the team follow up.")
    turn_rules = config.get("turn_taking_rules", "Ask ONE question at a time. Wait for the full answer.")
    rush_mode  = config.get("rush_mode_rules", "If caller is rushed, get name and phone only.")
    no_loop    = config.get("no_loop_rule", "Never repeat the same question twice.")
    escalation = config.get("escalation_rules", "Say 'human' or press 0 to reach a live person.")
    greeting   = config.get("greeting", f"Thanks for calling {brand} — I'm {ai_name}. How can I help you today?")
    custom     = config.get("custom_instructions", "")

    # Special handling for Receptionist.co sales line
    is_sales_line = business_id == "00000000-0000-0000-0000-000000000099"
    sales_note = """
IMPORTANT — YOU ARE THE PRODUCT DEMO:
You are Aria, answering a call for Receptionist.co. This caller is a potential customer
who wants to know about AI receptionist services. You ARE the AI receptionist they're 
considering buying. Be enthusiastic, knowledgeable, and offer to book a demo with Max.
""" if is_sales_line else ""

    prompt = f"""
{role}
{sales_note}

BUSINESS: {brand}
PRIMARY GOAL: {goal}

━━━ KNOWLEDGE BASE ━━━━━━━━━━━━━━━━━━━━━━
{memory if memory else "No specific knowledge loaded yet."}

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ANTI-HALLUCINATION: {anti_hal}
TURN-TAKING: {turn_rules}
RUSH MODE: {rush_mode}
NO-LOOP: {no_loop}
ESCALATION: {escalation}
HUMAN ESCALATION (ALWAYS): If caller says "human", "agent", "representative", or presses 0 — offer to have the team call back immediately.

━━━ VOICE RULES (CRITICAL) ━━━━━━━━━━━━━
This is a PHONE CALL. Keep responses SHORT.
- Max 1-2 sentences per response
- No bullet points, no markdown — natural speech only
- Never say "Certainly!", "Absolutely!", "Great question!" — filler
- Answer first, then ask one follow-up question if needed
- Do NOT say "Is there anything else I can help with?" at the end of every turn

━━━ GREETING ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Start the call with exactly: {greeting}

{custom}
""".strip()

    return prompt


@app.get("/health")
async def health():
    return {"status": "ok", "service": "aria-call-handler"}


@app.websocket("/ws/call")
async def call_websocket(ws: WebSocket):
    """
    WebSocket endpoint for Twilio Media Streams.
    Twilio connects here and streams bidirectional audio (μ-law 8kHz).
    We relay to OpenAI Realtime API and stream responses back.
    """
    await ws.accept()
    logger.info("Twilio Media Stream connected")

    # Parse query params
    query       = dict(item.split("=") for item in ws.url.query.split("&") if "=" in item)
    business_id = query.get("business_id", "00000000-0000-0000-0000-000000000099")
    call_sid    = query.get("call_sid", "")
    contact_id  = query.get("contact_id", "")

    # Load business context
    ctx     = await get_business_context(business_id)
    prompt  = build_system_prompt(business_id, ctx)
    stream_sid = None

    logger.info(f"Call started | business={business_id} | sid={call_sid}")

    # Connect to OpenAI Realtime
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "OpenAI-Beta": "realtime=v1",
    }

    import websockets

    try:
        async with websockets.connect(OPENAI_WS_URL, additional_headers=headers) as openai_ws:

            # Configure OpenAI session
            await openai_ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "turn_detection": {
                        "type":                   "server_vad",
                        "threshold":              0.5,
                        "prefix_padding_ms":      300,
                        "silence_duration_ms":    1500,
                    },
                    "input_audio_format":  "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "input_audio_transcription": { "model": "whisper-1" },
                    "voice": "alloy",
                    "instructions": prompt,
                    "modalities": ["text", "audio"],
                    "temperature": 0.4,
                }
            }))

            # Send initial greeting trigger
            await openai_ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "[Call started. Give your greeting now.]"}]
                }
            }))
            await openai_ws.send(json.dumps({"type": "response.create"}))

            transcript_parts = []

            async def receive_from_twilio():
                """Receive audio from Twilio, forward to OpenAI."""
                nonlocal stream_sid
                async for message in ws.iter_text():
                    data = json.loads(message)
                    event = data.get("event", "")

                    if event == "start":
                        stream_sid = data["start"]["streamSid"]
                        logger.info(f"Stream started: {stream_sid}")

                    elif event == "media":
                        # Forward audio to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "input_audio_buffer.append",
                            "audio": data["media"]["payload"],
                        }))

                    elif event == "stop":
                        logger.info("Twilio stream stopped")
                        break

            async def receive_from_openai():
                """Receive responses from OpenAI, forward audio to Twilio."""
                async for raw in openai_ws:
                    data = json.loads(raw)
                    event_type = data.get("type", "")

                    # Stream audio back to Twilio
                    if event_type == "response.audio.delta" and stream_sid:
                        await ws.send_text(json.dumps({
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {"payload": data["delta"]},
                        }))

                    # Collect transcript
                    elif event_type == "response.audio_transcript.done":
                        text = data.get("transcript", "")
                        if text:
                            transcript_parts.append(f"Aria: {text}")

                    elif event_type == "conversation.item.input_audio_transcription.completed":
                        text = data.get("transcript", "")
                        if text:
                            transcript_parts.append(f"Caller: {text}")

                    # Clear audio buffer after speech detected
                    elif event_type == "input_audio_buffer.speech_started":
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))

            # Run both streams concurrently
            await asyncio.gather(
                receive_from_twilio(),
                receive_from_openai(),
            )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Call handler error: {e}")
    finally:
        # Save transcript summary to Supabase
        if call_sid and transcript_parts:
            sb = get_supabase()
            if sb:
                try:
                    summary = " | ".join(transcript_parts[-10:])  # last 10 turns
                    sb.table("calls").update({
                        "transcript_summary": summary,
                        "outcome":            "completed",
                        "ended_at":           datetime.utcnow().isoformat(),
                    }).eq("twilio_call_sid", call_sid).execute()
                    logger.info(f"Call logged: {call_sid}")
                except Exception as e:
                    logger.warning(f"Failed to update call: {e}")

        logger.info("Call session ended")


if __name__ == "__main__":
    logger.info(f"Starting Aria Call Handler on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
