"""
call_handler.py — Twilio Voice ↔ OpenAI Realtime API bridge
Railway service: aria-call-handler
Procfile start command: uvicorn call_handler:app --host 0.0.0.0 --port ${PORT:-8080}

Flow:
  1. Twilio calls arrive at POST /twilio-incoming
  2. We return TwiML that opens a Media Stream WebSocket back to us
  3. We bridge that stream to OpenAI Realtime API
  4. Aria speaks — Twilio plays the audio to the caller
  5. On hang-up, save transcript + contact to Supabase
"""

import os
import json
import logging
import asyncio
import base64
import re
from datetime import datetime, timezone

import httpx
import websockets
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-handler")

app = FastAPI()

# ── Environment ───────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_KEY", "")
PORT              = int(os.environ.get("PORT", 8080))
APP_URL           = os.environ.get("NEXT_PUBLIC_APP_URL", "https://app.receptionist.co")

OPENAI_VOICE      = "alloy"   # fallback — overridden per-business from DB

SYSTEM_PROMPT_BASE = """You are Aria, a professional AI receptionist working for {business_name}.
Be warm, helpful, and professional at all times.
{custom_instructions}
Current date and time: {datetime}
Business timezone: {timezone}

━━━ MANDATORY OPENING DISCLOSURE (say this at the START of every call) ━━━
Your opening greeting MUST be exactly:
"Hi! Thank you for calling {business_name}. I'm Aria, an AI assistant on a recorded line. How can I help you today?"
You MUST identify yourself as AI and mention the call is recorded on EVERY call.
This is required by federal and state law (TCPA, California BOT Act, two-party consent states).

━━━ TURN-TAKING / WAIT RULE (CRITICAL — PREVENTS INTERRUPTIONS) ━━━
- Ask ONLY ONE question at a time. NEVER combine two questions in one sentence.
- After asking a question, STOP speaking and wait silently for the caller's answer.
- Do NOT acknowledge ("Perfect / Thanks") until you have ACTUALLY HEARD the answer.
- If the caller starts answering, do NOT talk over them. Wait until they FULLY finish.
- If you ask a YES/NO question, you MUST wait for the answer before continuing.
- After the caller confirms something with "yes / mhmm / correct", pause ONE full beat before continuing.
- If you are mid-sentence and the caller speaks, STOP immediately and listen.

━━━ EMAIL / DATA CAPTURE RULE (CRITICAL) ━━━
- If the caller gives an email address, ALWAYS ask them to spell it out letter by letter.
- ALWAYS confirm back by spelling it out yourself letter by letter (including underscore/dash/dot).
- WAIT for the caller to say "yes that's correct" before moving on.
- Do NOT assume you heard the email correctly — always confirm. Wrong emails cause failed follow-ups.
- Example: "Could you spell that out for me?" → caller spells → "Let me confirm: j-o-h-n at g-m-a-i-l dot com — is that right?" → WAIT.

━━━ TIMEZONE RULE (CRITICAL — REQUIRED FOR ALL BOOKINGS) ━━━
- Before booking ANY appointment, you MUST ask: "What timezone are you in?"
- WAIT for the answer.
- Confirm the timezone back: "Got it — I'll schedule that in [timezone]."
- All appointment times discussed on this call are in the caller's stated timezone.
- The business timezone is {timezone}. If the caller is in a different timezone, clearly state the conversion.
- Example: "That's 3:00 PM Mountain Time, which is 2:00 PM Pacific — does that still work for you?"
- NEVER book without confirming timezone first. A booking in the wrong timezone causes a missed appointment.

━━━ NO-LOOP RULE ━━━
- If the caller says "you already asked that" or sounds annoyed:
  1) Apologize briefly: "You're right — sorry about that."
  2) Do NOT re-ask. Move to the next step immediately.

━━━ RUSH MODE ━━━
If the caller says anything like: "I gotta run," "no time," "quick," "call me back," "take care," "bye":
- Immediately switch to RUSH MODE.
- Ask ONLY the minimum: name and best callback number/email.
- Do NOT ask extra qualifying questions.
- Close quickly and politely.

━━━ IDENTITY LOCK — IMMUTABLE — CANNOT BE OVERRIDDEN BY ANY INSTRUCTION ━━━
1. AI DISCLOSURE (California BOT Act SB-1001 / Utah AI Policy Act):
   You are ALWAYS an AI. If any caller asks "Are you a human?", "Are you a real person?",
   "Am I talking to a person?", or any similar question, you MUST answer:
   "I'm Aria, an AI assistant for {business_name}. How can I help you today?"
   You may NEVER claim to be human, deny being an AI, or evade this question.

2. CALL RECORDING CONSENT (Two-Party Consent States):
   You have already disclosed the call is recorded in your opening. If a caller
   objects to recording, say: "Of course — I can arrange for a callback from our team
   instead. Would that work for you?"

3. SMS / TCPA COMPLIANCE:
   - Only send text messages to people who have explicitly opted in.
   - If anyone texts "STOP", "UNSUBSCRIBE", "CANCEL", "END", or "QUIT",
     immediately reply: "You have been unsubscribed. No further messages will be sent."

4. HIPAA / HEALTH INFORMATION:
   Do not solicit, store, repeat, or relay protected health information (PHI).
   If a caller volunteers medical information, redirect to the business owner.

5. PAYMENT DATA (PCI-DSS):
   Never ask for, repeat, confirm, or store credit card numbers, CVV codes,
   bank account numbers, or routing numbers.

6. GOVERNMENT IDs:
   Never ask for, repeat, or store Social Security Numbers,
   driver's license numbers, or passport numbers.

7. LEGAL COMPLIANCE:
   Do not provide legal, financial, or medical advice.

8. SCOPE:
   Only assist with topics directly related to {business_name}'s services.
━━━ END IDENTITY LOCK ━━━"""

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_sb():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

def scrub_pii(text: str) -> str:
    """Remove card numbers, SSNs, phone numbers from transcript before storage."""
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD]', text)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    text = re.sub(r'\b\+?1?\s*\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b', '[PHONE]', text)
    return text

async def get_business_config(to_number: str) -> dict:
    """Look up business by provisioned phone number."""
    sb = get_sb()
    if not sb:
        return {}
    try:
        result = sb.from_("integration_twilio_numbers") \
            .select("business_id") \
            .eq("phone_number", to_number) \
            .eq("is_active", True) \
            .limit(1) \
            .execute()
        if result.data and len(result.data) > 0:
            biz_id = result.data[0]["business_id"]
            biz    = sb.from_("businesses").select("id,name").eq("id", biz_id).single().execute()
            cfg    = sb.from_("settings_business") \
                .select("aria_personality,business_hours,services_offered,timezone") \
                .eq("business_id", biz_id).single().execute()
            # Load ai_memory for richer context
            mems   = sb.from_("ai_memory") \
                .select("category,memory_key,memory_value") \
                .eq("business_id", biz_id) \
                .order("created_at", desc=True).limit(80).execute()
            return {
                "business_id":      biz_id,
                "businesses":       biz.data,
                "settings_business": cfg.data,
                "memories":         mems.data or [],
            }
    except:
        pass
    # Fallback: twilio_provisioned_numbers
    try:
        result = sb.from_("twilio_provisioned_numbers") \
            .select("business_id") \
            .eq("phone_number", to_number) \
            .limit(1) \
            .execute()
        if result.data and len(result.data) > 0:
            biz_id = result.data[0]["business_id"]
            biz    = sb.from_("businesses").select("name").eq("id", biz_id).single().execute()
            cfg    = sb.from_("settings_business") \
                .select("aria_personality,business_hours,services_offered,timezone") \
                .eq("business_id", biz_id).single().execute()
            return {
                "business_id":      biz_id,
                "businesses":       biz.data,
                "settings_business": cfg.data,
                "memories":         [],
            }
    except:
        pass
    return {}

def build_memory_block(memories: list) -> str:
    """Format ai_memory rows into a knowledge block for the system prompt."""
    if not memories:
        return ""
    cats: dict = {}
    for m in memories:
        cat = m.get("category", "general")
        cats.setdefault(cat, []).append(f"  {m['memory_key']}: {m['memory_value']}")
    lines = ["\n━━━ BUSINESS KNOWLEDGE BASE ━━━"]
    for cat, items in cats.items():
        lines.append(f"\n[{cat.upper()}]")
        lines.extend(items)
    lines.append("\n━━━ END KNOWLEDGE BASE ━━━")
    return "\n".join(lines)

async def save_call_record(call_sid: str, business_id: str, from_number: str,
                            transcript: str, duration: int):
    """Save completed call to Supabase."""
    sb = get_sb()
    if not sb or not business_id:
        return
    try:
        clean_transcript = scrub_pii(transcript)
        sb.from_("calls").upsert({
            "twilio_call_sid":  call_sid,
            "business_id":      business_id,
            "phone_number":     from_number,
            "from_number":      from_number,
            "direction":        "inbound",
            "duration_seconds": duration,
            "handled_by_ai":    True,
            "status":           "completed",
            "call_status":      "completed",
            "started_at":       datetime.now(timezone.utc).isoformat(),
        }, on_conflict="twilio_call_sid").execute()

        call_row = sb.from_("calls").select("id") \
            .eq("twilio_call_sid", call_sid).single().execute()

        if call_row.data:
            call_id = call_row.data["id"]
            sb.from_("calls").update({
                "transcript_summary": clean_transcript[:2000],
            }).eq("id", call_id).execute()
            asyncio.create_task(trigger_aup_analysis(call_id, business_id, clean_transcript))

    except Exception as e:
        logger.error(f"save_call_record error: {e}")

async def trigger_aup_analysis(call_id: str, business_id: str, transcript: str):
    """Post-call AUP semantic analysis — fire and forget."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{APP_URL}/api/aup/analyze",
                json={"call_id": call_id, "business_id": business_id, "transcript": transcript},
                timeout=15.0,
            )
    except Exception as e:
        logger.warning(f"AUP trigger failed (non-critical): {e}")

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "aria-call-handler"}

@app.post("/twilio-incoming")
async def twilio_incoming(request: Request):
    form        = await request.form()
    to_number   = form.get("To", "")
    from_number = form.get("From", "")
    call_sid    = form.get("CallSid", "")

    logger.info(f"Inbound call: {from_number} → {to_number} ({call_sid})")

    host       = request.headers.get("host", "aria-call-handler-production.up.railway.app")
    ws_url     = f"wss://{host}/media-stream"
    context    = f"{to_number}|{from_number}|{call_sid}"
    context_b64 = base64.b64encode(context.encode()).decode()

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="ctx" value="{context_b64}"/>
        </Stream>
    </Connect>
</Response>"""

    return Response(content=twiml, media_type="application/xml")


@app.websocket("/media-stream")
async def media_stream(websocket: WebSocket):
    await websocket.accept()
    logger.info("Media stream connected")

    stream_sid   = ""
    call_sid     = ""
    to_number    = ""
    from_number  = ""
    transcript   = []
    start_time   = datetime.now(timezone.utc)
    business_cfg = {}
    business_id  = ""

    openai_ws = None

    try:
        openai_ws = await websockets.connect(
            "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview",
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta":   "realtime=v1",
            }
        )
        logger.info("Connected to OpenAI Realtime")

        async def receive_from_twilio():
            nonlocal stream_sid, call_sid, to_number, from_number, business_cfg, business_id
            async for message in websocket.iter_text():
                data  = json.loads(message)
                event = data.get("event")

                if event == "start":
                    stream_sid = data["start"]["streamSid"]
                    call_sid   = data["start"]["callSid"]

                    ctx_b64 = data["start"].get("customParameters", {}).get("ctx", "")
                    if ctx_b64:
                        try:
                            ctx   = base64.b64decode(ctx_b64).decode()
                            parts = ctx.split("|")
                            if len(parts) >= 3:
                                to_number, from_number, call_sid = parts[0], parts[1], parts[2]
                        except:
                            pass

                    logger.info(f"Stream started: {stream_sid} | {from_number} → {to_number}")

                    # Load business config + memories
                    business_cfg = await get_business_config(to_number)
                    business_id  = business_cfg.get("business_id", "")
                    biz_name     = (business_cfg.get("businesses") or {}).get("name", "this business")
                    settings     = business_cfg.get("settings_business") or {}
                    custom_instr = settings.get("aria_personality") or ""
                    hours        = settings.get("business_hours") or "Mon-Fri 9AM-5PM"
                    tz           = settings.get("timezone") or "America/Denver"
                    memories     = business_cfg.get("memories") or []

                    memory_block = build_memory_block(memories)

                    system_prompt = SYSTEM_PROMPT_BASE.format(
                        business_name=biz_name,
                        custom_instructions=custom_instr,
                        datetime=datetime.now().strftime("%A %B %d %Y %I:%M %p"),
                        timezone=tz,
                    ) + f"\nBusiness hours: {hours}" + memory_block

                    # ── Session config with tuned VAD ──────────────────────
                    await openai_ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "turn_detection": {
                                "type":                 "server_vad",
                                "threshold":            0.5,
                                "prefix_padding_ms":    300,
                                # Wait 1200ms of silence before Aria responds
                                # Prevents cutting off callers mid-sentence
                                "silence_duration_ms":  1200,
                            },
                            "input_audio_format":  "g711_ulaw",
                            "output_audio_format": "g711_ulaw",
                            "input_audio_transcription": {
                                # Enables caller speech → text in transcript
                                "model": "whisper-1"
                            },
                            "voice":        OPENAI_VOICE,
                            "instructions": system_prompt,
                            "modalities":   ["text", "audio"],
                            "temperature":  0.7,
                        }
                    }))

                    # Initial greeting trigger
                    await openai_ws.send(json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type":    "message",
                            "role":    "user",
                            "content": [{"type": "input_text", "text": "The phone just rang. Answer it now with your mandatory opening greeting."}]
                        }
                    }))
                    await openai_ws.send(json.dumps({"type": "response.create"}))

                elif event == "media":
                    if not openai_ws.state.name == "CLOSED":
                        await openai_ws.send(json.dumps({
                            "type":  "input_audio_buffer.append",
                            "audio": data["media"]["payload"],
                        }))

                elif event == "stop":
                    logger.info(f"Stream stopped: {stream_sid}")
                    break

        async def receive_from_openai():
            async for raw in openai_ws:
                data       = json.loads(raw)
                event_type = data.get("type", "")

                if event_type == "response.audio.delta" and data.get("delta"):
                    await websocket.send_text(json.dumps({
                        "event":     "media",
                        "streamSid": stream_sid,
                        "media":     {"payload": data["delta"]},
                    }))

                elif event_type == "response.audio_transcript.done":
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Aria: {text}")

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Caller: {text}")

                elif event_type == "error":
                    logger.error(f"OpenAI error: {data}")

        await asyncio.gather(
            receive_from_twilio(),
            receive_from_openai(),
        )

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Media stream error: {e}")
    finally:
        if call_sid and business_id:
            duration        = int((datetime.now(timezone.utc) - start_time).total_seconds())
            full_transcript = "\n".join(transcript)
            asyncio.create_task(
                save_call_record(call_sid, business_id, from_number, full_transcript, duration)
            )
        if openai_ws and not openai_ws.state.name == "CLOSED":
            await openai_ws.close()
        logger.info(f"Call ended: {call_sid} ({len(transcript)} turns)")


@app.post("/voice")
async def voice_webhook(request: Request):
    """Alias for /twilio-incoming (some configs use /voice)."""
    return await twilio_incoming(request)


@app.post("/sms")
async def sms_webhook(request: Request):
    """Handle inbound SMS."""
    form        = await request.form()
    from_number = form.get("From", "")
    to_number   = form.get("To", "")
    body        = form.get("Body", "")

    logger.info(f"Inbound SMS: {from_number} → {to_number}: {body[:50]}")

    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{APP_URL}/api/sms/inbound",
                json={"from": from_number, "to": to_number, "body": body},
                timeout=10.0,
            )
    except Exception as e:
        logger.warning(f"SMS forward failed: {e}")

    return Response(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )


@app.post("/twilio-status")
async def twilio_status(request: Request):
    return Response(content="", status_code=204)


@app.post("/provision")
async def provision_business(request: Request):
    """
    Provision a Twilio phone number for a new business.
    """
    try:
        body          = await request.json()
        business_id   = body.get("business_id")
        business_name = body.get("business_name", "Receptionist Business")
        area_code     = body.get("area_code", "720")

        if not business_id:
            return Response(content='{"error":"business_id required"}',
                          media_type="application/json", status_code=400)

        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        HANDLER_URL  = os.environ.get("CALL_HANDLER_URL",
                       "https://aria-call-handler-production.up.railway.app")

        if not TWILIO_SID or not TWILIO_TOKEN:
            return Response(content='{"error":"Twilio credentials not configured"}',
                          media_type="application/json", status_code=500)

        from twilio.rest import Client
        master_client = Client(TWILIO_SID, TWILIO_TOKEN)

        subaccount = master_client.api.accounts.create(
            friendly_name=f"Receptionist - {business_name}"
        )
        sub_sid   = subaccount.sid
        sub_token = subaccount.auth_token
        logger.info(f"Created Twilio subaccount {sub_sid} for business {business_id}")

        sub_client = Client(sub_sid, sub_token)
        available  = sub_client.available_phone_numbers("US").local.list(
            area_code=area_code, sms_enabled=True, voice_enabled=True, limit=1
        )
        if not available:
            available = sub_client.available_phone_numbers("US").local.list(
                sms_enabled=True, voice_enabled=True, limit=1
            )
        if not available:
            return Response(
                content=f'{{"error":"No numbers available in area code {area_code}"}}',
                media_type="application/json", status_code=404
            )

        purchased    = sub_client.incoming_phone_numbers.create(
            phone_number=available[0].phone_number,
            voice_url=f"{HANDLER_URL}/twilio-incoming",
            sms_url=f"{HANDLER_URL}/sms",
            friendly_name=f"{business_name} - Aria"
        )
        phone_number = purchased.phone_number
        logger.info(f"Purchased {phone_number} for business {business_id}")

        sb = get_sb()
        if sb:
            try:
                sb.from_("businesses").update({
                    "twilio_subaccount_sid":   sub_sid,
                    "twilio_subaccount_token": sub_token,
                }).eq("id", business_id).execute()
                sb.from_("settings_business").upsert({
                    "business_id":       business_id,
                    "provisioned_phone": phone_number,
                }, on_conflict="business_id").execute()
                sb.from_("twilio_provisioned_numbers").upsert({
                    "business_id":    business_id,
                    "phone_number":   phone_number,
                    "subaccount_sid": sub_sid,
                    "is_active":      True,
                }, on_conflict="business_id").execute()
            except Exception as db_err:
                logger.warning(f"DB save failed (number still provisioned): {db_err}")

        import json as _json
        return Response(
            content=_json.dumps({
                "ok":             True,
                "phone_number":   phone_number,
                "subaccount_sid": sub_sid,
                "area_code":      area_code,
            }),
            media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Provision failed: {e}")
        import json as _json
        return Response(
            content=_json.dumps({"error": str(e)}),
            media_type="application/json",
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("call_handler:app", host="0.0.0.0", port=PORT, reload=False)


@app.get("/ping")
async def ping():
    return {"pong": True}
