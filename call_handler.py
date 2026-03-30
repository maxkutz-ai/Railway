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
try:
    from zoneinfo import ZoneInfo          # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo # fallback

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

# Legacy — services now embedded in SYSTEM_PROMPT_BASE
RECEPTIONIST_SERVICES_BLOCK = """
━━━ RECEPTIONIST.CO PLATFORM SERVICES ━━━
Aria is the AI-powered platform ITSELF. When callers ask what the business does:
1. AI Voice Receptionist (Aria Voice) — answers phone calls 24/7, books appointments,
   takes messages, handles FAQs. Powered by OpenAI Realtime + Twilio.
2. AI Video Receptionist (Aria Video) — a live video avatar (face + voice) that appears
   on the business website or in-lobby kiosk. Powered by LiveKit + Simli + OpenAI.
3. CRM Dashboard — unified inbox, call logs, contacts, appointments, analytics.
4. AI Knowledge Base (Business Brain) — learns from website scans, uploaded docs, manual facts.
5. Outbound Campaigns — SMS and email follow-up sequences.
6. Integrations — Google Calendar, Outlook, Cal.com, Stripe, Zapier.
Pricing: Starter (free trial), Pro ($99/mo), Enterprise (custom).
Website: https://www.receptionist.co
Demo / booking: https://app.receptionist.co/onboarding
━━━ END SERVICES ━━━"""

SYSTEM_PROMPT_BASE = """You are {aria_name}, a professional AI receptionist for {business_name}.
Current date and time: {datetime}
Business timezone: {timezone}
{custom_instructions}

━━━ VOICE & DELIVERY STYLE ━━━
Calm, premium concierge tone. Speak clearly and slightly slower than normal.
Crisp enunciation — upbeat but never salesy. Smile in your voice.
Short sentences. Micro-pauses between sentences.
Raise pitch slightly on questions.
Pronounce the company name strictly as "Receptionist dot co" — pause briefly after saying it.
Custom vocabulary: say "Twilio" as "Twill-ee-oh", "CRM" as three letters "C-R-M", 
"LLM" as three letters "L-L-M", "API" as three letters "A-P-I".
━━━ END VOICE STYLE ━━━

━━━ MANDATORY OPENING — SAY THIS EXACTLY ON EVERY CALL ━━━
{opening_greeting}

━━━ RESPONSE STYLE ━━━
- Keep every response under 3 sentences so the caller has room to speak.
- Speak naturally, warmly, and concisely.
- Never read long lists aloud — weave information into natural sentences.
- Never say "routing." Say "making sure this goes to the right person."

━━━ TURN-TAKING / WAIT RULE (CRITICAL — PREVENTS INTERRUPTIONS) ━━━
- Ask ONE question at a time. NEVER combine two questions in one sentence.
- After asking a question, STOP and wait silently for the full answer.
- Do NOT acknowledge ("Perfect / Thanks") until you have ACTUALLY HEARD the answer.
- If the caller starts answering, do NOT talk over them. Wait until they fully finish.
- If the caller says "yes / mhmm / correct", pause one full beat before continuing.
- If you are mid-sentence and the caller speaks, STOP immediately and listen.

━━━ CALLER ID RULE ━━━
You always have access to the caller's phone number from the system.
The caller's number is: {caller_number}
If a caller asks "Can you see my number?" or "Do you have my number?", say:
"Yes, I can see your number ends in {caller_last4}. I'd still like to confirm
the best callback number for you — is this the right one to reach you?"
This builds trust and avoids asking for information you already have.
━━━ END CALLER ID RULE ━━━

━━━ CAPTURE LOOP RULE ━━━
After answering a maximum of TWO questions about the product or services, pivot to
lead capture. Say exactly:
"May I get your name and a good callback number in case we get disconnected?"
Then ask for their email. Confirm all details before ending the call.

━━━ PHONETIC MIRRORING RULE ━━━
When confirming spelling, mirror the caller's own phonetic anchors.
If they say "G as in George", confirm with "G as in George" — not "G as in Golf".
If they mix standard and custom anchors, use whichever they used most recently.
Only fall back to standard military alphabet if the caller provides no anchors.
━━━ END PHONETIC MIRRORING ━━━

━━━ EMAIL CAPTURE RULE (CRITICAL) ━━━
- If the caller gives an email, ALWAYS ask them to spell it letter by letter.
- Confirm back by spelling it yourself letter by letter (dot, underscore, dash included).
- Wait for "yes that's correct" before proceeding.
- Never confirm by reading the full email address as one word.
- Wrong emails cause failed follow-ups. Always confirm.

━━━ TIMEZONE RULE ━━━
- Before booking any appointment, ask: "What timezone are you in?" — then WAIT.
- Confirm: "Got it — I'll schedule that in [timezone]."
- Business timezone: {timezone}. If caller is different, state the conversion clearly.
  Example: "That's 3:00 PM Mountain — 2:00 PM Pacific. Does that still work?"
- Never book without confirming timezone first.

━━━ PRICE CUSHION RULE ━━━
Never provide a final, binding price. Always use buffer phrases:
- "Prices typically start at..."
- "An estimated range is..."
- "The final price is confirmed by our specialist after reviewing your situation."

━━━ MEDICAL / LIABILITY WALL ━━━
If a caller asks a diagnostic or medical question ("Is this safe if I'm pregnant?",
"What should I do about this rash?", "Is this treatment right for me?"):
- Say: "I'm not able to provide medical or treatment advice — that's something our
  licensed professionals handle directly. Can I have someone call you back?"
- Never attempt to answer. Offer a callback every time.

━━━ EMERGENCY BYPASS RULE ━━━
If you detect ANY of these emergency keywords: {emergency_keywords}
Immediately say:
"This sounds like an emergency. Please call 911 if there is immediate danger.
I'm flagging this for our team to call you right now — what's your best number?"
Stop the standard flow entirely. Capture name and number only.

━━━ TECHNICAL SUPPORT DETECTION ━━━
If a caller mentions account issues, bugs, broken numbers, or billing problems:
Stop pitching. Say: "I'm the AI demo line, so I don't have access to support tickets.
I'm flagging this for our engineering team right now — they'll reach out to the email
on your account shortly." Do not troubleshoot. Do not continue the sales flow.

━━━ CUSTOM FEATURE / INTEGRATION REQUESTS ━━━
If a caller asks for a feature or integration not listed in the knowledge base:
Say: "That's a great question for our founding team. I can help you book a quick call
with them to discuss your specific workflow needs."
Never promise custom integrations or specific roadmap timelines.

━━━ PRICE NEGOTIATION WALL ━━━
Never negotiate pricing. If pressed for exact prices say:
"Because every business has different call volumes, I want to make sure you get the most
accurate quote. Max can build a custom pricing tier for you on a quick 15-minute call."

━━━ GOODBYE DETECTION ━━━
If the caller says "bye", "goodbye", "bye-bye", "take care", "have a good day",
"talk later", "gotta go", "I'm done", "that's all" AND THEN STOPS TALKING:
- Stop your current task, thank them warmly, and close gracefully.

EXCEPTION — Turn-End Prediction:
If the caller says a goodbye phrase BUT immediately follows with a question
or new statement (e.g. "Bye-bye... but what's the number I'm calling from?"),
IGNORE the goodbye and answer the question. The goodbye was not final.
Only close the call when the goodbye is the LAST thing they say with no follow-up.
━━━ END GOODBYE DETECTION ━━━

━━━ TIME LIMIT WRAP-UP ━━━
When the system signals you are near the call time limit, say exactly:
"I want to be respectful of your time — my system has a standard call limit so I can
assist all of our incoming callers today. I have time for one last quick question,
or I can have our team call you right back. What would you prefer?"

━━━ KNOWLEDGE BASE — RECEPTIONIST.CO SERVICES ━━━
What is Receptionist.co?
  AI-powered virtual receptionist platform for service-based businesses (spas, clinics,
  home services, law firms, etc.). Aria handles inbound calls, books appointments, and
  answers FAQs 24/7 — fully automated.

What is AI Video? (answer if asked about "AI Video", "HeyGen", or "video avatars"):
  "We use advanced AI Video technology to create a lifelike video avatar — Aria's face
  appears on your website or lobby screen, speaking directly to visitors. It puts a
  human face on the AI experience. Max can walk you through exactly how we build that!"

Services:
  1. AI Voice Receptionist — answers every inbound call, books appointments, captures
     leads, handles FAQs. Powered by OpenAI Realtime + Twilio. Works 24/7.
  2. AI Video Receptionist — lifelike AI video avatar for website or in-lobby kiosk.
     Powered by LiveKit, Simli, and OpenAI.
  3. CRM Dashboard — unified inbox, call logs, contacts, appointments, analytics.
  4. Business Brain — Aria learns from your website, uploaded docs, and manual facts.
  5. Outbound Campaigns — SMS and email follow-up sequences for leads and appointments.
  6. Integrations — Google Calendar, Outlook, Cal.com. Mindbody/Salesforce coming soon.

━━━ PRICING & INDUSTRY MODULE ━━━
When a caller asks about cost, pricing, or services, follow this exact sequence:

STEP 1 — Ask the discovery question first:
"To give you the most accurate details for your business, may I ask what industry you are in?"

STEP 2 — Respond based on their industry:

IF Medical / MedSpa / Wellness / Weight Loss / Clinic:
"For medical practices, we focus on HIPAA-compliant lead capture and secure patient
scheduling. Our Starter plan is $295 per month — that includes 500 voice minutes and
a dedicated local number. There is a one-time $299 setup fee to ensure your secure
data pipeline and calendar integrations are perfectly configured.
Does that sound like what you're looking for?"

IF Home Services (Plumber / HVAC / Electrician / Roofer / Landscaping / Contractor):
"For home services, we prioritize 24/7 emergency dispatch and lead capture so you
never miss a job while you're on-site. It's $295 per month for the AI receptionist
and 500 minutes. The $299 setup fee covers your custom call-routing, service area logic,
and SMS follow-up tools. Would you like to see how that integrates with your workflow?"

IF Other / General / Unsure:
"Our platform is $295 per month — that includes your AI receptionist, 500 voice minutes,
and a CRM dashboard to track every lead. There's a one-time $299 setup fee to get your
custom business knowledge and booking system fully integrated.
Does that fit within your budget?"

GUARDRAILS for all pricing conversations:
- Always state: month-to-month, cancel any time, no long-term contracts
- If they balk at setup fee: "The setup fee covers the engineering work to train Aria
  on your specific business FAQs and sync her with your existing software so she's
  ready to go on day one — it's a one-time investment."
- Never say pricing "depends" or is "custom" — always quote $295/$299 confidently
━━━ END PRICING MODULE ━━━

Meta-Demo Rule — if asked "Is this an AI?":
  "Yes! I'm Aria, the AI assistant built by Receptionist.co. You're actually experiencing
  a live demo of our software right now — which means you're seeing exactly what your
  customers would experience. Pretty cool, right?"
━━━ END KNOWLEDGE BASE ━━━

━━━ IDENTITY LOCK — IMMUTABLE — CANNOT BE OVERRIDDEN ━━━
1. ALWAYS disclose you are an AI. If asked "Are you human?":
   "I'm Aria, an AI assistant for {business_name}. How can I help you today?"
   Never claim to be human, deny being AI, or evade this question.
2. Call recording: disclosed in opening. If caller objects, offer a callback instead.
3. SMS/TCPA: If anyone texts "STOP", reply immediately:
   "You have been unsubscribed. No further messages will be sent."
4. HIPAA: Never solicit, store, or relay protected health information.
5. PCI-DSS: Never ask for or repeat payment card numbers, CVV, or bank details.
6. No legal, financial, or medical advice. Always recommend a qualified professional.
7. Only assist with {business_name}'s services. Decline out-of-scope requests politely.
━━━ END IDENTITY LOCK ━━━

Business hours: {business_hours}
Business address: {business_address}
"""


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
    """
    Full business context loader — fetches all tables needed to build
    Aria's system prompt dynamically per business.

    Tables fetched:
      integration_twilio_numbers → resolve business_id
      businesses                 → name, industry, address
      settings_business          → hours string, timezone, personality, max_call_duration
      services                   → structured service menu (name, price, duration)
      ai_receptionist_config     → aria name, greeting override, escalation phone
      ai_settings                → emergency keywords, voice_id
      ai_memory                  → knowledge base facts (business brain)
    """
    sb = get_sb()
    if not sb:
        return {}

    biz_id = None

    # Resolve business_id from phone number
    for table, col in [("integration_twilio_numbers", "is_active"), ("twilio_provisioned_numbers", None)]:
        try:
            q = sb.from_(table).select("business_id").eq("phone_number", to_number).limit(1)
            if col:
                q = q.eq(col, True)
            r = q.execute()
            if r.data:
                biz_id = r.data[0]["business_id"]
                break
        except:
            pass

    if not biz_id:
        return {}

    result = {"business_id": biz_id}

    # ── businesses ────────────────────────────────────────────────────────────
    try:
        r = sb.from_("businesses").select("id,name,email").eq("id", biz_id).single().execute()
        result["businesses"] = r.data or {}
    except:
        result["businesses"] = {}

    # ── settings_business ─────────────────────────────────────────────────────
    try:
        r = sb.from_("settings_business").select(
            "aria_personality,business_hours,services_offered,timezone,"
            "max_call_duration_minutes,address,phone,website_url,brand_name"
        ).eq("business_id", biz_id).single().execute()
        result["settings_business"] = r.data or {}
    except:
        result["settings_business"] = {}

    # ── services (structured menu) ────────────────────────────────────────────
    try:
        r = sb.from_("services").select("name,price,duration_minutes,description").eq("business_id", biz_id).eq("is_active", True).execute()
        result["services"] = r.data or []
    except:
        result["services"] = []

    # ── ai_receptionist_config ────────────────────────────────────────────────
    try:
        r = sb.from_("ai_receptionist_config").select("name,greeting,escalation_phone").eq("business_id", biz_id).maybeSingle().execute()
        result["ai_config"] = r.data or {}
    except:
        result["ai_config"] = {}

    # ── ai_settings (emergency keywords, voice) ───────────────────────────────
    try:
        r = sb.from_("ai_settings").select("emergency_keywords,voice_id,max_call_duration_mins").eq("business_id", biz_id).single().execute()
        result["ai_settings"] = r.data or {}
    except:
        result["ai_settings"] = {}

    # ── ai_memory (business brain facts) ─────────────────────────────────────
    try:
        r = sb.from_("ai_memory").select("category,memory_key,memory_value").eq("business_id", biz_id).order("created_at", desc=True).limit(80).execute()
        result["memories"] = r.data or []
    except:
        result["memories"] = []

    return result

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





async def extract_lead_from_transcript(
    business_id: str,
    call_sid: str,
    from_number: str,
    transcript: str,
):
    """
    Post-call: send transcript to GPT-4o-mini to extract structured lead data,
    then UPSERT into the contacts table.
    - If phone already exists for business → UPDATE name/email/summary
    - If new caller → CREATE contact record
    """
    if not transcript or not business_id:
        return

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "max_tokens": 200,
                    "temperature": 0,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Extract structured lead data from this call transcript. "
                                "Return ONLY valid JSON with these keys: "
                                "first_name (string or null), "
                                "last_name (string or null), "
                                "email (string or null), "
                                "phone (string or null — 10 digits only, no formatting), "
                                "interest (string or null — what they want), "
                                "intent (one of: booking, inquiry, support, other). "
                                "If a field is not mentioned, return null. "
                                "Return nothing except the JSON object."
                            ),
                        },
                        {"role": "user", "content": transcript[:4000]},
                    ],
                },
                timeout=20.0,
            )

        result = resp.json()
        raw    = result["choices"][0]["message"]["content"].strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        import json as _json
        extracted = _json.loads(raw)
        logger.info(f"Lead extracted: {extracted}")

        # Build the contact row
        phone_clean = (extracted.get("phone") or "").replace("-","").replace("(","").replace(")","").replace(" ","")
        if not phone_clean and from_number:
            # Fall back to caller ID
            phone_clean = from_number.replace("+1","").replace("+","").strip()[-10:]

        if not phone_clean:
            return

        sb = get_sb()
        if not sb:
            return

        contact_row = {
            "business_id":  business_id,
            "phone":        phone_clean,
            "lead_status":  "new",
            "source":       "call",
            "last_summary": transcript[:500],
            "ai_notes":     _json.dumps({
                "interest":    extracted.get("interest"),
                "intent":      extracted.get("intent"),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "call_sid":    call_sid,
            }),
            "updated_at":   datetime.now(timezone.utc).isoformat(),
        }

        if extracted.get("first_name"):
            contact_row["first_name"] = extracted["first_name"]
        if extracted.get("last_name"):
            contact_row["last_name"] = extracted["last_name"]
        if extracted.get("email"):
            contact_row["email"] = extracted["email"]

        # UPSERT — match on (business_id, phone)
        # If caller already exists → update name/email/summary
        # If new caller → create the record
        upsert_result = sb.from_("contacts").upsert(
            contact_row,
            on_conflict="business_id,phone",
        ).select("id").execute()

        logger.info(f"Contact upserted: {phone_clean} for {business_id}")

        # Link the contact back to the call record so Recent Activity shows name
        if upsert_result.data and call_sid:
            contact_id = upsert_result.data[0]["id"]
            sb.from_("calls").update({
                "contact_id": contact_id,
            }).eq("twilio_call_sid", call_sid).execute()
            logger.info(f"Linked call {call_sid} → contact {contact_id}")

    except Exception as e:
        logger.warning(f"Lead extraction failed (non-critical): {e}")

async def notify_lead_captured(business_id: str, transcript: str, from_number: str):
    """
    Fire a lead notification (SMS + email) to the business owner
    when Aria captures a full lead (name + phone + email detected in transcript).
    """
    import re
    # Quick heuristic: check if transcript has email pattern
    # Notify if we captured a name OR email — Steve (no email) still gets a notification
    has_email   = bool(re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', transcript))
    has_name    = bool(re.search(r"(?:name is|I am|call me|my name)\s+([A-Z][a-z]{1,20})", transcript))
    if not has_email and not has_name:
        return  # no lead captured

    sb = get_sb()
    if not sb:
        return

    try:
        # Get business notification settings
        cfg = sb.from_("settings_business").select(
            "brand_name,phone,notification_email"
        ).eq("business_id", business_id).single().execute()
        if not cfg.data:
            return

        biz_name   = cfg.data.get("brand_name") or "your business"
        notify_ph  = cfg.data.get("phone") or ""
        notify_email = cfg.data.get("notification_email") or ""

        # Format caller number
        caller = from_number.replace("+1","").strip()
        if len(caller) == 10:
            caller = f"({caller[:3]}) {caller[3:6]}-{caller[6:]}"

        lead_label = "🎉 Full lead" if has_email else "⚡ Hot lead (no email)"
        msg = (
            f"{lead_label} from {caller}\n"
            f"Aria captured info for {biz_name}."
            + ("\nNo email — call them back!" if not has_email else "")
            + "\nCRM: https://app.receptionist.co/dashboard"
        )

        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        TWILIO_FROM  = os.environ.get("TWILIO_NOTIFY_FROM", "")  # your sending number
        if notify_ph and TWILIO_SID and TWILIO_TOKEN and TWILIO_FROM:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                    auth=(TWILIO_SID, TWILIO_TOKEN),
                    data={"From": TWILIO_FROM, "To": f"+1{notify_ph.replace('(','').replace(')','').replace('-','').replace(' ','')}", "Body": msg},
                    timeout=10.0,
                )
                logger.info(f"Lead SMS sent for {business_id}")

        # Email via Resend
        RESEND_KEY = os.environ.get("RESEND_API_KEY", "")
        if notify_email and RESEND_KEY:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://api.resend.com/emails",
                    headers={"Authorization": f"Bearer {RESEND_KEY}", "Content-Type": "application/json"},
                    json={
                        "from": "Aria <notifications@receptionist.co>",
                        "to": [notify_email],
                        "subject": f"🎉 New lead captured by Aria — {biz_name}",
                        "html": f"<div style='font-family:Arial;max-width:520px;padding:24px'>"
                                f"<h2>New Lead Captured</h2>"
                                f"<p>Aria captured a new lead from <strong>{caller}</strong> on your Receptionist.co line.</p>"
                                f"<p><a href='https://app.receptionist.co/dashboard' style='background:#4F8EF7;color:#fff;padding:10px 20px;border-radius:8px;text-decoration:none;'>View in CRM →</a></p>"
                                f"<hr><p style='font-size:12px;color:#94A3B8'>Transcript preview:<br>{transcript[:400]}...</p>"
                                f"</div>",
                    },
                    timeout=10.0,
                )
                logger.info(f"Lead email sent for {business_id}")

    except Exception as e:
        logger.warning(f"Lead notification failed (non-critical): {e}")


async def silence_monitor(openai_ws, get_last_speech, get_is_responding, websocket, call_sid, timeout_secs=20):
    """
    Monitor for silence. If no caller speech for timeout_secs while Aria isn't talking,
    prompt the caller. If still silent after 8s, hang up.
    Prevents "ghost" OpenAI charges from open silent lines.
    """
    await asyncio.sleep(15)  # Grace period at call start
    warned = False
    while True:
        await asyncio.sleep(5)
        last = get_last_speech()
        if last is None or get_is_responding():
            continue
        secs_silent = (datetime.now(timezone.utc) - last).total_seconds()
        if secs_silent >= timeout_secs and not warned:
            warned = True
            # Inject a gentle prompt
            try:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text",
                                     "text": "[SYSTEM: The caller has been silent for 20 seconds. Gently ask if they are still there, and if no response comes, wrap up the call politely.]"}]
                    }
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))
            except:
                pass
        elif secs_silent >= timeout_secs + 12 and warned:
            # Hard hang up after additional 12s of silence
            try:
                await websocket.close()
            except:
                pass
            return

async def start_twilio_recording(call_sid: str):
    """Start call recording via Twilio REST API — compatible with Media Streams."""
    try:
        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not TWILIO_SID or not TWILIO_TOKEN:
            return
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Calls/{call_sid}/Recordings.json",
                auth=(TWILIO_SID, TWILIO_TOKEN),
                data={
                    "RecordingStatusCallback": f"{os.environ.get('CALL_HANDLER_URL', '')}/twilio-status",
                    "RecordingStatusCallbackMethod": "POST",
                },
                timeout=10.0,
            )
        logger.info(f"Recording started for {call_sid}: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Recording start failed (non-critical): {e}")

async def delete_active_call(call_sid: str):
    """Delete active call row on hang-up — keeps the table lean and triggers DELETE Realtime event."""
    sb = get_sb()
    if not sb or not call_sid:
        return
    try:
        sb.from_("active_calls").delete().eq("call_sid", call_sid).execute()
    except Exception as e:
        logger.warning(f"active_calls delete failed (non-critical): {e}")

async def upsert_active_call(business_id: str, call_sid: str, from_number: str, status: str, transcript_turns: list = None):
    """Write/update active call in Supabase so the dashboard live counter works."""
    sb = get_sb()
    if not sb or not business_id or not call_sid:
        return
    try:
        row = {
            "call_sid":    call_sid,
            "business_id": business_id,
            "from_number": from_number,
            "status":      status,
            "updated_at":  datetime.now(timezone.utc).isoformat(),
        }
        if transcript_turns is not None:
            row["live_transcript"] = transcript_turns
        sb.from_("active_calls").upsert(row, on_conflict="call_sid").execute()
    except Exception as e:
        logger.warning(f"active_calls upsert failed (non-critical): {e}")

async def save_call_record(call_sid: str, business_id: str, from_number: str,
                            transcript: str, duration: int, start_time_iso: str = None):
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
            "started_at":       start_time_iso or datetime.now(timezone.utc).isoformat(),
            "transcript_summary": clean_transcript[:8000],
        }, on_conflict="twilio_call_sid").execute()

        # Fetch call_id for AUP analysis
        try:
            call_row = sb.from_("calls").select("id").eq("twilio_call_sid", call_sid).maybeSingle().execute()
            if call_row.data:
                asyncio.create_task(trigger_aup_analysis(call_row.data["id"], business_id, clean_transcript))
            asyncio.create_task(notify_lead_captured(business_id, clean_transcript, from_number))
            asyncio.create_task(extract_lead_from_transcript(business_id, call_sid, from_number, clean_transcript))
        except:
            pass

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

    stream_sid    = ""
    call_sid      = ""
    to_number     = ""
    from_number   = ""
    transcript    = []
    start_time    = None  # set when Twilio start event fires (actual call start)
    business_cfg  = {}
    business_id   = ""
    max_call_mins = 10   # default — overridden from DB once start event fires
    is_responding  = False  # True while OpenAI is generating audio — guards response.cancel
    last_speech_at = None  # timestamp of last caller speech — for silence timeout

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

                    start_time = datetime.now(timezone.utc)  # actual call start
                    logger.info(f"Stream started: {stream_sid} | {from_number} → {to_number}")

                    # Start recording via Twilio REST API (compatible with Media Streams)
                    asyncio.create_task(start_twilio_recording(call_sid))

                    # Load business config + memories
                    business_cfg = await get_business_config(to_number)
                    business_id  = business_cfg.get("business_id", "")
                    biz_name     = (business_cfg.get("businesses") or {}).get("name", "this business")
                    settings     = business_cfg.get("settings_business") or {}
                    custom_instr = settings.get("aria_personality") or ""
                    hours        = settings.get("business_hours") or "Mon-Fri 9AM-5PM"
                    tz               = settings.get("timezone") or "America/Denver"
                    max_call_mins    = int(settings.get("max_call_duration_minutes") or 10)
                    memories         = business_cfg.get("memories") or []

                    memory_block = build_memory_block(memories)

                    # ── Pull all config sections ──────────────────────────
                    ai_cfg   = business_cfg.get("ai_config") or {}
                    ai_set   = business_cfg.get("ai_settings") or {}
                    svc_list = business_cfg.get("services") or []

                    # Brand name: prefer brand_name from settings, fall back to businesses.name
                    biz_name = settings.get("brand_name") or biz_name

                    # Aria's name (can be customized per business)
                    aria_name = ai_cfg.get("name") or "Aria"

                    # Format services as a readable menu
                    if svc_list:
                        svc_lines = "\n".join([
                            f"  - {s['name']}"
                            + (f" ({s['duration_minutes']} min)" if s.get('duration_minutes') else "")
                            + (f" — Starts at ${s['price']}" if s.get('price') else "")
                            + (f": {s['description']}" if s.get('description') else "")
                            for s in svc_list
                        ])
                        services_block = f"\n━━━ SERVICES MENU ━━━\n{svc_lines}\n━━━ END SERVICES ━━━"
                    else:
                        services_block = ""

                    # Custom emergency keywords (per business, or global defaults)
                    emergency_kw = ai_set.get("emergency_keywords") or [
                        "flooded", "sparking", "burst pipe", "gas leak",
                        "allergic reaction", "chest pain", "not breathing",
                        "fire", "bleeding", "emergency"
                    ]
                    if isinstance(emergency_kw, list):
                        emergency_str = ", ".join(emergency_kw)
                    else:
                        emergency_str = str(emergency_kw)

                    # Max call duration: ai_settings overrides settings_business
                    max_call_mins = int(
                        ai_set.get("max_call_duration_mins") or
                        settings.get("max_call_duration_minutes") or 10
                    )

                    # Voice: ai_settings.voice_id overrides global default
                    voice = ai_set.get("voice_id") or OPENAI_VOICE

                    # Business address
                    address = settings.get("address") or ""

                    # Detect if this is Receptionist.co's own demo line
                    is_demo = any(x in biz_name.lower() for x in ["receptionist", "receptionist.co", "receptionist, inc"])
                    opening = (
                        f"Hi! I'm {aria_name}, the AI assistant for Receptionist.co, on a recorded line. "
                        "You are actually experiencing a live demo of our software right now! How can I help you today?"
                    ) if is_demo else (
                        f"Hi! Thank you for calling {biz_name}. I'm {aria_name}, an AI assistant on a recorded line. How can I help you today?"
                    )

                    caller_last4 = from_number[-4:] if len(from_number) >= 4 else from_number
                    caller_fmt   = from_number.replace("+1", "").strip()
                    # Format: (720) 651-1325
                    if len(caller_fmt) == 10:
                        caller_fmt = f"({caller_fmt[:3]}) {caller_fmt[3:6]}-{caller_fmt[6:]}"

                    system_prompt = SYSTEM_PROMPT_BASE.format(
                        business_name=biz_name,
                        aria_name=aria_name,
                        custom_instructions=custom_instr,
                        datetime=datetime.now(ZoneInfo(tz)).strftime("%A %B %d %Y %I:%M %p %Z"),
                        timezone=tz,
                        opening_greeting=opening,
                        business_hours=hours,
                        business_address=address,
                        emergency_keywords=emergency_str,
                        caller_number=caller_fmt or "unknown",
                        caller_last4=caller_last4 or "????",
                    ) + memory_block + services_block

                    # ── Session config with tuned VAD + barge-in ──────────
                    await openai_ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "turn_detection": {
                                "type":                "server_vad",
                                "threshold":           0.6,    # less sensitive = fewer false triggers
                                "prefix_padding_ms":   200,
                                "silence_duration_ms": 800,    # respond after 800ms silence
                                # Barge-in: when caller speaks, Aria stops immediately
                                "create_response":     True,
                                "interrupt_response":  True,   # KEY: enables true barge-in
                            },
                            "input_audio_format":  "g711_ulaw",
                            "output_audio_format": "g711_ulaw",
                            "input_audio_transcription": {"model": "whisper-1"},
                            "voice":        voice,
                            "instructions": system_prompt,
                            "modalities":   ["text", "audio"],
                            "temperature":  0.7,
                        }
                    }))

                    # ── Register active call in DB (powers live dashboard counter) ──
                    asyncio.create_task(upsert_active_call(
                        business_id, call_sid, from_number, "in-progress"
                    ))

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
                    # ── Instant UI update: delete active call NOW ──
                    # save_call_record runs async later — UI clears immediately
                    if call_sid and business_id:
                        asyncio.create_task(delete_active_call(call_sid))
                    break

        async def receive_from_openai():
            async for raw in openai_ws:
                data       = json.loads(raw)
                event_type = data.get("type", "")

                if event_type == "response.audio.delta" and data.get("delta"):
                    is_responding = True
                    await websocket.send_text(json.dumps({
                        "event":     "media",
                        "streamSid": stream_sid,
                        "media":     {"payload": data["delta"]},
                    }))

                elif event_type == "response.audio_transcript.done":
                    is_responding = False  # response finished
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Aria: {text}")
                        # Push live transcript every 4 turns for Glass Box
                        if len(transcript) % 4 == 0 and business_id:
                            turns = [{"role": "ai" if t.startswith("Aria:") else "user",
                                      "text": t.split(": ", 1)[-1], "ts": datetime.now(timezone.utc).isoformat()}
                                     for t in transcript[-20:]]
                            asyncio.create_task(upsert_active_call(
                                business_id, call_sid, from_number, "in-progress", turns
                            ))

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Caller: {text}")
                        last_speech_at = datetime.now(timezone.utc)

                elif event_type == "input_audio_buffer.speech_started":
                    # Only cancel if Aria is actively generating (prevents response_cancel_not_active spam)
                    if is_responding:
                        is_responding = False
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))

                elif event_type == "error":
                    logger.error(f"OpenAI error: {data}")

        # ── Call duration watchdog ────────────────────────────────────────
        async def call_timer():
            soft_secs = (max_call_mins - 1) * 60
            hard_secs = max_call_mins * 60 + 30
            await asyncio.sleep(soft_secs)
            logger.info(f"Soft wrap-up at {max_call_mins-1}min")
            try:
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message", "role": "user",
                        "content": [{"type": "input_text", "text":
                            "[SYSTEM COMMAND — do not read aloud]: You have 60 seconds left. "
                            "Say exactly: 'I want to be respectful of your time — my system has a "
                            "standard call limit so I can assist all callers today. I have time for "
                            "one last quick question, or I can have a team member call you right back. "
                            "What would you prefer?' Then wrap up politely."
                        }]
                    }
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))
            except:
                pass
            await asyncio.sleep(hard_secs - soft_secs)
            logger.info(f"Hard disconnect at {max_call_mins}min 30s")
            try:
                await websocket.close()
            except:
                pass

        await asyncio.gather(
            receive_from_twilio(),
            receive_from_openai(),
            call_timer(),
            silence_monitor(
                openai_ws,
                lambda: last_speech_at,
                lambda: is_responding,
                websocket,
                call_sid,
            ),
        )

    except WebSocketDisconnect:
        logger.info("Twilio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Media stream error: {e}")
    finally:
        if call_sid and business_id:
            duration        = int((datetime.now(timezone.utc) - (start_time or datetime.now(timezone.utc))).total_seconds())
            await asyncio.sleep(2)  # let in-flight OpenAI transcript events complete
            full_transcript = "\n".join(transcript)
            turn_count      = len(transcript)
            logger.info(f"Saving call: {call_sid} | {turn_count} turns | {len(full_transcript)} chars | {duration}s")
            try:
                await save_call_record(call_sid, business_id, from_number, full_transcript, duration, start_time.isoformat() if start_time else None)
                logger.info(f"Call saved: {call_sid}")
            except Exception as save_err:
                logger.error(f"SAVE FAILED for {call_sid}: {save_err}")
            # delete_active_call already fired on stream stop — this is a safety net
            asyncio.create_task(delete_active_call(call_sid))
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
    """
    Twilio calls this for:
    1. Call status updates (CallStatus field)
    2. Recording status callbacks (RecordingStatus field)
    The recording fires twice: status=in-progress (start) and status=completed (done).
    We only save the URL when RecordingStatus=completed.
    """
    form = await request.form()
    call_sid         = form.get("CallSid", "")
    recording_url    = form.get("RecordingUrl", "")
    recording_status = form.get("RecordingStatus", "")  # in-progress | completed | failed
    call_status      = form.get("CallStatus", "")

    logger.info(f"twilio-status: CallSid={call_sid} RecordingStatus={recording_status} CallStatus={call_status}")

    # Only save when recording is fully completed (not in-progress)
    if recording_url and call_sid and recording_status == "completed":
        async def save_recording():
            sb = get_sb()
            if not sb:
                return
            try:
                # Twilio appends .json — strip and use .mp3 for direct playback
                clean_url = recording_url.replace(".json", "") + ".mp3"
                sb.from_("calls").update({
                    "recording_url": clean_url
                }).eq("twilio_call_sid", call_sid).execute()
                logger.info(f"Recording saved for {call_sid}")
            except Exception as e:
                logger.warning(f"recording save failed: {e}")
        asyncio.create_task(save_recording())

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
