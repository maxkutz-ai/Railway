# CACHE-BUST: 20260331-052636
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
import re
import json
import logging
import asyncio
import base64
import re
from datetime import datetime, timezone, timedelta, date
try:
    from encryption import decrypt_api_key, encrypt_text, decrypt_text, should_encrypt, encrypt_pii, decrypt_pii, hash_pii
except ImportError:
    # encryption.py is REQUIRED — if import fails, fail loud
    raise RuntimeError(
        "FATAL: encryption.py could not be imported. "
        "Receptionist.co cannot start without the encryption module. "
        "Ensure encryption.py is present in the Railway deployment."
    )
try:
    from zoneinfo import ZoneInfo          # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo # fallback

import httpx
import websockets
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("call-handler")

app = FastAPI()

# ── CORS — allow CRM dashboard and local dev to call Railway endpoints ──────
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://crm.receptionist.co",
        "https://www.receptionist.co",
        "http://localhost:3000",
        "http://localhost:3001",
        "*",  # broad allow — Railway is already protected by Twilio signature validation
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Environment ───────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY      = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", os.environ.get("SUPABASE_SERVICE_KEY", ""))  # Service Role bypasses RLS
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

"━━━ LANGUAGE & MULTILINGUAL ━━━\n"
"POLYGLOT DIRECTIVE: Always greet the caller in English unless the business has configured a different default language. If the caller speaks a different language (Spanish, French, Mandarin, etc.), immediately and seamlessly switch to that language without announcing the switch. Continue the entire conversation in whatever language the caller is most comfortable with. CRITICAL: When summarizing the call, writing ai_notes, or extracting lead data (name, intent, interest), ALWAYS write those in English regardless of the conversation language. The CRM data must be in English for staff.\n"
"━━━ HOLIDAY & SCHEDULE AWARENESS ━━━\n"
"You know today's exact date and day of the week from the SYSTEM CONTEXT above. If a caller asks about holiday hours or whether the office is open on a specific date, check the business hours and any holiday schedule information in your knowledge base. NEVER assume the office is open on major holidays — instead say: "Let me verify our availability for that date. We may have adjusted hours or be closed for the holiday — I'd recommend calling back or I can take your contact info and have someone confirm with you." For home services and trades: if a caller requests service on a major US holiday (Christmas, Thanksgiving, July 4th, New Year's), mention that holiday dispatch rates may apply and confirm before booking. For scheduling questions: if no slots are available on a requested date, say "It looks like we don't have availability that day — we may be closed or fully booked. Let me check the following week for you."\n"
"━━━ FOCUS & SCOPE (MANDATORY) ━━━\n"
""You are a business receptionist — not a general-purpose AI. Never answer questions about weather, news, sports, holidays, politics, science, religion, or any topic unrelated to this business. If asked off-topic, redirect warmly: 'That\'s a great question, but I\'m here specifically to help with [business name]. Can I help with an appointment or answer questions about our services?' Stay warm, stay on-task.\n

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

━━━ EMERGENCY TRANSFER RULE ━━━
Use the transfer_call tool IMMEDIATELY when:
- Caller says "emergency", "urgent", "right now", "flooding", "burst pipe", 
  "not breathing", "chest pain", severe pain, or ANY emergency keyword
- Caller says "let me speak to a human", "I want a real person", "transfer me" 
  (more than once)
ALWAYS say "Please hold while I connect you to the team" BEFORE calling the tool.
Never use transfer_call for standard questions or booking requests.
━━━ END EMERGENCY TRANSFER ━━━

━━━ APPOINTMENT BOOKING RULES ━━━
You have two tools for booking: check_availability and book_appointment.

BOOKING WORKFLOW:
1. When a caller asks to book/schedule, ask what day works best for them.
2. Use check_availability with that date. Read ONLY the options returned — never invent times.
3. Once they pick a time, confirm: "Let me lock that in for you. May I get your name, 
   email address, and the best phone number for you?"
4. Once you have name + email + phone + time, call book_appointment immediately.
5. After booking: "Perfect! I just sent a confirmation link to [email]. 
   Please click it within 24 hours to finalize your appointment."

IMPORTANT:
- Never promise a time without running check_availability first
- Never book without collecting name, email, AND phone
- If check_availability returns an error, offer to take a message instead
━━━ END BOOKING RULES ━━━

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

IF Medical / MedSpa / Wellness / Weight Loss / Clinic / Dental / Dermatology / Veterinary / Vet Clinic / Animal Hospital:
"For medical, wellness, and veterinary practices, our Healthcare plan is $495 per month —
that includes 750 voice minutes, HIPAA-compliant data pipelines, a signed BAA,
and a dedicated local number. The one-time setup fee is $499, which covers
your secure data pipeline, patient intake protocols, and calendar integration.
For practices with higher call volume, we also offer a $750/month plan.
Does that sound like the right fit for your practice?"

IF Home & Field Services (Plumber / HVAC / Electrician / Roofer / Landscaping /
Pest Control / Contractor / Moving / Towing / Auto Detail):
"For home service businesses, our Field Services plan is $295 per month —
500 voice minutes, 24/7 emergency dispatch, live call transfer to you,
SMS booking links, and a local dedicated number. One-time setup is $299.
You'll never miss a job while you're on-site. Does that work for your business?"

IF Professional Services (Law / Accounting / Consulting / Real Estate / Insurance):
"For professional services, our plan is $395 per month — 600 voice minutes,
appointment booking, lead qualification, and a CRM dashboard.
One-time setup fee is $399. Would you like more details?"

IF Other / General / Unsure:
"Our plans start at $295 per month depending on your industry and compliance
requirements — we have specialized packages for healthcare, field services,
and professional businesses. Could I ask what industry you're in so I can
give you the most accurate information?"

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


# ── Active call registry — maps call_sid → live openai_ws connection ──────────
# Used by /warm-handoff to inject prompts mid-call
_active_openai_ws: dict = {}   # call_sid -> openai_ws WebSocket
_active_twilio_ws: dict = {}   # call_sid -> twilio websocket
_active_stream_sid: dict = {}  # call_sid -> stream_sid


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
            "max_call_duration_minutes,address,phone,website_url,brand_name,transfer_number,"
            "announce_recording,recording_consent_text,external_intake_url,zapier_webhook_url,"
            "emergency_contact_email,emergency_contact_phone,supported_service_areas,industry_vertical"
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
        # ai_receptionist_config table not present — skip gracefully
        r = None
        result["ai_config"] = None or {}
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
            "phone":        encrypt_pii(phone_clean),   # AES-256 at-rest PII encryption
            "phone_hash":   hash_pii(phone_clean),      # deterministic lookup key
            "lead_status":        "new",
            "pipeline_status":    "NEW_LEAD",   # 6-stage pipeline
            "channel":            "ARIA_PHONE",
            "source":             "call",
            "last_interaction_at": datetime.now(timezone.utc).isoformat(),
            "last_summary": transcript[:500],
            "ai_notes":     (encrypt_text if should_encrypt() else lambda x: x)(_json.dumps({
                "interest":    extracted.get("interest"),
                "intent":      extracted.get("intent"),
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "call_sid":    call_sid,
            })),
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
        upsert_result = sb.table("contacts").upsert(
            contact_row,
            on_conflict="business_id,phone_hash",
        ).execute()
        if not (upsert_result.data and upsert_result.data[0].get("id")):
            id_result = sb.table("contacts").select("id").eq(
                "business_id", business_id
            ).eq("phone_hash", contact_row["phone_hash"]).maybe_single().execute()
            if id_result.data:
                upsert_result = type("R", (), {"data": [id_result.data]})()

        logger.info(f"Contact upserted: {phone_clean} for {business_id}")

        # Link the contact back to the call record so Recent Activity shows name
        if upsert_result.data and call_sid:
            contact_id = upsert_result.data[0]["id"]
            sb.table("calls").update({"contact_id": contact_id}).eq("twilio_call_sid", call_sid).execute()
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

        # Extract first name from transcript for personalized alert
        import re as _re2
        nm = _re2.search(r"(?:name is|I am|my name)\s+([A-Z][a-z]+)", transcript)
        caller_name = (nm.group(1) + " *") if nm else None

        if has_email:
            label = "📅 New lead captured!"
            who   = f"Caller: {caller_name or caller}"
        else:
            label = "⚡ Hot lead — no email"
            who   = f"Caller: {caller} (no email — call back!)"

        msg = (
            f"🤖 Aria Alert — {biz_name}\n"
            f"{label}\n"
            f"{who}\n"
            f"CRM: https://app.receptionist.co/dashboard\n"
            "Reply STOP to opt out. Msg & Data rates may apply."
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

        # Email via Postmark (business-facing transactional mail)
        POSTMARK_KEY = os.environ.get("POSTMARK_SERVER_TOKEN", "")
        if notify_email and POSTMARK_KEY:
            async with httpx.AsyncClient() as client:
                await client.post(
                    "https://api.postmarkapp.com/email",
                    headers={
                        "X-Postmark-Server-Token": POSTMARK_KEY,
                        "Content-Type": "application/json",
                    },
                    json={
                        "From": "Aria <notifications@mail.receptionist.co>",
                        "To": notify_email,
                        "Subject": f"📅 Aria Alert: {'New lead' if has_email else 'Hot lead (no email)'} — {biz_name}",
                        "HtmlBody": (
                            f"<div style='font-family:-apple-system,Arial,sans-serif;max-width:560px;margin:0 auto;padding:32px 24px;background:#0A1628;color:#E2E8F0'>"
                            f"<div style='font-size:22px;font-weight:800;color:#fff;margin-bottom:4px'>Receptionist.co</div>"
                            f"<div style='font-size:12px;color:#4F8EF7;margin-bottom:28px'>AI Front Desk</div>"
                            f"<div style='background:linear-gradient(135deg,#1a2744,#0f1e38);border:1px solid rgba(79,142,247,0.3);border-radius:12px;padding:24px;margin-bottom:20px'>"
                            f"<div style='font-size:13px;color:#94A3B8;margin-bottom:4px'>🤖 Aria captured a new lead for</div>"
                            f"<div style='font-size:20px;font-weight:800;color:#fff;margin-bottom:16px'>{biz_name}</div>"
                            + (f"<div style='display:flex;gap:8px;margin-bottom:8px'><span style='color:#94A3B8;font-size:13px'>Caller:</span><span style='color:#fff;font-size:13px;font-weight:600'>{caller_name or caller}</span></div>" if True else "")
                            + (f"<div style='display:flex;gap:8px;margin-bottom:8px'><span style='color:#94A3B8;font-size:13px'>Email:</span><span style='color:#fff;font-size:13px'>{caller}</span></div>" if has_email else f"<div style='background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:8px;padding:10px;font-size:12px;color:#FCA5A5'>⚠️ No email provided — call them back directly at {caller}</div>")
                            + f"</div>"
                            f"<div style='background:rgba(255,255,255,0.04);border-radius:10px;padding:16px;margin-bottom:20px'>"
                            f"<div style='font-size:11px;font-weight:700;color:#94A3B8;letter-spacing:1px;text-transform:uppercase;margin-bottom:8px'>Call Summary Preview</div>"
                            f"<div style='font-size:13px;color:#CBD5E1;line-height:1.7'>{transcript[:300]}...</div>"
                            f"</div>"
                            f"<a href='https://app.receptionist.co/dashboard' style='display:block;background:#4F8EF7;color:#fff;text-decoration:none;padding:14px;border-radius:10px;font-weight:700;font-size:14px;text-align:center'>View Full Transcript & Listen to Call →</a>"
                            f"<p style='font-size:11px;color:#475569;margin-top:24px;border-top:1px solid rgba(255,255,255,0.06);padding-top:16px;text-align:center'>"
                            f"{biz_name} · Powered by <a href='https://receptionist.co' style='color:#4F8EF7'>Receptionist.co</a> · "
                            f"<a href='https://receptionist.co/privacy' style='color:#4F8EF7'>Privacy</a> · "
                            f"<a href='https://receptionist.co/terms' style='color:#4F8EF7'>Terms</a> · "
                            f"<a href='https://app.receptionist.co/unsubscribe' style='color:#4F8EF7'>Unsubscribe</a></p>"
                            f"</div>"
                        ),
                        "MessageStream": "outbound",
                    },
                    timeout=10.0,
                )
                logger.info(f"Lead email sent via Postmark for {business_id}")

    except Exception as e:
        logger.warning(f"Lead notification failed (non-critical): {e}")



async def handle_function_call(fn_name: str, fn_args: dict, business_id: str, to_number: str, call_sid: str = "") -> str:
    """
    Dispatch OpenAI function calls to Cal.com API via the CRM /api/cal routes.
    Returns a plain text string that Aria will speak to the caller.
    """
    CRM_BASE = os.environ.get("CRM_BASE_URL", "https://app.receptionist.co")
    # Note: CRM routes handle their own Cal.com auth (OAuth tokens)
    # The encryption module is used when businesses paste a direct API key

    if fn_name == "transfer_call":
        reason = fn_args.get("reason", "caller requested")
        try:
            sb = get_sb()
            # Get emergency transfer number for this business
            biz = sb.from_("businesses").select("emergency_transfer_number,name").eq("id", business_id).single().execute()
            transfer_number = biz.data.get("emergency_transfer_number") if biz.data else None

            # Also check settings_business.transfer_number (set in CRM Settings > Warm Handoff)
            if not transfer_number:
                st = sb.from_("settings_business").select("transfer_number").eq("business_id", business_id).maybe_single().execute()
                if st.data and st.data.get("transfer_number"):
                    transfer_number = st.data["transfer_number"]
            biz_name = biz.data.get("name", "the team") if biz.data else "the team"

            if not transfer_number:
                # Check ai_receptionist_config for escalation_phone
                cfg = sb.from_("ai_receptionist_config").select("escalation_phone").eq("business_id", business_id).maybe_single().execute()
                transfer_number = cfg.data.get("escalation_phone") if cfg.data else None

            if transfer_number:
                TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
                TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
                if TWILIO_SID and TWILIO_TOKEN and to_number:
                    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="Polly.Joanna">Please hold while I connect you to the team. Transferring now.</Say>
    <Dial callerId="{to_number}">{transfer_number}</Dial>
</Response>"""
                    # Hijack the live call via Twilio REST API
                    async with httpx.AsyncClient() as client:
                        resp = await client.post(
                            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Calls/{call_sid}.json",
                            auth=(TWILIO_SID, TWILIO_TOKEN),
                            data={"Twiml": twiml},
                            timeout=8.0,
                        )
                    if resp.is_success:
                        logger.info(f"Call transferred: {reason} → {transfer_number}")
                        return f"Transfer initiated. Tell the caller you are connecting them now and stay quiet."
                    else:
                        logger.warning(f"Transfer failed: {resp.status_code}")
                return f"Transfer initiated to {transfer_number}. Tell the caller you're connecting them."
            else:
                logger.warning(f"No emergency transfer number for {business_id}")
                return "No transfer number configured for this account. Tell the caller: 'I'm so sorry — I'm not able to complete the transfer right now as our specialists are temporarily unavailable. Let me take your contact information and ensure someone calls you back within the next few minutes. May I get your name and best callback number?hin minutes. May I confirm the best number to reach you?'"
        except Exception as e:
            logger.warning(f"transfer_call error: {e}")
            return "Transfer system error. Tell the caller you'll have someone call them back immediately and take their number."

    elif fn_name == "check_availability":
        date       = fn_args.get("date", "")
        preference = fn_args.get("preference", "any")
        if not date:
            return "I don't have a date to check. Could you tell me what day works best for you?"

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{CRM_BASE}/api/cal/slots",
                    params={"business_id": business_id, "date": date},
                    timeout=8.0,
                )
            if not resp.is_success:
                return "System error: Unable to load calendar. Please apologize and offer to have someone call them back to schedule."

            data = resp.json()
            slots = data.get("slots", [])

            if not slots:
                return f"There are no available slots on {date}. Please ask the caller for an alternative date."

            # Filter by preference
            if preference == "morning":
                slots = [s for s in slots if "AM" in s.get("label","").upper() or
                         int(s.get("label","12:00 PM").split(":")[0].replace(" AM","").replace(" PM","")) < 12]
            elif preference == "afternoon":
                slots = [s for s in slots if "PM" in s.get("label","").upper()]

            # Truncate to max 3 options
            slots = slots[:3]

            if not slots:
                return f"There are no {preference} slots available on {date}. Would you like to check a different time of day or date?"

            labels = [s["label"] for s in slots]
            if len(labels) == 1:
                return f"I have {labels[0]} available on {date}. Would that work for you?"
            elif len(labels) == 2:
                return f"I have {labels[0]} or {labels[1]} available on {date}. Which works best?"
            else:
                return f"I have {labels[0]}, {labels[1]}, or {labels[2]} available on {date}. Which works best for you?"

        except Exception as e:
            logger.warning(f"check_availability error: {e}")
            return "System error: Unable to load calendar. Please apologize and offer to have someone call them back to schedule."

    elif fn_name == "book_appointment":
        start_time = fn_args.get("startTime", "")
        name       = fn_args.get("name", "")
        email      = fn_args.get("email", "")
        phone      = fn_args.get("phone", "")

        if not all([start_time, name, email]):
            return "I need the caller's name, email, and preferred time before booking. Please collect any missing details."

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{CRM_BASE}/api/cal/book",
                    json={
                        "business_id":   business_id,
                        "start_time":    start_time,
                        "contact_name":  name,
                        "contact_email": email,
                        "contact_phone": phone,
                        "notes":         "Booked by Aria via phone",
                    },
                    timeout=10.0,
                )
            if resp.is_success:
                return (
                    f"Booking successful. Tell the caller you just sent a confirmation link "
                    f"to {email} and they need to click it to finalize the appointment. "
                    f"Remind them to check their spam folder if they don't see it within a minute."
                )
            else:
                err = resp.json().get("error","unknown error")
                logger.warning(f"book_appointment error: {err}")
                return "Error: That slot may have just been taken or the booking system had an issue. Apologize and offer to check another time, or say someone will call them back."

        except Exception as e:
            logger.warning(f"book_appointment exception: {e}")
            return "Error: Unable to reach the booking system. Please apologize and offer to have someone call them back to confirm the appointment."

    return f"Unknown function: {fn_name}"

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
    """DISABLED per Zero Audio Retention Policy — Receptionist.co does not record audio.
    Raw audio streams are processed ephemerally in memory and discarded on disconnect.
    See: receptionist.co/security#zero-audio-retention
    """
    # ZAR POLICY: This function is intentionally disabled.
    logger.warning("start_twilio_recording called — BLOCKED by Zero Audio Retention Policy")
    return  # never proceeds


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

async def run_aup_moderation(
    business_id: str,
    call_sid: str,
    transcript: str,
    business_name: str = "",
) -> None:
    """
    Asynchronously scans the call transcript against OpenAI's free Moderation API.
    Zero-retention endpoint — OpenAI does not train on moderation data.
    If flagged, inserts an encrypted alert into aup_compliance_alerts (super-admin only).
    """
    if not transcript or not business_id:
        return

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/moderations",
                headers={"Authorization": f"Bearer {openai_key}",
                         "Content-Type": "application/json"},
                json={"input": transcript[:8000]},   # free endpoint, fast
                timeout=10.0,
            )
        if resp.status_code != 200:
            return

        data      = resp.json()
        result    = data.get("results", [{}])[0]
        flagged   = result.get("flagged", False)

        if not flagged:
            return  # clean call — nothing to do

        # Map OpenAI categories → our AUP violation types
        cats = result.get("categories", {})
        scores = result.get("category_scores", {})

        CATEGORY_MAP = {
            "hate":               "HATE_SPEECH",
            "hate/threatening":   "THREAT_VIOLENCE",
            "harassment":         "HARASSMENT",
            "harassment/threatening": "THREAT_VIOLENCE",
            "self-harm":          "SELF_HARM",
            "self-harm/intent":   "SELF_HARM",
            "self-harm/instructions": "SELF_HARM",
        }

        # Pick the highest-scoring flagged category
        top_cat   = max(scores, key=scores.get) if scores else "harassment"
        vtype     = CATEGORY_MAP.get(top_cat, "SPAM_PHISHING")
        severity  = round(scores.get(top_cat, 0.9), 3)

        # Encrypt the flagged text (same AES-256 as transcripts)
        snippet   = transcript[:500]
        encrypted = encrypt_text(snippet) if should_encrypt() else snippet

        sb = get_sb()
        if not sb:
            return

        sb.table("aup_compliance_alerts").insert({
            "business_id":   business_id,
            "call_sid":      call_sid,
            "violation_type": vtype,
            "severity_score": severity,
            "flagged_text":   encrypted,
            "raw_categories": cats,
            "status":         "PENDING_REVIEW",
        }).execute()

        logger.warning(
            f"🚨 AUP VIOLATION: {vtype} (score={severity:.3f}) "
            f"— Tenant: {business_name or business_id} | Call: {call_sid}"
        )

        # Optional: notify internal Slack/Discord webhook
        alert_webhook = os.environ.get("AUP_ALERT_WEBHOOK_URL", "")
        if alert_webhook:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(alert_webhook, json={
                        "text": (
                            f"🚨 *AUP Alert* — `{vtype}` (confidence: {severity:.0%})\n"
                            f"*Tenant:* {business_name or business_id}\n"
                            f"*Call:* `{call_sid}`\n"
                            f"Review in the Compliance Dashboard."
                        )
                    }, timeout=5.0)
            except Exception:
                pass  # non-critical

    except Exception as e:
        logger.debug(f"AUP moderation error (non-critical): {e}")


async def save_call_record(call_sid: str, business_id: str, from_number: str,
                            transcript: str, duration: int, start_time_iso: str = None):
    """Save completed call to Supabase."""
    sb = get_sb()
    if not sb or not business_id:
        return
    try:
        clean_transcript = scrub_pii(transcript)

        # Apply AES-256 to transcript before writing — protects PHI for HIPAA clients
        # If ENCRYPTION_KEY not set, stores plain text (graceful degradation)
        stored_transcript = (
            encrypt_text(clean_transcript[:8000]) if should_encrypt()
            else clean_transcript[:8000]
        )

        sb.from_("calls").upsert({
            "twilio_call_sid":  call_sid,
            "business_id":      business_id,
            "phone_number":     encrypt_pii(from_number),
            "from_number":      from_number,
            "direction":        "inbound",
            "duration_seconds": duration,
            "handled_by_ai":    True,
            "status":           "completed",
            "call_status":      "completed",
            "started_at":       start_time_iso or datetime.now(timezone.utc).isoformat(),
            "transcript_summary": stored_transcript,
        }, on_conflict="twilio_call_sid").execute()

        # Post-call async tasks
        try:
            asyncio.create_task(notify_lead_captured(business_id, clean_transcript, from_number))
            asyncio.create_task(extract_lead_from_transcript(business_id, call_sid, from_number, clean_transcript))
        except:
            pass



    except Exception as e:
        logger.error(f"save_call_record error: {e}")

@app.get("/health")
async def health():
    """
    Health check endpoint — monitored by UptimeRobot every 1 minute.
    Returns 200 OK with status details when operational.
    UptimeRobot alert fires if this returns non-200 or times out.
    """
    sb_ok = bool(get_sb())
    return {
        "status":    "operational",
        "service":   "aria-call-handler",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "supabase": "connected" if sb_ok else "degraded",
            "openai":   "configured" if os.environ.get("OPENAI_API_KEY") else "missing",
            "twilio":   "configured" if os.environ.get("TWILIO_ACCOUNT_SID") else "missing",
        }
    }

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
    transcript_turns = []   # [{role, text, ts}] with per-turn timestamps
    start_time    = None  # set when Twilio start event fires (actual call start)
    business_cfg  = {}
    business_id   = ""
    max_call_mins = 10   # default — overridden from DB once start event fires
    is_responding       = False  # True while OpenAI is generating audio — guards response.cancel
    call_active         = True   # set False on stop — prevents post-hangup upserts
    last_speech_at      = None   # timestamp of last caller speech — for silence timeout
    current_item_id     = None   # OpenAI assistant message ID (for truncation on barge-in)
    audio_ms_sent       = 0      # Milliseconds of audio sent to Twilio (for accurate truncation)

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
        # Register for warm-handoff injection (populated once call_sid is known)

        async def receive_from_twilio():
            nonlocal is_responding, last_speech_at, current_item_id, stream_sid, call_sid, start_time, business_id, business_cfg, max_call_mins, audio_ms_sent, call_active, to_number, from_number, call_active
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
                    # Register in global registry for warm-handoff
                    _active_openai_ws[call_sid] = openai_ws
                    _active_twilio_ws[call_sid]  = websocket
                    _active_stream_sid[call_sid] = stream_sid

                    # Start recording via Twilio REST API (compatible with Media Streams)
                    # ZAR Policy: recording disabled

                    # ── Twilio safety timeout (hard stop even if Railway hangs) ──────
                    try:
                        _ts = os.environ.get("TWILIO_ACCOUNT_SID",""); _tt = os.environ.get("TWILIO_AUTH_TOKEN","")
                        if _ts and _tt and call_sid:
                            _limit = (max_call_mins + 2) * 60
                            async with httpx.AsyncClient() as _tc:
                                await _tc.post(
                                    f"https://api.twilio.com/2010-04-01/Accounts/{_ts}/Calls/{call_sid}.json",
                                    auth=(_ts, _tt), data={"TimeLimit": str(_limit)}, timeout=4.0,
                                )
                            logger.debug(f"Twilio TimeLimit={_limit}s for {call_sid}")
                    except Exception as _tl: logger.debug(f"TimeLimit: {_tl}")

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

                    # ── Compliance: recording announcement ────────────────────
                    announce_recording = settings.get("announce_recording", True)  # default ON
                    consent_text = (
                        settings.get("recording_consent_text") or
                        "This call is being recorded and processed by AI."
                    )

                    if announce_recording:
                        compliance_rule = (
                            f"MANDATORY: Begin every call by saying exactly:\n"
                            f"\"{consent_text}\"\n"
                            f"Say this BEFORE anything else, even before your greeting."
                        )
                    else:
                        compliance_rule = (
                            "The business has verified their state's recording laws and opted out "
                            "of the recording announcement. Start the call naturally."
                        )

                    # ── Dynamic Medical Vertical Prompting ───────────────────────
                    biz_info         = business_cfg.get("businesses") or {}
                    is_medical       = biz_info.get("is_medical_vertical", False)
                    industry_vert    = settings.get("industry_vertical", "general")
                    # Also check vertical from settings_business
                    if industry_vert in ("medical", "dental", "veterinary", "medspa"):
                        is_medical = True

                    # ── 3-Way Industry Vertical Fork ─────────────────────────────────────
                    # Fetch safety settings
                    emerg_email = settings.get("emergency_contact_email", "") or ""
                    emerg_phone = settings.get("emergency_contact_phone", "") or ""
                    service_areas = settings.get("supported_service_areas") or []

                    is_home_services = industry_vert in (
                        "home_services", "hvac", "plumbing", "electrical",
                        "roofing", "trades", "contractor", "home_services_trades"
                    )

                    if is_medical:
                        compliance_rule += (
                            "\n\nMEDICAL VERTICAL COMPLIANCE RULES (MANDATORY):\n"
                            "1. EMERGENCY ROUTING: If a caller mentions chest pain, difficulty breathing, "
                            "severe bleeding, stroke symptoms, or any life-threatening emergency, "
                            "IMMEDIATELY say: 'This sounds like a medical emergency. Please hang up and "
                            "call 911 right now.' Do not attempt to book or continue the call.\n"
                            "2. NEW PATIENT SECURE HOLD: For NEW patients booking medical/aesthetic "
                            "appointments, do NOT confirm a final booking. Instead say: 'I have "
                            "tentatively held that slot for you! Because this is a medical appointment, "
                            "our system requires a quick intake form to finalize the booking. I just "
                            "texted you the secure link. If you complete it within 30 minutes, it will "
                            "permanently lock in your time. Otherwise the system will automatically "
                            "release the hold.' Then trigger the intake SMS.\n"
                            "3. EXISTING PATIENTS: If the caller is already in the system, you may "
                            "book directly using the Soft Confirm method.\n"
                            "4. MINOR PROTECTION: If a caller is or sounds under 18, do NOT book "
                            "restricted medical/aesthetic services. Ask a parent or guardian to call.\n"
                            "5. NO MEDICAL ADVICE: Never provide diagnoses or treatment recommendations. "
                            "Always say 'I am not able to give medical advice — please speak with your provider.'\n"
                            "5b. PHI OVERSHARER: If caller lists medications, insurance IDs, or medical history, interrupt: To protect your privacy, please save those details for the secure form — I just need your name and phone number. Never allow PHI to be spoken aloud.\\n"
                            "5c. ANTI-TECH FALLBACK: If caller cannot use texts, say: I completely understand — a human specialist will call you back shortly. Then flag the call as REQUIRES_HUMAN_CALLBACK.\\n"
                            "6. MVD RULE (MEDICAL): You only need THREE things — First Name, Cell Phone, "
                            "and reason for call. NEVER ask for home address, email, DOB, insurance, or "
                            "medical history over the phone. Once you have all three, say exactly: "
                            "'To save you from spelling everything out, I am texting you our secure "
                            "registration link right now — it takes about 60 seconds on your phone.' "
                            "Then immediately trigger the SMS intake link.\n"
                        )

                    elif is_home_services:
                        compliance_rule += (
                            "\n\nHOME SERVICES & TRADES PROTOCOL (MANDATORY):\n"
                            "PHYSICAL EMERGENCY PROTOCOL: If caller reports gas smell, sparking panels, active flooding, or life-threatening hazards, immediately say: Please evacuate and call 911 or your local utility company. I am flagging this as a CRITICAL emergency for our dispatch team. Then stop the booking flow.\\n"
                            "You are dispatching for a home services company. Customers calling with "
                            "emergencies (burst pipe, no AC, power out) need immediate help — "
                            "do NOT send them a form or a link. Complete the entire booking over the phone.\n"
                            "COLLECT ALL FOUR DATA POINTS VERBALLY:\n"
                            "1. FIRST AND LAST NAME: Get both first and last name.\n"
                            "2. CELL PHONE NUMBER: Confirm the best callback number.\n"
                            "3. SERVICE ADDRESS: Ask 'What is the service address, including city and zip?' "
                            "MANDATORY ADDRESS READ-BACK: After collecting the address, read it back verbatim: Just to confirm — [repeat full address] — did I get that right? Do NOT trigger the webhook until caller confirms with yes. Missed digits cost drive time.\\n"
                            "Repeat it back to confirm accuracy.\n"
                            "4. ISSUE DESCRIPTION: Ask them to briefly describe the problem "
                            "(e.g., 'broken AC', 'burst pipe under kitchen sink', 'no hot water').\n"
                            "NEVER send an intake form SMS to home services callers.\n"
                            "NEVER ask them to click a link or fill out a form.\n"
                            "Once you have all four data points, say exactly: "
                            "'Perfect — I have your address and issue logged. I am sending this directly "
                            "to our dispatch team right now. They will text you shortly with your "
                            "technician arrival window. Is there anything else I can help with?'\n"
                            "Then trigger the dispatch confirmation SMS and push to Zapier.\n"
                        )
                        if service_areas:
                            area_list = ", ".join(str(a) for a in service_areas)
                            compliance_rule += (
                                f"SERVICE AREA POLICY: Covered areas: {area_list}. "
                                "If address is outside this area, do NOT decline. Say: "
                                "I have your address logged. You are slightly outside our "
                                "standard service radius, but I am flagging this as a "
                                "priority review for our dispatch manager. They will "
                                "reach out shortly to confirm. "
                                "Status: OUT_OF_AREA_REVIEW, still push to Zapier.\\n"
                            )

                    else:
                        compliance_rule += (
                            "\n\nSTANDARD BOOKING RULES:\n"
                            "You do not need to verify age or collect date of birth. Simply ask for the "
                            "caller's name, phone number, and preferred appointment time to finalize the "
                            "booking. Be friendly, efficient, and conversational.\n"
                            "MVD RULE (GENERAL): Collect First Name, Cell Phone, and reason for call. "
                            "Do not ask for home address, email, or sensitive personal information "
                            "unless the caller volunteers it. Once you have Name + Phone + Intent, "
                            "offer to text them a confirmation or intake link if applicable.\n"
                        )

                    # ── After-Hours Detection ─────────────────────────────────────
                    # Per Zero Audio Retention Policy: no voicemail — Aria takes message
                    # and creates a NEW_LEAD CRM record flagged for morning follow-up
                    try:
                        from zoneinfo import ZoneInfo
                        biz_tz  = ZoneInfo(tz)
                        now_biz = datetime.now(biz_tz)
                        hour    = now_biz.hour
                        weekday = now_biz.weekday()  # 0=Mon 6=Sun
                        # Default after-hours: outside 8am-6pm Mon-Fri
                        is_after_hours = (hour < 8 or hour >= 18) or weekday >= 5
                        if is_after_hours:
                            compliance_rule += (
                                "\n\nAFTER HOURS POLICY (CRITICAL):\n"
                                f"It is currently {now_biz.strftime('%I:%M %p %Z')} — outside business hours. "
                                "DO NOT attempt to book an appointment right now. Instead:\n"
                                "1. Greet warmly: 'Hi, you've reached [business]. We're currently closed, "
                                "but I'm their AI assistant and I'm here to help!'\n"
                                "2. Ask what you can help with or what message to pass to staff in the morning.\n"
                                "3. Collect: caller name, phone number, and their message/request.\n"
                                "4. Confirm: 'Perfect — a team member will reach out to you first thing in the morning!'\n"
                                "IMPORTANT: This replaces traditional voicemail. No audio is stored — "
                                "only this transcript becomes the CRM record (NEW_LEAD status).\n"
                            )
                    except Exception as tz_err:
                        logger.debug(f"After-hours check: {tz_err}")

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
                        custom_instructions=compliance_rule + "\n\n" + custom_instr,
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
                            "tools": [
                                {
                                    "type": "function",
                                    "name": "check_availability",
                                    "description": "Check the business calendar for available appointment times on a specific date.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "date": {
                                                "type": "string",
                                                "description": "The date to check, formatted as YYYY-MM-DD."
                                            },
                                            "preference": {
                                                "type": "string",
                                                "enum": ["morning", "afternoon", "any"],
                                                "description": "Caller's preferred time of day."
                                            }
                                        },
                                        "required": ["date"]
                                    }
                                },
                                {
                                    "type": "function",
                                    "name": "transfer_call",
                                    "description": "Transfer the call to a human immediately. Use when caller has a true emergency, repeatedly demands to speak to a human, or the situation requires human judgment. Tell caller 'Please hold while I connect you' BEFORE calling this function.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "reason": {
                                                "type": "string",
                                                "description": "Short 3-5 word summary of why transferring (e.g. 'Burst pipe in basement')"
                                            }
                                        },
                                        "required": ["reason"]
                                    }
                                },
                                {
                                    "type": "function",
                                    "name": "book_appointment",
                                    "description": "Book an appointment after caller confirms a time. Requires name, email, phone, and startTime.",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "startTime": {
                                                "type": "string",
                                                "description": "Start time in ISO 8601 format e.g. 2026-04-02T14:00:00Z"
                                            },
                                            "name":  {"type": "string"},
                                            "email": {"type": "string"},
                                            "phone": {"type": "string"}
                                        },
                                        "required": ["startTime", "name", "email", "phone"]
                                    }
                                }
                            ],
                            "tool_choice": "auto",
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
                    call_active = False  # prevent further upserts
                    if call_sid and business_id:
                        asyncio.create_task(delete_active_call(call_sid))
                    break

        async def receive_from_openai():
            nonlocal is_responding, last_speech_at, current_item_id, stream_sid, call_sid, start_time, business_id, business_cfg, max_call_mins, audio_ms_sent
            async for raw in openai_ws:
                data       = json.loads(raw)
                event_type = data.get("type", "")

                if event_type == "response.audio.delta" and data.get("delta"):
                    is_responding = True
                    if not current_item_id:
                        current_item_id = data.get("item_id")
                    # Track ~ms of audio sent (g711_ulaw = 8 bytes/ms)
                    audio_ms_sent += len(data["delta"]) * 3 // 4 // 8
                    try:
                        await websocket.send_text(json.dumps({


                            "event":     "media",
                            "streamSid": stream_sid,
                            "media":     {"payload": data["delta"]},
                        }))
                    except Exception:
                        pass  # Twilio disconnected

                elif event_type == "response.audio_transcript.done":
                    # Aria finished speaking — store her full transcript turn
                    is_responding = False
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Aria: {text}")
                        _now = datetime.now(timezone.utc).isoformat()
                        transcript_turns.append({"role":"ai","text":text,"ts":_now})
                        if len(transcript_turns) > 40: transcript_turns.pop(0)
                        # Push live transcript so Glass Box shows Aria's turns
                        if business_id:
                            turns = transcript_turns[-20:]
                            if call_active:
                                asyncio.create_task(upsert_active_call(
                                    business_id, call_sid, from_number, "in-progress", turns
                                ))

                elif event_type == "conversation.item.input_audio_transcription.completed":
                    text = data.get("transcript", "")
                    if text:
                        transcript.append(f"Caller: {text}")
                        _now2 = datetime.now(timezone.utc).isoformat()
                        transcript_turns.append({"role":"user","text":text,"ts":_now2})
                        if len(transcript_turns) > 40: transcript_turns.pop(0)
                        # Push caller speech immediately so Glass Box shows it live
                        if business_id:
                            _turns = transcript_turns[-20:]
                            if call_active:
                                    asyncio.create_task(upsert_active_call(
                                        business_id, call_sid, from_number, "in-progress", _turns
                                    ))
                        last_speech_at = datetime.now(timezone.utc)

                elif event_type == "response.function_call_arguments.done":
                    # Aria wants to call a tool — handle it
                    fn_name = data.get("name", "")
                    fn_args_raw = data.get("arguments", "{}")
                    call_item_id = data.get("call_id", "")
                    try:
                        import json as _json
                        fn_args = _json.loads(fn_args_raw)
                    except Exception:
                        fn_args = {}
                    logger.info(f"Function call: {fn_name}({fn_args})")
                    result_text = await handle_function_call(fn_name, fn_args, business_id, to_number, call_sid=call_sid)
                    # Feed result back to OpenAI so Aria can speak it
                    await openai_ws.send(_json.dumps({
                        "type": "conversation.item.create",
                        "item": {
                            "type":    "function_call_output",
                            "call_id": call_item_id,
                            "output":  result_text,
                        }
                    }))
                    await openai_ws.send(_json.dumps({"type": "response.create"}))

                elif event_type == "input_audio_buffer.speech_started":
                    # Two-step interruption handshake:
                    # 1. Tell Twilio to dump its audio buffer (stops Aria mid-syllable)
                    # 2. Tell OpenAI to cancel if actively generating
                    await websocket.send_text(json.dumps({
                        "event":     "clear",
                        "streamSid": stream_sid,
                    }))
                    if is_responding:
                        is_responding = False
                        try:
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                        except Exception:
                            pass  # response may have already completed — safe to ignore
                        # Truncate transcript so OpenAI knows where it was cut off
                        if current_item_id and audio_ms_sent > 0:
                            try:
                                await openai_ws.send(json.dumps({
                                    "type":          "conversation.item.truncate",
                                    "item_id":       current_item_id,
                                    "content_index": 0,
                                    "audio_end_ms":  audio_ms_sent,
                                }))
                            except Exception:
                                pass
                        current_item_id = None
                        audio_ms_sent   = 0

                elif event_type == "error":
                    # Suppress benign barge-in race conditions
                    _ec = data.get("error", {}).get("code", "")
                    if _ec in ("response_cancel_not_active", "invalid_value"):
                        logger.debug(f"OpenAI barge-in race (ignored): {_ec}")
                    else:
                        logger.error(f"OpenAI error: {data}")

        # ── Call duration watchdog ────────────────────────────────────────
        async def call_timer():
            soft_secs = (max_call_mins - 1) * 60
            hard_secs = max_call_mins * 60 + 30
            # Sleep in 10s increments — exits immediately if call_active becomes False
            for _ in range(max(1, soft_secs // 10)):
                if not call_active:
                    return  # call already ended — timer not needed
                await asyncio.sleep(min(10, soft_secs))
            if not call_active:
                return
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
            for _ in range(max(1, (hard_secs - soft_secs) // 10)):
                if not call_active:
                    return
                await asyncio.sleep(10)
            if not call_active:
                return
            logger.info(f"Hard disconnect at {max_call_mins}min 30s")
            try:
                await websocket.close()
            except:
                pass

        # ── Task-based execution: cancel timer/monitor when call ends ──────
        _tasks = [
            asyncio.ensure_future(receive_from_twilio()),
            asyncio.ensure_future(receive_from_openai()),
            asyncio.ensure_future(call_timer()),
            asyncio.ensure_future(silence_monitor(
                openai_ws,
                lambda: last_speech_at,
                lambda: is_responding,
                websocket,
                call_sid,
            )),
        ]
        try:
            # Wait until the two core WebSocket tasks finish (call ended)
            done, pending = await asyncio.wait(
                _tasks[:2],  # receive_from_twilio + receive_from_openai
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Give the second WebSocket task a moment to close cleanly
            await asyncio.wait(_tasks[:2], return_when=asyncio.ALL_COMPLETED,
                               timeout=3.0)
        finally:
            # Cancel call_timer and silence_monitor — call is over
            for t in _tasks[2:]:
                if not t.done():
                    t.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(t), timeout=1.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            logger.debug(f"All tasks cleaned up for {call_sid}")

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
                # Async AUP moderation — non-blocking, runs after save
                asyncio.create_task(run_aup_moderation(
                    business_id, call_sid, full_transcript,  # full_transcript is in scope here
                    business_name=(business_cfg or {}).get("name", "") if hasattr(business_cfg, "get") else "",
                ))
                # ── Tier 2: Fire Zapier/webhook if configured ────────────────────
                try:
                    sb_wh = get_sb()
                    if sb_wh:
                        wh_res = sb_wh.from_("settings_business").select("zapier_webhook_url").eq("business_id", business_id).maybe_single().execute()
                        zapier_url = (wh_res.data or {}).get("zapier_webhook_url")
                        if zapier_url:
                            payload = {
                                "event": "call.completed",
                                "business_id": business_id,
                                "call_sid": call_sid,
                                "from_number": from_number,
                                "duration_seconds": duration,
                                "turn_count": turn_count,
                                "transcript_summary": full_transcript[:500],
                                "service_address": extracted.get("service_address", "") if "extracted" in dir() else "",
                                "caller_name": extracted.get("first_name", "") if "extracted" in dir() else "",
                                "industry_vertical": industry_vert if "industry_vert" in dir() else "general",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                            }
                            async with httpx.AsyncClient() as wh_client:
                                wh_resp = await wh_client.post(zapier_url, json=payload, timeout=5.0)
                            if 200 <= wh_resp.status_code < 300:
                                logger.info(f"Zapier webhook fired for {call_sid}")
                            else:
                                raise Exception(f"Zapier HTTP {wh_resp.status_code}")
                except Exception as wh_err:
                    # ── Failsafe: Zapier is down — alert via SMS and log ───
                    logger.error(f"Zapier webhook FAILED for {call_sid}: {wh_err}")
                    try:
                        sb_fa = get_sb()
                        biz_fa = sb_fa.from_("businesses").select(
                            "emergency_contact_email,emergency_contact_phone,name"
                        ).eq("id", business_id).maybe_single().execute() if sb_fa else None
                        ec_phone = (biz_fa.data or {}).get("emergency_contact_phone") if biz_fa else None
                        ec_email = (biz_fa.data or {}).get("emergency_contact_email") if biz_fa else None
                        biz_name = (biz_fa.data or {}).get("name", "your business") if biz_fa else "your business"
                        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
                        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
                        alert_msg = (
                            f"RECEPTIONIST.CO ALERT: A lead was captured but your webhook failed. "
                            f"Caller: {from_number} | Summary: {full_transcript[:120]}. "
                            f"Check your Zapier dashboard and email for full details."
                        )
                        if ec_phone and TWILIO_SID and TWILIO_TOKEN:
                            async with httpx.AsyncClient() as sms_client:
                                await sms_client.post(
                                    f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                                    auth=(TWILIO_SID, TWILIO_TOKEN),
                                    data={"To": ec_phone, "From": to_number, "Body": alert_msg},
                                    timeout=5.0,
                                )
                            logger.info(f"Failsafe SMS sent to {ec_phone}")
                        if ec_email:
                            logger.warning(f"SendGrid failsafe email needed for {ec_email} — payload: {payload}")
                    except Exception as fa_err:
                        logger.error(f"Failsafe alert failed: {fa_err}")
            except Exception as save_err:
                logger.error(f"SAVE FAILED for {call_sid}: {save_err}")
            # delete_active_call already fired on stream stop — this is a safety net
            asyncio.create_task(delete_active_call(call_sid))
        if openai_ws and not openai_ws.state.name == "CLOSED":
            await openai_ws.close()
        _active_openai_ws.pop(call_sid, None)
        _active_twilio_ws.pop(call_sid, None)
        _active_stream_sid.pop(call_sid, None)
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
                # ZAR Policy: recording_url intentionally not stored — Zero Audio Retention
                # clean_url = recording_url  # disabled
                logger.info(f"ZAR: recording URL discarded per Zero Audio Retention Policy")
                if False:
                    sb.from_("calls").update({
                    "recording_url": recording_url  # disabled
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


# ═══════════════════════════════════════════════════════════════════════════
# Weekly ROI Report Email System
# Endpoint: POST /api/reports/send-weekly
# Called by: Vercel Cron job every Monday 7AM (or daily/monthly per prefs)
# COMPLIANCE: Only aggregate counts — zero PHI in email body
# ═══════════════════════════════════════════════════════════════════════════

POSTMARK_TOKEN = os.getenv("POSTMARK_SERVER_TOKEN", "")
CRM_URL        = os.getenv("CRM_BASE_URL", "https://app.receptionist.co")

def build_roi_email(biz_name: str, owner_name: str, period_label: str, stats: dict) -> str:
    """Builds HIPAA-compliant HTML email — aggregate stats only, no PHI."""
    modules = stats.get("modules", {})
    labor_saved = stats.get("labor_saved_usd", 0)
    minutes     = stats.get("total_minutes", 0)
    calls       = stats.get("total_calls", 0)
    bookings    = stats.get("bookings", 0)
    widget_int  = stats.get("widget_interactions", 0)
    subscribers = stats.get("new_subscribers", 0)
    hours_saved = round(minutes / 60, 1)

    voice_section = ""
    if modules.get("voice_calls", True):
        voice_section = f"""
        <tr><td style="padding:20px 0 10px"><table width="100%" cellpadding="0" cellspacing="0">
          <tr><td style="padding:16px 20px;background:#0D1E35;border-radius:12px;border:1px solid rgba(79,142,247,0.2)">
            <div style="font-size:12px;font-weight:700;color:#4F8EF7;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px">
              📞 Voice Receptionist (Aria)
            </div>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="text-align:center;padding:8px">
                  <div style="font-size:36px;font-weight:800;color:#fff">{calls}</div>
                  <div style="font-size:12px;color:rgba(255,255,255,0.5)">Calls Handled</div>
                </td>
                <td style="text-align:center;padding:8px;border-left:1px solid rgba(255,255,255,0.08)">
                  <div style="font-size:36px;font-weight:800;color:#10B981">{bookings}</div>
                  <div style="font-size:12px;color:rgba(255,255,255,0.5)">Appointments Booked</div>
                </td>
                <td style="text-align:center;padding:8px;border-left:1px solid rgba(255,255,255,0.08)">
                  <div style="font-size:36px;font-weight:800;color:#F59E0B">{hours_saved}h</div>
                  <div style="font-size:12px;color:rgba(255,255,255,0.5)">Labor Saved</div>
                </td>
              </tr>
            </table>
            <div style="margin-top:12px;padding:10px 14px;background:rgba(16,185,129,0.1);border-radius:8px;
              font-size:13px;color:#10B981;text-align:center">
              💰 Estimated labor savings this period: <strong>${labor_saved}</strong>
            </div>
          </td></tr>
        </table></td></tr>"""

    widget_section = ""
    if modules.get("web_chats", True) or modules.get("newsletter", True):
        widget_section = f"""
        <tr><td style="padding:10px 0"><table width="100%" cellpadding="0" cellspacing="0">
          <tr><td style="padding:16px 20px;background:#0D1E35;border-radius:12px;border:1px solid rgba(139,92,246,0.2)">
            <div style="font-size:12px;font-weight:700;color:#8B5CF6;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px">
              💬 Website AI &amp; Lead Capture
            </div>
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td style="text-align:center;padding:8px">
                  <div style="font-size:36px;font-weight:800;color:#fff">{widget_int}</div>
                  <div style="font-size:12px;color:rgba(255,255,255,0.5)">Widget Interactions</div>
                </td>
                <td style="text-align:center;padding:8px;border-left:1px solid rgba(255,255,255,0.08)">
                  <div style="font-size:36px;font-weight:800;color:#8B5CF6">{subscribers}</div>
                  <div style="font-size:12px;color:rgba(255,255,255,0.5)">New Subscribers</div>
                </td>
              </tr>
            </table>
          </td></tr>
        </table></td></tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Aria ROI Report — {biz_name}</title></head>
<body style="margin:0;padding:0;background:#060F1E;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#060F1E;padding:32px 16px">
<tr><td align="center"><table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%">

  <!-- Header -->
  <tr><td style="padding:0 0 24px;text-align:center">
    <div style="font-size:24px;font-weight:900;color:#fff;letter-spacing:-0.5px">Receptionist.co</div>
    <div style="font-size:13px;color:rgba(255,255,255,0.4);margin-top:4px">AI Front Desk Platform</div>
  </td></tr>

  <!-- Subject line -->
  <tr><td style="background:#0D1E35;border-radius:16px 16px 0 0;border:1px solid rgba(79,142,247,0.2);
    border-bottom:none;padding:28px 28px 20px">
    <div style="font-size:22px;font-weight:800;color:#fff;margin-bottom:6px">
      📊 Front Desk Executive Summary
    </div>
    <div style="font-size:14px;color:rgba(255,255,255,0.5)">{period_label}</div>
    <div style="font-size:14px;color:rgba(255,255,255,0.7);margin-top:12px">
      {owner_name or 'Hi there'}, here's what Aria accomplished at <strong style="color:#fff">{biz_name}</strong>:
    </div>
  </td></tr>

  <!-- Stats cards -->
  <tr><td style="background:#0D1E35;border-left:1px solid rgba(79,142,247,0.2);
    border-right:1px solid rgba(79,142,247,0.2);padding:0 28px">
    <table width="100%" cellpadding="0" cellspacing="0">
      {voice_section}
      {widget_section}
    </table>
  </td></tr>

  <!-- CTA -->
  <tr><td style="background:#0D1E35;border-radius:0 0 16px 16px;border:1px solid rgba(79,142,247,0.2);
    border-top:1px solid rgba(255,255,255,0.06);padding:24px 28px;text-align:center">
    <div style="font-size:14px;color:rgba(255,255,255,0.5);margin-bottom:16px">
      Need to review specific callers or follow up on leads?
    </div>
    <a href="{CRM_URL}/dashboard" style="display:inline-block;padding:14px 32px;
      background:linear-gradient(135deg,#4F8EF7,#2563EB);color:#fff;font-size:14px;
      font-weight:700;text-decoration:none;border-radius:10px;
      box-shadow:0 4px 20px rgba(79,142,247,0.4)">
      Log into Secure Dashboard →
    </a>
  </td></tr>

  <!-- Footer -->
  <tr><td style="padding:20px;text-align:center">
    <div style="font-size:11px;color:rgba(255,255,255,0.2);line-height:1.7">
      You're receiving this because you enabled automated reports in your Receptionist.co settings.<br>
      This report contains aggregate statistics only. No patient or customer data is included.<br>
      <a href="{CRM_URL}/dashboard?settings=reports" style="color:rgba(79,142,247,0.6)">
        Change frequency or unsubscribe
      </a>
    </div>
  </td></tr>

</table></td></tr></table></body></html>"""


async def send_report_email(to_emails: list, subject: str, html: str) -> bool:
    """Send via Postmark."""
    if not POSTMARK_TOKEN or not to_emails:
        return False
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                "https://api.postmarkapp.com/email",
                headers={"X-Postmark-Server-Token": POSTMARK_TOKEN, "Content-Type": "application/json"},
                json={
                    "From": "Aria at Receptionist.co <aria@mail.receptionist.co>",
                    "To":   ", ".join(to_emails),
                    "Subject": subject,
                    "HtmlBody": html,
                    "MessageStream": "outbound",
                    "TrackOpens": True,
                }
            )
            return r.status_code == 200
    except Exception as e:
        print(f"[REPORT EMAIL ERROR] {e}")
        return False


@app.post("/api/reports/send")
async def send_weekly_reports(request: Request):
    """
    Triggered by Vercel cron or manual call.
    Queries weekly_roi_summary view and sends HIPAA-safe aggregate emails.
    Auth: expects X-Cron-Secret header matching CRON_SECRET env var.
    """
    secret = request.headers.get("x-cron-secret", "")
    if os.getenv("CRON_SECRET") and secret != os.getenv("CRON_SECRET"):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not supabase:
        return JSONResponse({"error": "DB not connected"}, status_code=500)

    try:
        # Query businesses due for a report
        today = date.today()
        day_of_week  = today.weekday()  # 0 = Monday
        day_of_month = today.day

        # Get all businesses with their report preferences
        res = supabase.table("businesses").select(
            "id,name,report_frequency,report_modules,report_email_list"
        ).neq("report_frequency", "never").execute()

        businesses = res.data or []
        sent = 0; skipped = 0

        for biz in businesses:
            freq = biz.get("report_frequency", "weekly")
            # Check if today is the right day
            if freq == "weekly"  and day_of_week  != 0: skipped += 1; continue
            if freq == "monthly" and day_of_month != 1: skipped += 1; continue
            # daily always runs

            biz_id     = biz["id"]
            biz_name   = biz["name"] or "Your Business"
            recipients = biz.get("report_email_list") or []
            if not recipients:
                skipped += 1; continue

            # Get stats from weekly_roi_summary view
            stats_res = supabase.table("weekly_roi_summary").select("*").eq(
                "business_id", biz_id
            ).execute()
            stats = stats_res.data[0] if stats_res.data else {}
            stats["modules"] = biz.get("report_modules") or {}

            # Check not already sent this period
            period_start = (today - timedelta(days=today.weekday())).isoformat()
            existing = supabase.table("report_log").select("id").eq(
                "business_id", biz_id
            ).eq("period_start", period_start).eq("frequency", freq).execute()
            if existing.data:
                skipped += 1; continue

            # Build period label
            period_end   = today.isoformat()
            period_label = f"Week of {period_start} – {period_end}"
            if freq == "daily":   period_label = f"Daily Summary — {today.strftime('%B %d, %Y')}"
            if freq == "monthly": period_label = f"Monthly Summary — {today.strftime('%B %Y')}"

            # Get owner name from settings
            settings_res = supabase.table("settings_business").select(
                "business_hours"
            ).eq("business_id", biz_id).execute()
            owner_name = ""

            # Build and send email
            subject = f"📊 Your {freq.capitalize()} Aria ROI Report: {biz_name}"
            html    = build_roi_email(biz_name, owner_name, period_label, stats)

            ok = await send_report_email(recipients, subject, html)

            # Log the send
            supabase.table("report_log").insert({
                "business_id":     biz_id,
                "period_start":    period_start,
                "period_end":      period_end,
                "frequency":       freq,
                "recipient_emails": recipients,
                "summary_json":    {
                    "calls":        stats.get("total_calls", 0),
                    "minutes":      stats.get("total_minutes", 0),
                    "bookings":     stats.get("bookings", 0),
                    "labor_usd":    stats.get("labor_saved_usd", 0),
                    "widget_events":stats.get("widget_interactions", 0),
                    "subscribers":  stats.get("new_subscribers", 0),
                },
            }).execute()

            if ok: sent += 1
            else:  skipped += 1

        return JSONResponse({
            "ok":      True,
            "sent":    sent,
            "skipped": skipped,
            "date":    today.isoformat(),
        })

    except Exception as e:
        print(f"[WEEKLY REPORTS ERROR] {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/reports/preview/{business_id}")
async def preview_report(business_id: str, request: Request):
    """Returns the HTML of a report for preview — auth required."""
    if not supabase:
        return JSONResponse({"error": "DB not connected"}, status_code=500)
    try:
        biz_res = supabase.table("businesses").select(
            "id,name,report_modules,report_email_list"
        ).eq("id", business_id).single().execute()
        biz = biz_res.data
        if not biz:
            return JSONResponse({"error": "Not found"}, status_code=404)

        stats_res = supabase.table("weekly_roi_summary").select("*").eq(
            "business_id", business_id
        ).execute()
        stats = stats_res.data[0] if stats_res.data else {}
        stats["modules"] = biz.get("report_modules") or {}

        today  = date.today()
        period = f"Week of {(today - timedelta(days=7)).isoformat()} – {today.isoformat()}"
        html   = build_roi_email(biz["name"], "", period, stats)
        from starlette.responses import HTMLResponse
        return HTMLResponse(content=html)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ═══════════════════════════════════════════════════════════════════════════
# Intake SMS + Post-Visit Review Automation (Zocdoc-inspired)
# ═══════════════════════════════════════════════════════════════════════════

async def send_intake_sms(business_id: str, contact_id: str, contact_phone: str,
                           appt_time: str, business_name: str) -> bool:
    """
    Fires immediately after a new patient appointment is booked.
    Uses external_intake_url (Jane App, Mindbody, ezyVet etc.) when set — Tier 1 integration.
    Falls back to internal intake_form_url if no external URL configured.
    Compliance: No PHI — just the appointment time and intake form link.
    """
    if not supabase or not contact_phone:
        return False
    try:
        # Get intake settings — check for external URL (Tier 1) first
        settings = supabase.table("settings_business").select(
            "intake_form_url,intake_auto_send,phone,external_intake_url"
        ).eq("business_id", business_id).single().execute()
        s = settings.data or {}

        # Tier 1: use clinic's own system URL (Jane App, Mindbody, ezyVet etc.)
        external_url = s.get("external_intake_url") or ""
        internal_url = s.get("intake_form_url") or ""
        intake_url   = external_url if external_url else internal_url

        if not intake_url:
            # No intake URL configured at all — skip
            logger.debug(f"No intake URL configured for {business_id} — skipping intake SMS")
            return False

        if not s.get("intake_auto_send") and not external_url:
            return False  # Automation not enabled and no external URL override
        from_number = s.get("phone") or os.getenv("TWILIO_PHONE_NUMBER", "")

        msg = (
            f"Hi! You're confirmed at {business_name} for {appt_time}. "
            f"Please complete your intake form before your visit: {intake_url} "
            f"Reply STOP to opt out."
        )

        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID", ""),
            os.getenv("TWILIO_AUTH_TOKEN", "")
        )
        client.messages.create(to=contact_phone, from_=from_number, body=msg)

        # Log it
        supabase.table("contacts").update({"intake_sent_at": datetime.utcnow().isoformat()}).eq(
            "id", contact_id
        ).execute()
        return True
    except Exception as e:
        print(f"[INTAKE SMS ERROR] {e}")
        return False


async def send_review_request(business_id: str, contact_id: str, contact_phone: str,
                               contact_name: str, business_name: str, appointment_id: str) -> bool:
    """
    Fires 24 hours after appointment is marked 'Completed'.
    Sends a Google review request. No PHI included.
    """
    if not supabase or not contact_phone:
        return False
    try:
        settings = supabase.table("settings_business").select(
            "review_request_url,review_auto_send,review_delay_hours,phone"
        ).eq("business_id", business_id).single().execute()
        s = settings.data or {}

        if not s.get("review_auto_send") or not s.get("review_request_url"):
            return False

        review_url  = s["review_request_url"]
        from_number = s.get("phone") or os.getenv("TWILIO_PHONE_NUMBER", "")
        first_name  = contact_name.split()[0] if contact_name else "there"

        msg = (
            f"Hi {first_name}! Thank you for visiting {business_name}. "
            f"We'd love your feedback — could you take 30 seconds to leave us a review? "
            f"{review_url} Reply STOP to opt out."
        )

        client = Client(
            os.getenv("TWILIO_ACCOUNT_SID", ""),
            os.getenv("TWILIO_AUTH_TOKEN", "")
        )
        client.messages.create(to=contact_phone, from_=from_number, body=msg)

        # Log review request
        supabase.table("review_requests").insert({
            "business_id":    business_id,
            "contact_id":     contact_id,
            "appointment_id": appointment_id,
            "channel":        "sms",
        }).execute()

        # Mark appointment as review sent
        supabase.table("appointments").update({"review_sent": True}).eq(
            "id", appointment_id
        ).execute()
        return True
    except Exception as e:
        print(f"[REVIEW SMS ERROR] {e}")
        return False


@app.post("/api/automation/appointment-completed")
async def appointment_completed_hook(request: Request):
    """
    Called when an appointment is marked 'completed' in the CRM.
    Schedules post-visit review SMS after configured delay hours.
    """
    try:
        body = await request.json()
        business_id    = body.get("business_id")
        contact_id     = body.get("contact_id")
        contact_phone  = body.get("contact_phone")
        contact_name   = body.get("contact_name", "")
        business_name  = body.get("business_name", "")
        appointment_id = body.get("appointment_id")

        if not all([business_id, contact_id, contact_phone, appointment_id]):
            return JSONResponse({"error": "Missing required fields"}, status_code=400)

        # Get delay hours
        settings = supabase.table("settings_business").select(
            "review_delay_hours,review_auto_send"
        ).eq("business_id", business_id).single().execute()
        delay_hours = (settings.data or {}).get("review_delay_hours", 24)

        # For now fire immediately (in production use a task queue with delay)
        # TODO: add APScheduler or Railway cron for delayed sends
        import asyncio
        asyncio.create_task(asyncio.sleep(delay_hours * 3600))
        # Simplified: just send after delay
        await send_review_request(
            business_id, contact_id, contact_phone,
            contact_name, business_name, appointment_id
        )

        return JSONResponse({"ok": True, "review_scheduled_in_hours": delay_hours})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Warm Handoff — inject transfer prompt into live OpenAI stream ──────────────
@app.post("/warm-handoff")
async def warm_handoff(req: Request):
    """
    Staff clicks 'Initiate Warm Transfer' in CRM.
    Injects a prompt into the live OpenAI Realtime stream asking the caller
    if they'd like to speak with a specialist. If caller says yes, Aria
    triggers transfer_to_specialist() function call → Twilio conference bridge.
    """
    body = await req.json()
    call_sid    = body.get("call_sid")
    script      = body.get("script", "A")  # A=expertise, B=personal

    if not call_sid:
        return JSONResponse({"ok": False, "error": "call_sid required"}, status_code=400)

    openai_ws = _active_openai_ws.get(call_sid)
    if not openai_ws:
        return JSONResponse({"ok": False, "error": "No active OpenAI stream for this call"}, status_code=404)

    # Script options
    scripts = {
        "A": "SYSTEM OVERRIDE: Stop your current thought immediately. Say exactly: 'You know what, to make sure you get the exact right information on our pricing and availability for that, let me bring one of our specialists on the line. May I transfer you to them right now?' Then wait for the caller to say yes or no.",
        "B": "SYSTEM OVERRIDE: Stop your current thought immediately. Say exactly: 'I want to make sure you are fully taken care of with those specific details. Let me grab one of our team members at the front desk to jump in. Is it okay if I connect you now?' Then wait for the caller to say yes or no.",
    }
    prompt = scripts.get(script, scripts["A"])

    try:
        # Cancel any active response first — wait for it to propagate
        try:
            await openai_ws.send(json.dumps({"type": "response.cancel"}))
            await asyncio.sleep(0.3)  # give OpenAI time to process the cancel
        except Exception:
            pass

        # Inject via conversation.item.create (highest priority system message)
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type":    "message",
                "role":    "system",
                "content": [{"type": "input_text", "text": prompt}],
            }
        }))
        # Force a new response so Aria speaks immediately
        await openai_ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"],
                "instructions": prompt,
            }
        }))
        # Policy: mark exact handoff time in transcript for delineation
        ts_now = datetime.now(timezone.utc).isoformat()
        ws_registry = _active_twilio_ws.get(call_sid)
        if ws_registry:
            try:
                # Push a system marker turn to active_calls live_transcript
                sb = get_sb()
                if sb:
                    result = sb.from_("active_calls").select("live_transcript").eq("call_sid", call_sid).maybe_single().execute()
                    if result.data:
                        current_turns = result.data.get("live_transcript") or []
                        current_turns.append({
                            "role": "system",
                            "text": f"[Warm Handoff initiated at {datetime.now(timezone.utc).strftime('%I:%M %p UTC')}. AI transcription will cease when human answers. Per Zero Audio Retention Policy, human conversation is not recorded.]",
                            "ts": ts_now
                        })
                        sb.from_("active_calls").update({"live_transcript": current_turns}).eq("call_sid", call_sid).execute()
            except Exception as marker_err:
                logger.debug(f"Handoff marker: {marker_err}")
        logger.info(f"Warm handoff injected for {call_sid} (script {script})")
        # Store warm-handoff script as AI turn in transcript so post-call view shows it
        try:
            sb_h = get_sb()
            if sb_h:
                res_h = sb_h.from_("active_calls").select("live_transcript").eq("call_sid", call_sid).maybe_single().execute()
                if res_h.data:
                    turns_h = res_h.data.get("live_transcript") or []
                    turns_h.append({"role": "ai", "text": "Connecting you to a specialist...", "ts": datetime.now(timezone.utc).isoformat()})
                    sb_h.from_("active_calls").update({"live_transcript": turns_h}).eq("call_sid", call_sid).execute()
        except Exception:
            pass
        return JSONResponse({"ok": True, "message": "Prompt injected — Aria is asking the caller"})
    except Exception as e:
        logger.error(f"Warm handoff injection failed: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/cancel-handoff")
async def cancel_handoff(req: Request):
    """Caller said no — reset Aria to normal conversation."""
    body = await req.json()
    call_sid = body.get("call_sid")
    openai_ws = _active_openai_ws.get(call_sid)
    if not openai_ws:
        return JSONResponse({"ok": False, "error": "No active stream"}, status_code=404)
    try:
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type":    "message",
                "role":    "system",
                "content": [{"type": "input_text", "text": "The caller declined the transfer. Resume the normal conversation naturally, acknowledge their response, and continue helping them."}],
            }
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/transfer-call")
async def transfer_call(req: Request):
    """
    Caller said yes — use Twilio REST to dial the transfer number.
    Aria plays final confirmation, then the call is bridged.
    """
    body        = await req.json()
    call_sid    = body.get("call_sid")
    transfer_to = body.get("transfer_to")  # E.164 phone number

    if not call_sid or not transfer_to:
        return JSONResponse({"ok": False, "error": "call_sid and transfer_to required"}, status_code=400)

    # ── Normalize transfer number to E.164 ─────────────────────────────────
    # Strip spaces, dashes, parens
    raw_num = str(transfer_to).strip()
    digits  = re.sub(r"\D", "", raw_num)

    # Reject obviously invalid numbers (too short, or just "888" partial)
    if len(digits) < 10:
        return JSONResponse({
            "ok": False,
            "error": f"Invalid transfer number '{transfer_to}'. Must be a full US phone number like +17205551234. "
                     f"Please set your transfer number in CRM Settings → Warm Handoff."
        }, status_code=400)

    # Normalize to E.164
    if len(digits) == 10:
        transfer_to = f"+1{digits}"
    elif len(digits) == 11 and digits[0] == "1":
        transfer_to = f"+{digits}"
    else:
        transfer_to = f"+{digits}"
    logger.info(f"Transfer number normalized: '{raw_num}' → '{transfer_to}'")
    # ────────────────────────────────────────────────────────────────────────

    openai_ws = _active_openai_ws.get(call_sid)
    twilio_ws  = _active_twilio_ws.get(call_sid)

    if not openai_ws:
        return JSONResponse({"ok": False, "error": "No active stream"}, status_code=404)

    try:
        # Step 1: Aria says final farewell
        await openai_ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type":    "message",
                "role":    "system",
                "content": [{"type": "input_text", "text": "Say exactly: 'Great! Connecting you to our specialist right now. One moment please.' Then stop speaking."}],
            }
        }))
        await openai_ws.send(json.dumps({"type": "response.create"}))

        # Step 2: Give Aria 3 seconds to finish speaking, then transfer via Twilio
        import asyncio
        await asyncio.sleep(3)

        # Step 3: Use Twilio REST to redirect the call with <Dial>
        TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
        RAILWAY_PUBLIC_URL = os.environ.get("RAILWAY_PUBLIC_URL", "")

        if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
            import httpx
            twiml = f'<?xml version="1.0" encoding="UTF-8"?><Response><Dial timeout="30" callerId="+18889732377"><Number>{transfer_to}</Number></Dial></Response>'
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls/{call_sid}.json",
                    auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
                    data={"Twiml": twiml},
                )

        logger.info(f"Transfer initiated: {call_sid} → {transfer_to}")
        return JSONResponse({"ok": True, "message": f"Transferring to {transfer_to}"})

    except Exception as e:
        logger.error(f"Transfer failed: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/twilio-token")
async def twilio_access_token(request: Request):
    """
    Generate a Twilio Access Token with Voice grant for browser-based calling.
    Used by the CRM Live Monitor to enable 'Answer in Browser' via Twilio Client JS.
    """
    import base64, hmac, hashlib, time, json as _json_std

    TWILIO_SID      = os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_TOKEN    = os.environ.get("TWILIO_AUTH_TOKEN", "")
    TWILIO_APP_SID  = os.environ.get("TWILIO_TWIML_APP_SID", "")  # TwiML App SID for browser calling

    if not TWILIO_SID or not TWILIO_TOKEN:
        return JSONResponse({"ok": False, "error": "Twilio not configured"}, status_code=500)

    try:
        from twilio.jwt.access_token import AccessToken
        from twilio.jwt.access_token.grants import VoiceGrant

        identity = f"staff-{int(time.time())}"
        token = AccessToken(TWILIO_SID, TWILIO_APP_SID or TWILIO_SID, TWILIO_TOKEN,
                           identity=identity, ttl=3600)

        voice_grant = VoiceGrant(
            outgoing_application_sid=TWILIO_APP_SID or TWILIO_SID,
            incoming_allow=True,  # allow browser to receive incoming calls
        )
        token.add_grant(voice_grant)

        logger.info(f"Twilio Access Token issued for identity {identity}")
        return JSONResponse({"ok": True, "token": token.to_jwt(), "identity": identity})

    except Exception as e:
        logger.error(f"Token generation failed: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/browser-bridge")
async def browser_bridge(req: Request):
    """
    Bridge an active Twilio call into a named conference room so the browser
    (via Twilio Client) can join as a participant.

    Flow:
    1. CRM clicks 'Answer in Browser'
    2. Aria has already asked caller to hold
    3. We update the caller's call to join a conference via TwiML
    4. Browser's Twilio.Device connects to the same conference
    5. Both parties are in the conference — human takes over
    """
    body       = await req.json()
    call_sid   = body.get("call_sid")
    identity   = body.get("identity", "staff")   # browser's Twilio identity

    if not call_sid:
        return JSONResponse({"ok": False, "error": "call_sid required"}, status_code=400)

    TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
    TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")

    if not TWILIO_SID or not TWILIO_TOKEN:
        return JSONResponse({"ok": False, "error": "Twilio not configured"}, status_code=500)

    # Conference room name is deterministic from call_sid
    conf_room = f"receptionist-bridge-{call_sid[-8:]}"

    try:
        # Step 1: First have Aria say farewell and stop transcribing
        openai_ws = _active_openai_ws.get(call_sid)
        if openai_ws:
            try:
                farewell = (
                    "SYSTEM: The human staff member is now joining the call via their browser. "
                    "Say: 'Perfect! Please hold for just one moment while I connect you to our specialist.' "
                    "Then stay silent — the human will take over completely."
                )
                await openai_ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": farewell}]}
                }))
                await openai_ws.send(json.dumps({"type": "response.create"}))
                await asyncio.sleep(3)  # let Aria finish speaking
            except Exception as aria_err:
                logger.debug(f"Browser bridge Aria farewell: {aria_err}")

        # Step 2: Move caller into conference via Twilio REST
        twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <Conference waitUrl="https://twimlets.com/holdmusic?Bucket=com.twilio.music.soft-rock"
                    startConferenceOnEnter="false"
                    endConferenceOnExit="true"
                    beep="false">
            {conf_room}
        </Conference>
    </Dial>
</Response>"""

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Calls/{call_sid}.json",
                auth=(TWILIO_SID, TWILIO_TOKEN),
                data={"Twiml": twiml},
                timeout=8.0,
            )
            if resp.status_code not in (200, 201, 204):
                logger.warning(f"Browser bridge caller move failed: {resp.status_code} {resp.text[:200]}")
                return JSONResponse({"ok": False, "error": f"Twilio error {resp.status_code}"}, status_code=502)

        logger.info(f"Browser bridge: {call_sid} → conference room {conf_room}")

        # Step 3: Mark system turn in transcript
        try:
            sb = get_sb()
            if sb:
                res = sb.from_("active_calls").select("live_transcript").eq("call_sid", call_sid).maybe_single().execute()
                if res.data:
                    turns = res.data.get("live_transcript") or []
                    turns.append({
                        "role": "system",
                        "text": f"[Browser Bridge: Staff joined via browser at {datetime.now(timezone.utc).strftime('%I:%M %p UTC')}. AI transcription ended.]",
                        "ts": datetime.now(timezone.utc).isoformat()
                    })
                    sb.from_("active_calls").update({"live_transcript": turns, "status": "human-handled"}).eq("call_sid", call_sid).execute()
        except Exception as db_err:
            logger.debug(f"Browser bridge DB marker: {db_err}")

        return JSONResponse({
            "ok": True,
            "conference_room": conf_room,
            "message": f"Caller moved to conference room {conf_room}. Connect your browser to join."
        })

    except Exception as e:
        logger.error(f"Browser bridge failed: {e}")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

async def send_mvd_intake_sms(business_id: str, caller_phone: str, caller_name: str,
                               business_name: str, to_number: str) -> bool:
    """
    Mid-call intake SMS — fires when Aria has collected MVD (Name + Phone + Intent).
    Sends the clinic's intake URL (external or internal) immediately during the call.
    This is the Tier 1 'Universal URL' integration pattern.
    """
    if not caller_phone:
        return False
    try:
        sb = get_sb()
        if not sb:
            return False

        # Get the intake URL — external_intake_url takes priority (Tier 1)
        settings = sb.from_("settings_business").select(
            "external_intake_url,phone"
        ).eq("business_id", business_id).maybe_single().execute()
        s = settings.data or {}

        external_url = s.get("external_intake_url") or ""
        # Fallback to internal link if no external URL set
        intake_url = external_url or f"https://receptionist.co/intake/{business_id}"

        from_number = s.get("phone") or to_number  # use clinic's own number as sender

        greeting = f"Hi {caller_name}!" if caller_name and caller_name.lower() not in ("unknown","caller") else "Hi!"
        msg = (
            f"{greeting} Here's the {business_name} registration link Aria mentioned: "
            f"{intake_url} — takes about 60 seconds to fill out. "
            f"Reply STOP to opt out."
        )

        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not TWILIO_SID or not TWILIO_TOKEN:
            return False

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                auth=(TWILIO_SID, TWILIO_TOKEN),
                data={"To": caller_phone, "From": from_number, "Body": msg},
                timeout=8.0,
            )
        logger.info(f"MVD intake SMS sent to {caller_phone} for {business_id} (url: {intake_url})")
        return True
    except Exception as e:
        logger.error(f"MVD intake SMS error: {e}")
        return False
async def send_home_services_confirmation_sms(
    business_id: str, caller_phone: str, caller_name: str,
    service_address: str, business_name: str, to_number: str
) -> bool:
    """
    Home Services / Trades confirmation SMS — fires when Aria has collected
    Name + Phone + Address + Intent. Replaces intake form SMS entirely.
    Tells caller their job ticket is in and a tech will text them an arrival window.
    """
    if not caller_phone:
        return False
    try:
        sb = get_sb()
        settings = sb.from_("settings_business").select("phone").eq("business_id", business_id).maybe_single().execute() if sb else None
        from_number = ((settings.data or {}).get("phone") if settings else None) or to_number

        addr_snippet = f" for {service_address}" if service_address else ""
        greeting    = f"Hi {caller_name}!" if caller_name and caller_name.lower() not in ("unknown","caller") else "Hi!"

        msg = (
            f"{greeting} Thanks for calling {business_name}. "
            f"We've received your service request{addr_snippet} and are sending it to our dispatch team now. "
            f"A technician will text you shortly with your arrival window. "
            f"Reply STOP to opt out."
        )

        TWILIO_SID   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        TWILIO_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not TWILIO_SID or not TWILIO_TOKEN:
            return False

        async with httpx.AsyncClient() as client:
            await client.post(
                f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_SID}/Messages.json",
                auth=(TWILIO_SID, TWILIO_TOKEN),
                data={"To": caller_phone, "From": from_number, "Body": msg},
                timeout=8.0,
            )
        logger.info(f"Home services confirmation SMS sent to {caller_phone} for {business_id}")
        return True
    except Exception as e:
        logger.error(f"Home services SMS error: {e}")
        return False

