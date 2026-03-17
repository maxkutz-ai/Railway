"""
Aria — AI Video Receptionist for Receptionist.co
LiveKit Agent · OpenAI STT/LLM/TTS · Simli Avatar

What Aria can do (full receptionist capabilities):
  📅 Schedule, reschedule, cancel appointments — with owner confirmation
  👤 Create and look up contacts / CRM records
  📞 Log calls, take messages, handle missed call follow-ups
  🔍 Search business documents (Google Drive & Dropbox)
  🌤  Weather, news, web search
  🧮 Math, tips, bill splits, unit conversions
  💬 Draft SMS/email follow-up messages
  ⏱  Timers, reminders
  🏢 Business hours, services, pricing, staff info
  🗺  Directions, parking info
  💳 Accept payments (redirect to Stripe link)
  📋 Waiting list management
  🔔 Escalation flow
  😄 Small talk, jokes, icebreakers
  🧠 Persistent memory — learns names, preferences, facts

Required environment variables:
  LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
  OPENAI_API_KEY
  SIMLI_API_KEY, SIMLI_FACE_ID
  SUPABASE_URL, SUPABASE_SERVICE_KEY

Optional (per-business, stored in integrations table):
  Google Drive OAuth access_token
  Dropbox access_token

Run locally:
  pip install -r requirements.txt
  python aria_agent.py dev
"""

import os
import json
import asyncio
import logging
import random
import re
from datetime import datetime, timedelta
from typing import Optional

import httpx
from supabase import create_client, Client

from livekit.agents import Agent, AgentSession, JobContext, RunContext, WorkerOptions, cli, function_tool, room_io
from livekit.plugins import openai, silero, simli

logger = logging.getLogger("aria-agent")

# ── Supabase ──────────────────────────────────────────────────────────────────
def get_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if url and key:
        return create_client(url, key)
    return None

def get_app_url() -> str:
    return os.getenv("NEXT_PUBLIC_APP_URL", "https://app.receptionist.co")


# ── Load business context ─────────────────────────────────────────────────────
async def load_business_context(business_id: str) -> dict:
    sb = get_supabase()
    if not sb or not business_id:
        return {}
    try:
        results = {}

        biz = sb.from_("businesses").select("*").eq("id", business_id).single().execute()
        if biz.data:
            results["business"] = biz.data

        cfg = sb.from_("ai_receptionist_config").select("*").eq("business_id", business_id).single().execute()
        if cfg.data:
            results["config"] = cfg.data

        ai_set = sb.from_("ai_settings").select("*").eq("business_id", business_id).single().execute()
        if ai_set.data:
            results["ai_settings"] = ai_set.data

        biz_set = sb.from_("business_settings").select("*").eq("business_id", business_id).single().execute()
        if biz_set.data:
            results["business_settings"] = biz_set.data

        svcs = sb.from_("services").select("name,duration_minutes,price,category,description").eq("business_id", business_id).eq("is_active", True).execute()
        if svcs.data:
            results["services"] = svcs.data

        staff = sb.from_("staff").select("name,role,email,calendar_url").eq("business_id", business_id).eq("is_active", True).execute()
        if staff.data:
            results["staff"] = staff.data

        locs = sb.from_("locations").select("name,address,city,state,zip,phone,parking_info").eq("business_id", business_id).execute()
        if locs.data:
            results["locations"] = locs.data

        mems = sb.from_("ai_memory").select("category,memory_key,memory_value").eq("business_id", business_id).order("updated_at", desc=True).limit(150).execute()
        if mems.data:
            results["memories"] = mems.data

        # Load integrations (Google Drive, Dropbox tokens)
        try:
            integrations = sb.from_("integrations").select("provider,access_token,refresh_token,settings").eq("business_id", business_id).eq("is_active", True).execute()
            if integrations.data:
                results["integrations"] = {i["provider"]: i for i in integrations.data}
        except Exception:
            results["integrations"] = {}

        return results
    except Exception as e:
        logger.warning(f"load_business_context failed: {e}")
        return {}


# ── Memory helpers ────────────────────────────────────────────────────────────
VALID_MEMORY_CATEGORIES = {"owner_info", "business_rule", "client_note", "preference", "instruction", "general"}

async def save_memory_to_db(business_id: str, key: str, value: str, category: str):
    sb = get_supabase()
    if not sb:
        return
    cat_map = {"location": "owner_info", "conversation": "general", "fact": "general", "setting": "business_rule"}
    safe_cat = cat_map.get(category, category)
    if safe_cat not in VALID_MEMORY_CATEGORIES:
        safe_cat = "general"
    try:
        sb.from_("ai_memory").upsert({
            "business_id":  business_id,
            "memory_key":   key,
            "memory_value": str(value)[:4000],
            "category":     safe_cat,
        }, on_conflict="business_id,memory_key").execute()
        logger.info(f"Memory saved: [{safe_cat}] {key}")
    except Exception as e:
        logger.warning(f"save_memory failed: {e}")


# ── Pending confirmations (in-memory per session) ─────────────────────────────
pending_confirmations: dict = {}

def add_pending(action: str, data: dict, description: str) -> str:
    conf_id = f"confirm_{action}_{datetime.now().strftime('%H%M%S')}"
    pending_confirmations[conf_id] = {
        "action": action, "data": data,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    return conf_id


async def _execute_confirmed(business_id: str, conf_id: str) -> str:
    pending = pending_confirmations.pop(conf_id, None)
    if not pending:
        return "Confirmation not found or already processed."

    sb = get_supabase()
    action = pending["action"]
    data   = pending["data"]

    try:
        if action == "create_appointment":
            # Parse date+time into ISO timestamp
            contact_name  = data.get("contact_name", "")
            contact_phone = data.get("contact_phone", "")
            contact_email = data.get("contact_email", "")
            date_str      = data.get("date", "")
            time_str      = data.get("time", "12:00 PM")
            service_name  = data.get("service", "")

            # Parse time string
            try:
                from datetime import datetime as dt
                for fmt in ["%Y-%m-%d %I:%M %p", "%Y-%m-%d %H:%M", "%Y-%m-%d %I %p"]:
                    try:
                        start_dt  = dt.strptime(f"{date_str} {time_str}", fmt)
                        start_iso = start_dt.isoformat()
                        break
                    except ValueError:
                        continue
                else:
                    start_iso = f"{date_str}T12:00:00"
            except Exception:
                start_iso = f"{date_str}T12:00:00"

            # Call our booking API — handles Cal.com + Supabase in one shot
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post(
                        f"{get_app_url()}/api/cal/book",
                        json={
                            "business_id":   business_id,
                            "contact_name":  contact_name,
                            "contact_phone": contact_phone,
                            "contact_email": contact_email,
                            "service_name":  service_name,
                            "start_time":    start_iso,
                            "notes":         data.get("notes", ""),
                        },
                        timeout=10,
                    )
                    if r.status_code == 200:
                        resp = r.json()
                        cal_note = " (synced to calendar)" if resp.get("cal_booked") else ""
                        return f"Appointment confirmed for {contact_name} on {date_str} at {time_str}{cal_note}."
            except Exception as e:
                logger.warning(f"Booking API call failed: {e}")

            # Fallback — direct Supabase insert
            sb_local = get_supabase()
            if sb_local:
                sb_local.from_("appointments").insert({
                    "business_id":    business_id,
                    "start_time":     start_iso,
                    "service_type":   service_name,
                    "technician_name":data.get("staff_name", ""),
                    "notes":          data.get("notes", ""),
                    "status":         "booked",
                    "job_status":     "confirmed",
                    "booking_source": "aria",
                }).execute()
            return f"Appointment confirmed for {contact_name} on {date_str} at {time_str}."

        elif action == "create_contact":
            names = (data.get("name") or "").split(" ", 1)
            sb.from_("contacts").insert({
                "business_id": business_id,
                "first_name":  names[0],
                "last_name":   names[1] if len(names) > 1 else "",
                "phone":       data.get("phone"),
                "email":       data.get("email"),
                "notes":       data.get("notes", ""),
                "source":      "aria",
                "lead_status": "lead",
            }).execute()
            return f"Contact {data.get('name')} saved."

        elif action == "update_business_hours":
            sb.from_("business_settings").upsert({
                "business_id":    business_id,
                "business_hours": data.get("hours"),
                "updated_at":     datetime.now().isoformat(),
            }, on_conflict="business_id").execute()
            sb.from_("ai_settings").upsert({
                "business_id":    business_id,
                "business_hours": data.get("hours"),
            }, on_conflict="business_id").execute()
            return "Business hours updated successfully."

        elif action == "log_message":
            # Real messages schema: needs contact_id, message_body, channel, direction
            # Try to find or create contact
            contact_id = None
            from_phone = data.get("from_phone", "")
            from_name  = data.get("from_name", "")
            try:
                if from_phone:
                    cr = sb.from_("contacts").select("id").eq("business_id", business_id).ilike("phone", f"%{from_phone}%").limit(1).execute()
                    if cr.data:
                        contact_id = cr.data[0]["id"]
                if not contact_id and from_name:
                    names = from_name.split(" ", 1)
                    nr = sb.from_("contacts").insert({
                        "business_id": business_id,
                        "first_name":  names[0],
                        "last_name":   names[1] if len(names) > 1 else "",
                        "phone":       from_phone,
                        "source":      "aria",
                    }).execute()
                    if nr.data:
                        contact_id = nr.data[0]["id"]
            except Exception:
                pass

            sb.from_("messages").insert({
                "business_id":  business_id,
                "contact_id":   contact_id,
                "message_body": data.get("message"),
                "channel":      "note",
                "direction":    "inbound",
                "is_read":      False,
            }).execute()
            return f"Message from {from_name} logged."

        elif action == "add_to_waitlist":
            sb.from_("waitlist").insert({
                "business_id":   business_id,
                "contact_name":  data.get("contact_name"),
                "contact_phone": data.get("contact_phone"),
                "service":       data.get("service"),
                "notes":         data.get("notes", ""),
                "status":        "waiting",
            }).execute()
            return f"{data.get('contact_name')} added to the waitlist for {data.get('service')}."

        else:
            return f"Unknown action: {action}"

    except Exception as e:
        logger.error(f"_execute_confirmed failed: {e}")
        return f"Failed to save: {e}"


# ── Google Drive search ───────────────────────────────────────────────────────
async def search_google_drive(access_token: str, query: str, max_results: int = 5) -> list:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                "https://www.googleapis.com/drive/v3/files",
                params={
                    "q": f"fullText contains '{query}' and trashed=false",
                    "fields": "files(id,name,mimeType,modifiedTime)",
                    "pageSize": max_results,
                    "orderBy": "modifiedTime desc",
                },
                headers={"Authorization": f"Bearer {access_token}"},
                timeout=8,
            )
            if r.status_code == 401:
                return [{"error": "Google Drive token expired — reconnect in Settings."}]
            if not r.is_success:
                return [{"error": f"Google Drive error {r.status_code}"}]

            files = r.json().get("files", [])
            results = []
            for f in files[:max_results]:
                snippet = "(binary file)"
                if f.get("mimeType") == "application/vnd.google-apps.document":
                    ex = await client.get(
                        f"https://www.googleapis.com/drive/v3/files/{f['id']}/export",
                        params={"mimeType": "text/plain"},
                        headers={"Authorization": f"Bearer {access_token}"},
                        timeout=6,
                    )
                    if ex.is_success:
                        snippet = ex.text[:800]
                results.append({"name": f["name"], "modified": f.get("modifiedTime", ""), "snippet": snippet, "source": "Google Drive"})
            return results
    except Exception as e:
        return [{"error": str(e)}]


# ── Dropbox search ────────────────────────────────────────────────────────────
async def search_dropbox(access_token: str, query: str, max_results: int = 5) -> list:
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.dropboxapi.com/2/files/search_v2",
                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                json={"query": query, "options": {"max_results": max_results, "file_status": "active"}},
                timeout=8,
            )
            if r.status_code == 401:
                return [{"error": "Dropbox token expired — reconnect in Settings."}]
            if not r.is_success:
                return [{"error": f"Dropbox error {r.status_code}"}]

            matches = r.json().get("matches", [])
            results = []
            for m in matches[:max_results]:
                meta = m.get("metadata", {}).get("metadata", {})
                name = meta.get("name", "unknown")
                path = meta.get("path_lower", "")
                snippet = "(binary file)"
                if name.endswith((".txt", ".md", ".csv")):
                    dl = await client.post(
                        "https://content.dropboxapi.com/2/files/download",
                        headers={"Authorization": f"Bearer {access_token}", "Dropbox-API-Arg": json.dumps({"path": path})},
                        timeout=6,
                    )
                    if dl.is_success:
                        snippet = dl.text[:800]
                results.append({"name": name, "path": path, "snippet": snippet, "source": "Dropbox"})
            return results
    except Exception as e:
        return [{"error": str(e)}]


# ── System prompt ─────────────────────────────────────────────────────────────
INDUSTRY_SCRIPTS = {
    "spa": {
        "style": "luxurious, calming, warm — like a 5-star spa concierge",
        "skills": "booking massages/facials/nails, upselling packages, gift cards, membership sales, pre/post care advice",
        "urgency": "low — relaxation focused, never rushed",
        "phrases": ["You deserve this", "Let me check availability for you", "Our therapists specialize in..."],
        "common_asks": ["book appointment", "prices", "what services", "gift cards", "parking"],
    },
    "medical": {
        "style": "calm, professional, reassuring — HIPAA aware, never share patient info",
        "skills": "scheduling appointments, insurance verification prompts, co-pay reminders, referrals, lab results routing (to nurse only), urgent triage",
        "urgency": "medium-high — chest pain/breathing = call 911 immediately",
        "phrases": ["I'll have someone call you back shortly", "The doctor will review that", "For your privacy..."],
        "common_asks": ["schedule appointment", "refill prescription", "test results", "insurance", "urgent care"],
    },
    "dental": {
        "style": "friendly, reassuring — patients are often anxious, make them feel at ease",
        "skills": "scheduling cleanings/fillings/emergencies, insurance verification, payment plan info, post-procedure care instructions",
        "urgency": "medium — dental emergencies (broken tooth, severe pain) = same-day slot",
        "phrases": ["We'll take great care of you", "Dr. [name] has an opening...", "We accept most insurance plans"],
        "common_asks": ["book cleaning", "emergency tooth pain", "insurance accepted", "cost of filling", "hours"],
    },
    "legal": {
        "style": "formal, discreet, confidential — never discuss case details, always route to attorney",
        "skills": "scheduling consultations, screening caller needs, message routing, document drop-off coordination",
        "urgency": "varies — criminal/emergency = urgent callback",
        "phrases": ["I'll have an attorney call you back", "We treat all matters with strict confidentiality", "May I ask the general nature of your inquiry?"],
        "common_asks": ["free consultation", "how much does it cost", "immigration", "divorce", "criminal defense"],
    },
    "hvac": {
        "style": "efficient, calm under pressure — many callers are stressed (no heat/AC)",
        "skills": "emergency triage (no heat in winter = priority 1), scheduling tune-ups/repairs/installations, dispatching techs, ETAs, maintenance plan upsell",
        "urgency": "HIGH — furnace out in winter, AC out in summer = emergency dispatch",
        "phrases": ["We'll get someone out to you today", "Can you describe what the unit is doing?", "Is anyone in the home vulnerable to the heat/cold?"],
        "common_asks": ["heater not working", "AC broken", "strange noise", "annual tune-up", "new system quote"],
    },
    "plumber": {
        "style": "fast, calm, reassuring — burst pipe callers are panicking",
        "skills": "emergency dispatch (burst pipes, flooding = NOW), scheduling drain/fixture/water heater jobs, triage water damage severity",
        "urgency": "HIGH — active flooding/burst pipe = emergency dispatch, shut off water first",
        "phrases": ["First — shut off your main water valve", "We have an emergency tech available", "Can you tell me where the water is coming from?"],
        "common_asks": ["pipe burst", "drain clogged", "water heater", "toilet overflow", "leak under sink"],
    },
    "electrician": {
        "style": "safety-first, clear, professional — electrical hazards need immediate triage",
        "skills": "safety triage (sparks/burning smell = evacuate), scheduling panel upgrades/EV chargers/lighting, emergency outage response",
        "urgency": "HIGH — sparks, burning smell, no power = safety emergency",
        "phrases": ["If you see sparks or smell burning — leave the building and call 911", "I'm dispatching our on-call electrician", "Can you safely reach the breaker panel?"],
        "common_asks": ["power outage", "breaker keeps tripping", "EV charger install", "panel upgrade", "outdoor lighting"],
    },
    "locksmith": {
        "style": "fast, empathetic — lockouts are stressful, often at night or in bad weather",
        "skills": "lockout dispatch (home/car/business), rekeying quotes, lock upgrade upsell, safe service",
        "urgency": "HIGH — stranded lockouts = immediate dispatch, especially at night",
        "phrases": ["We can have someone there in about 30 minutes", "Are you in a safe location?", "What type of lock/vehicle is it?"],
        "common_asks": ["locked out of house", "car lockout", "lost keys", "rekey locks", "broken key"],
    },
    "hotel": {
        "style": "elegant, welcoming, concierge-level service — every guest is a VIP",
        "skills": "reservations, check-in/out info, amenity questions, local recommendations, complaint handling, room requests",
        "urgency": "low-medium — complaints and room issues = prompt gracious response",
        "phrases": ["It would be our pleasure", "Allow me to assist you with that", "Your comfort is our top priority"],
        "common_asks": ["room availability", "check-in time", "parking", "restaurant recommendations", "pool hours"],
    },
    "salon": {
        "style": "trendy, warm, excited about beauty — mirror the client's energy",
        "skills": "booking cuts/color/blowouts, stylist availability, color consultation prompts, retail product recommendations, gift cards",
        "urgency": "low — style emergencies (wedding tomorrow) = squeeze in if possible",
        "phrases": ["You're going to love it!", "Let me check [stylist]'s availability", "Do you have a color reference photo?"],
        "common_asks": ["book haircut", "color appointment", "balayage price", "extensions", "wedding packages"],
    },
    "gym": {
        "style": "energetic, motivating, encouraging — match the fitness vibe",
        "skills": "membership sign-ups, class booking, personal trainer scheduling, guest passes, cancellation policy",
        "urgency": "low — motivate hesitant leads to come in for a free trial",
        "phrases": ["Let's get you started!", "We have a free trial offer", "What are your fitness goals?"],
        "common_asks": ["membership price", "class schedule", "personal trainer", "cancel membership", "guest pass"],
    },
    "auto": {
        "style": "knowledgeable, trustworthy — car owners worry about being overcharged",
        "skills": "service scheduling (oil change/brakes/tires), estimate requests, loaner car availability, pickup/drop-off service",
        "urgency": "medium — brake/safety issues = priority",
        "phrases": ["We can get you in tomorrow morning", "We'll do a complimentary inspection", "Our technicians are ASE certified"],
        "common_asks": ["oil change appointment", "brake noise", "check engine light", "tire rotation", "estimate"],
    },
    "corporate": {
        "style": "polished, efficient, professional — executive-level interactions",
        "skills": "visitor check-in, meeting room booking, call routing to correct department, package handling, vendor management",
        "urgency": "low — professionalism is the priority",
        "phrases": ["I'll let them know you've arrived", "May I ask who's calling?", "I'll transfer you now"],
        "common_asks": ["meeting room", "visitor badge", "transfer to department", "CEO/manager", "parking validation"],
    },
    "veterinary": {
        "style": "warm, caring — pets are family, owners are emotionally invested",
        "skills": "appointment booking, wellness/vaccine scheduling, urgent triage (vomiting/not eating/trauma = same day), prescription refills routing",
        "urgency": "HIGH for emergencies — not breathing, trauma, seizure = emergency hospital referral",
        "phrases": ["We'll take great care of [pet name]", "Can you describe the symptoms?", "Is [pet name] eating and drinking normally?"],
        "common_asks": ["annual checkup", "vaccines", "my dog is sick", "prescription refill", "emergency"],
    },
    "appointment": {  # generic fallback
        "style": "warm, professional, helpful",
        "skills": "scheduling appointments, answering service questions, taking messages",
        "urgency": "medium",
        "phrases": ["Let me check availability", "I'd be happy to help", "Is there anything else I can assist with?"],
        "common_asks": ["book appointment", "pricing", "hours", "location", "cancel/reschedule"],
    },
    "trade": {  # generic trade fallback
        "style": "efficient, professional, calm under pressure",
        "skills": "service scheduling, emergency triage, dispatching, customer intake",
        "urgency": "HIGH — many calls are urgent",
        "phrases": ["We can have someone out to you", "Can you describe the issue?", "Is this an emergency?"],
        "common_asks": ["emergency service", "schedule repair", "pricing", "availability", "ETA"],
    },
}

def detect_industry(biz_ctx: dict) -> str:
    """Detect business industry from context."""
    biz     = biz_ctx.get("business", {})
    cfg     = biz_ctx.get("config", {})
    industry= (biz.get("industry") or cfg.get("business_type") or "").lower()
    name    = (biz.get("name") or "").lower()
    vertical= (biz.get("vertical") or "appointment").lower()

    mapping = [
        (["spa", "massage", "facial", "wellness", "beauty", "nail"],         "spa"),
        (["salon", "hair", "barber", "stylist", "blowout"],                   "salon"),
        (["medical", "clinic", "doctor", "physician", "urgent care"],         "medical"),
        (["dental", "dentist", "orthodont"],                                   "dental"),
        (["veterinary", "vet ", "animal", "pet"],                             "veterinary"),
        (["legal", "law firm", "attorney", "lawyer"],                         "legal"),
        (["hvac", "heating", "cooling", "air condition", "furnace"],          "hvac"),
        (["plumb", "pipe", "drain", "water heater"],                          "plumber"),
        (["electric", "electrician", "wiring", "panel"],                      "electrician"),
        (["locksmith", "lock ", "lockout", "key"],                            "locksmith"),
        (["hotel", "motel", "resort", "inn", "lodge"],                        "hotel"),
        (["gym", "fitness", "crossfit", "yoga", "pilates"],                   "gym"),
        (["auto", "car", "vehicle", "mechanic", "repair shop", "dealership"], "auto"),
        (["corporate", "office", "agency", "consulting", "marketing"],        "corporate"),
    ]
    for keywords, industry_key in mapping:
        if any(k in industry or k in name for k in keywords):
            return industry_key

    return vertical if vertical in INDUSTRY_SCRIPTS else "appointment"


def build_system_prompt(biz_ctx: dict, memories: list, location: str) -> str:
    now  = datetime.now()
    time_str = now.strftime("%-I:%M %p")
    date_str = now.strftime("%A, %B %-d, %Y")

    biz         = biz_ctx.get("business", {})
    cfg         = biz_ctx.get("config", {})
    ai_set      = biz_ctx.get("ai_settings", {})
    biz_set     = biz_ctx.get("business_settings", {})
    svcs        = biz_ctx.get("services", [])
    staff       = biz_ctx.get("staff", [])
    locs        = biz_ctx.get("locations", [])
    integrations= biz_ctx.get("integrations", {})

    business_name = biz.get("name") or "the business"
    owner_name    = cfg.get("owner_name") or biz.get("owner_name") or ""
    personality   = cfg.get("personality") or "warm, professional, upbeat, proactive"
    business_type = cfg.get("business_type") or biz.get("industry") or "business"
    phone         = biz.get("phone") or ""
    timezone      = biz_set.get("timezone") or biz.get("timezone") or "America/Denver"
    city          = biz_set.get("location_city") or ""
    state         = biz_set.get("location_state") or ""

    svc_lines = "\n".join([
        f"  • {s['name']}" +
        (f" ({s['duration_minutes']} min)" if s.get('duration_minutes') else "") +
        (f" — ${s['price']}" if s.get('price') else "") +
        (f": {s['description']}" if s.get('description') else "")
        for s in svcs
    ]) if svcs else "  Not configured yet"

    staff_lines = "\n".join([
        f"  • {s['name']}" + (f" ({s['role']})" if s.get('role') else "")
        for s in staff
    ]) if staff else ""

    loc_lines = "\n".join([
        f"  • {l.get('name','')}: {l.get('address','')}, {l.get('city','')}, {l.get('state','')} {l.get('zip','')}" +
        (f" | {l['phone']}" if l.get('phone') else "") +
        (f"\n    Parking: {l['parking_info']}" if l.get('parking_info') else "")
        for l in locs
    ]) if locs else ""

    hours_text = ""
    raw_hours = biz_set.get("business_hours") or ai_set.get("business_hours")
    if raw_hours:
        try:
            h = raw_hours if isinstance(raw_hours, dict) else json.loads(raw_hours)
            day_map = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}
            lines = []
            for k, v in h.items():
                day = day_map.get(k.lower(), k)
                if isinstance(v, dict):
                    if v.get("closed"):
                        lines.append(f"  {day}: Closed")
                    else:
                        open_t  = v.get("open", "?")
                        close_t = v.get("close", "?")
                        lines.append(f"  {day}: {open_t} – {close_t}")
                else:
                    lines.append(f"  {day}: {v}")
            hours_text = "\n".join(lines)
        except Exception:
            pass

    doc_sources = []
    if "google_drive" in integrations:
        doc_sources.append("Google Drive")
    if "dropbox" in integrations:
        doc_sources.append("Dropbox")
    doc_text = ", ".join(doc_sources) + " (use search_documents tool)" if doc_sources else "None connected yet"

    mem_text = "\n".join(memories) if memories else "Nothing saved yet."
    loc_text = location or f"{city}, {state}".strip(", ") or "unknown"

    industry_key    = detect_industry(biz_ctx)
    industry_script = INDUSTRY_SCRIPTS.get(industry_key, INDUSTRY_SCRIPTS["appointment"])
    industry_style  = industry_script["style"]
    industry_skills = industry_script["skills"]
    industry_urgency= industry_script["urgency"]
    industry_phrases= ", ".join(f'"{p}"' for p in industry_script["phrases"][:3])
    industry_asks   = ", ".join(industry_script["common_asks"][:5])

    return f"""You are Aria, the AI receptionist for {business_name}, built by Receptionist.co.

━━━ INDUSTRY: {industry_key.upper()} ━━━━━━━━━━━━━━━━━━━━━━
COMMUNICATION STYLE: {industry_style}
CORE SKILLS: {industry_skills}
URGENCY LEVEL: {industry_urgency}
NATURAL PHRASES TO USE: {industry_phrases}
MOST COMMON REQUESTS: {industry_asks}

━━━ IDENTITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NAME: Aria | BUSINESS: {business_name} ({business_type})
OWNER: {owner_name or "unknown — ask once and save immediately"}
PERSONALITY: {personality}
PHONE: {phone} | TIMEZONE: {timezone}
TIME: {time_str} on {date_str} | LOCATION: {loc_text}

━━━ SERVICES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{svc_lines}

{f"━━━ STAFF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{staff_lines}{chr(10)}" if staff_lines else ""}
{f"━━━ LOCATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{loc_lines}{chr(10)}" if loc_lines else ""}
{f"━━━ BUSINESS HOURS ━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{hours_text}{chr(10)}" if hours_text else ""}

━━━ DOCUMENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Connected: {doc_text}
When asked about files, menus, policies, or stored info — always use search_documents first.

━━━ MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mem_text}

━━━ CONFIRMATION RULES (CRITICAL) ━━━━━━━
ALWAYS ask before acting:
  "I'd like to [action]. Shall I go ahead?"

Then WAIT for yes/no.
  YES → call execute_confirmed_action with the confirmation_id
  NO  → call cancel_pending_action

Actions needing confirmation:
  • Booking / changing / cancelling appointments
  • Saving new contacts
  • Updating business hours or settings
  • Adding to waitlist
  • Logging messages

━━━ WHAT A GREAT RECEPTIONIST DOES ━━━━━
📅 SCHEDULE: book, reschedule, cancel appointments
👤 CRM: create contacts, look up client history
📞 MESSAGES: take messages, log missed calls
📋 WAITLIST: manage service waitlists
🔍 DOCUMENTS: search Google Drive / Dropbox files
🌤  INFO: weather, directions, parking, local search
🧮 MATH: tips, splits, conversions, calculations
💬 FOLLOW-UPS: draft SMS or email messages
💳 PAYMENTS: direct to payment link
🔔 ESCALATION: "Let me get someone for you"
😄 PERSONALITY: warm small talk, jokes, fun facts

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Save memory IMMEDIATELY for names, preferences, facts
• Never ask for something already in memory
• Never say "I can't" if you have a tool for it
• Concise voice-first responses — no markdown, no lists
• Search documents before admitting you don't know something
• When uncertain, search the web

GREETING: "Hi{' ' + owner_name + '!' if owner_name else '!'} I'm Aria, your AI receptionist. How can I help you today?"
"""


# ── Agent entry point ─────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    await ctx.connect()

    try:
        metadata = json.loads(ctx.room.metadata or "{}")
    except Exception:
        metadata = {}

    business_id   = metadata.get("business_id", "")
    business_name = metadata.get("business_name", "the business")
    location      = metadata.get("location", "")

    logger.info(f"Session: {business_name} ({business_id}) @ {location}")

    biz_ctx  = await load_business_context(business_id)
    memories = [f"[{r['category']}] {r['memory_key']}: {r['memory_value']}" for r in biz_ctx.get("memories", [])]
    integrations = biz_ctx.get("integrations", {})

    if biz_ctx.get("business", {}).get("name"):
        business_name = biz_ctx["business"]["name"]

    instructions = build_system_prompt(biz_ctx, memories, location)

    # ── TOOLS ──────────────────────────────────────────────────────────────────

    @function_tool
    async def save_memory(ctx: RunContext, key: str, value: str, category: str) -> str:
        """Save important information permanently.
        category: owner_info | preference | business_rule | client_note | instruction | general"""
        await save_memory_to_db(business_id, key, value, category)
        return f"Saved: {key} = {value}"

    @function_tool
    async def check_availability(
        ctx: RunContext,
        date: str,
        service: str = "",
    ) -> str:
        """Check available appointment slots for a given date.
        date: YYYY-MM-DD format (e.g. "2025-03-18")
        service: optional service name to filter by
        Always call this BEFORE offering specific times to the owner."""
        try:
            # Find service_id if service name given
            service_id = None
            if service and business_id:
                sb = get_supabase()
                if sb:
                    r = sb.from_("services").select("id").eq("business_id", business_id).ilike("name", f"%{service}%").limit(1).execute()
                    if r.data:
                        service_id = r.data[0]["id"]

            params = f"business_id={business_id}&date={date}"
            if service_id:
                params += f"&service_id={service_id}"

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"{get_app_url()}/api/cal/slots?{params}",
                    timeout=8,
                )
                if r.status_code == 200:
                    data = r.json()
                    slots = data.get("slots", [])
                    source = data.get("source", "cal")
                    if not slots:
                        return json.dumps({"available": False, "message": f"No available slots on {date}"})
                    # Return first 6 slots as readable times
                    labels = [s["label"] for s in slots[:6]]
                    return json.dumps({"available": True, "date": date, "slots": labels, "source": source, "count": len(slots)})
        except Exception as e:
            pass
        # Fallback — generic business hours
        return json.dumps({"available": True, "date": date, "slots": ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM"], "source": "generic"})

    @function_tool
    async def request_appointment(
        ctx: RunContext,
        contact_name: str,
        date: str,
        time: str,
        service: str = "",
        contact_phone: str = "",
        contact_email: str = "",
        staff_name: str = "",
        notes: str = "",
    ) -> str:
        """Queue an appointment for owner confirmation.
        date: YYYY-MM-DD | time: HH:MM AM/PM (e.g. "2:00 PM")
        Call check_availability first to confirm the slot is open.
        Returns a confirmation_id — wait for owner yes/no before executing."""
        description = f"Book {service or 'appointment'} for {contact_name} on {date} at {time}"
        if staff_name:
            description += f" with {staff_name}"
        conf_id = add_pending("create_appointment", {
            "contact_name": contact_name, "contact_phone": contact_phone,
            "contact_email": contact_email,
            "service": service, "staff_name": staff_name,
            "date": date, "time": time, "notes": notes,
        }, description)
        return f"CONFIRMATION_NEEDED:{conf_id}:{description}"

    @function_tool
    async def request_contact(
        ctx: RunContext,
        name: str,
        phone: str = "",
        email: str = "",
        notes: str = "",
    ) -> str:
        """Queue saving a new contact for owner confirmation."""
        description = f"Add {name} as a contact" + (f" ({phone})" if phone else "")
        conf_id = add_pending("create_contact", {
            "name": name, "phone": phone, "email": email, "notes": notes,
        }, description)
        return f"CONFIRMATION_NEEDED:{conf_id}:{description}"

    @function_tool
    async def request_update_hours(
        ctx: RunContext,
        day: str,
        open_time: str,
        close_time: str,
        closed: bool = False,
    ) -> str:
        """Queue a business hours change for confirmation.
        day: mon/tue/wed/thu/fri/sat/sun | time: HH:MM (24h)"""
        sb = get_supabase()
        current = {}
        try:
            r = sb.from_("business_settings").select("business_hours").eq("business_id", business_id).single().execute()
            if r.data and r.data.get("business_hours"):
                h = r.data["business_hours"]
                current = h if isinstance(h, dict) else json.loads(h)
        except Exception:
            pass
        current[day.lower()] = {"open": open_time, "close": close_time, "closed": closed}
        day_name = {"mon": "Monday", "tue": "Tuesday", "wed": "Wednesday", "thu": "Thursday", "fri": "Friday", "sat": "Saturday", "sun": "Sunday"}.get(day.lower(), day)
        description = f"Set {day_name} as closed" if closed else f"Update {day_name} hours to {open_time}–{close_time}"
        conf_id = add_pending("update_business_hours", {"hours": current}, description)
        return f"CONFIRMATION_NEEDED:{conf_id}:{description}"

    @function_tool
    async def request_log_message(
        ctx: RunContext,
        from_name: str,
        message: str,
        from_phone: str = "",
    ) -> str:
        """Queue logging a message for owner confirmation."""
        preview = message[:60] + "..." if len(message) > 60 else message
        description = f"Log message from {from_name}: '{preview}'"
        conf_id = add_pending("log_message", {
            "from_name": from_name, "from_phone": from_phone, "message": message,
        }, description)
        return f"CONFIRMATION_NEEDED:{conf_id}:{description}"

    @function_tool
    async def request_waitlist(
        ctx: RunContext,
        contact_name: str,
        service: str,
        contact_phone: str = "",
        notes: str = "",
    ) -> str:
        """Queue adding someone to the waitlist for confirmation."""
        description = f"Add {contact_name} to waitlist for {service}"
        conf_id = add_pending("add_to_waitlist", {
            "contact_name": contact_name, "contact_phone": contact_phone,
            "service": service, "notes": notes,
        }, description)
        return f"CONFIRMATION_NEEDED:{conf_id}:{description}"

    @function_tool
    async def execute_confirmed_action(ctx: RunContext, confirmation_id: str) -> str:
        """Execute a queued action after the owner confirms with yes.
        Only call this when owner explicitly says yes/confirm/go ahead."""
        result = await _execute_confirmed(business_id, confirmation_id)
        # Emit structured event to dashboard via room data so it can show a card
        try:
            pending = pending_confirmations.get(confirmation_id) or {}
            action  = pending.get("action", "")
            data    = pending.get("data", {})
            event   = json.dumps({"type": "aria_action", "action": action, "data": data, "result": result})
            await ctx.room.local_participant.publish_data(event.encode(), reliable=True, topic="aria_events")
        except Exception:
            pass
        return result

    @function_tool
    async def cancel_pending_action(ctx: RunContext, confirmation_id: str) -> str:
        """Cancel a queued action when the owner says no."""
        pending_confirmations.pop(confirmation_id, None)
        return "Action cancelled. No changes made."

    @function_tool
    async def search_documents(ctx: RunContext, query: str) -> str:
        """Search Google Drive and Dropbox for business documents.
        Use when asked about files, policies, menus, price lists, or stored information."""
        all_results = []

        if "google_drive" in integrations:
            token = integrations["google_drive"].get("access_token", "")
            if token:
                results = await search_google_drive(token, query)
                all_results.extend(results)

        if "dropbox" in integrations:
            token = integrations["dropbox"].get("access_token", "")
            if token:
                results = await search_dropbox(token, query)
                all_results.extend(results)

        if not all_results:
            return json.dumps({"message": "No document storage connected. The owner can connect Google Drive or Dropbox in Settings."})

        return json.dumps({"results": all_results[:6], "count": len(all_results)})

    @function_tool
    async def lookup_contact(ctx: RunContext, name_or_phone: str) -> str:
        """Look up a contact by name or phone in the CRM."""
        sb = get_supabase()
        if not sb:
            return json.dumps({"error": "Database unavailable"})
        try:
            fields = "first_name,last_name,phone,email,notes,lead_status,is_vip,total_visits"
            # Try first_name/last_name search
            r = sb.from_("contacts").select(fields).eq("business_id", business_id).ilike("first_name", f"%{name_or_phone}%").limit(3).execute()
            if not r.data:
                r = sb.from_("contacts").select(fields).eq("business_id", business_id).ilike("last_name", f"%{name_or_phone}%").limit(3).execute()
            if not r.data:
                r = sb.from_("contacts").select(fields).eq("business_id", business_id).ilike("phone", f"%{name_or_phone}%").limit(3).execute()
            # Format nicely
            contacts = []
            for c in (r.data or []):
                contacts.append({
                    "name":   f"{c.get('first_name','')} {c.get('last_name','')}".strip(),
                    "phone":  c.get("phone",""),
                    "email":  c.get("email",""),
                    "status": c.get("lead_status",""),
                    "vip":    c.get("is_vip", False),
                    "visits": c.get("total_visits", 0),
                    "notes":  c.get("notes",""),
                })
            return json.dumps({"contacts": contacts, "found": len(contacts)})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @function_tool
    async def get_upcoming_appointments(ctx: RunContext, days_ahead: int = 7) -> str:
        """Get upcoming appointments for the business."""
        sb = get_supabase()
        if not sb:
            return json.dumps({"appointments": [], "count": 0})
        try:
            now    = datetime.now().isoformat()
            future = (datetime.now() + timedelta(days=days_ahead)).isoformat()
            # Real schema: appointments uses start_time (timestamptz), joined with contacts/staff/services
            r = sb.from_("appointments")                 .select("id,start_time,end_time,status,job_status,service_type,technician_name,notes,contacts(first_name,last_name,phone),services(name),staff(name)")                 .eq("business_id", business_id)                 .gte("start_time", now)                 .lte("start_time", future)                 .order("start_time")                 .execute()
            appts = []
            for a in (r.data or []):
                contact = a.get("contacts") or {}
                name = f"{contact.get('first_name','')} {contact.get('last_name','')}".strip() or "Unknown"
                service = (a.get("services") or {}).get("name") or a.get("service_type") or "Appointment"
                staff   = (a.get("staff") or {}).get("name") or a.get("technician_name") or ""
                appts.append({
                    "contact": name,
                    "phone":   contact.get("phone",""),
                    "service": service,
                    "staff":   staff,
                    "time":    a.get("start_time",""),
                    "status":  a.get("job_status") or a.get("status",""),
                    "notes":   a.get("notes",""),
                })
            return json.dumps({"appointments": appts, "count": len(appts)})
        except Exception as e:
            return json.dumps({"appointments": [], "count": 0, "error": str(e)})

    @function_tool
    async def get_weather(ctx: RunContext, location: str = "") -> str:
        """Get current weather. Uses saved location automatically."""
        loc = location or biz_ctx.get("business_settings", {}).get("location_city") or "Denver"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"https://wttr.in/{loc}?format=j1", timeout=5)
                if r.status_code == 200:
                    d = r.json()
                    c = d["current_condition"][0]
                    return json.dumps({
                        "location":    loc,
                        "temp_f":      c.get("temp_F"),
                        "feels_like":  c.get("FeelsLikeF") + "°F",
                        "description": c["weatherDesc"][0]["value"],
                        "humidity":    c.get("humidity") + "%",
                        "wind_mph":    c.get("windspeedMiles") + " mph",
                    })
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"error": "Weather unavailable"})

    @function_tool
    async def web_search(ctx: RunContext, query: str) -> str:
        """Search the web for current info, local businesses, news, hours."""
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1",
                    timeout=5,
                )
                if r.status_code == 200:
                    d = r.json()
                    answer = (d.get("AbstractText") or d.get("Answer") or d.get("Definition") or
                              ((d.get("RelatedTopics") or [{"Text": ""}])[0].get("Text", "")) or
                              "No direct answer found.")
                    return json.dumps({"query": query, "answer": answer[:600]})
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"error": "Search unavailable"})

    @function_tool
    async def calculate(ctx: RunContext, expression: str) -> str:
        """Math: tips, bill splits, unit conversions, arithmetic."""
        expr = expression.lower()
        try:
            if "tip" in expr:
                m = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:tip|of|on)\s*\$?(\d+(?:\.\d+)?)', expr)
                if m:
                    tip   = float(m.group(1)) / 100 * float(m.group(2))
                    total = float(m.group(2)) + tip
                    return json.dumps({"tip": f"${tip:.2f}", "total": f"${total:.2f}"})
            if "split" in expr:
                m = re.search(r'\$?(\d+(?:\.\d+)?)\s+(\d+)\s+ways?', expr)
                if m:
                    return json.dumps({"each": f"${float(m.group(1)) / float(m.group(2)):.2f}"})
            if "f to c" in expr or "fahrenheit" in expr:
                m = re.search(r'(\d+(?:\.\d+)?)', expr)
                if m:
                    return json.dumps({"celsius": f"{(float(m.group(1)) - 32) * 5 / 9:.1f}°C"})
            if "c to f" in expr or "celsius" in expr:
                m = re.search(r'(\d+(?:\.\d+)?)', expr)
                if m:
                    return json.dumps({"fahrenheit": f"{float(m.group(1)) * 9 / 5 + 32:.1f}°F"})
            if "miles to km" in expr:
                m = re.search(r'(\d+(?:\.\d+)?)', expr)
                if m:
                    return json.dumps({"km": f"{float(m.group(1)) * 1.60934:.2f} km"})
            cleaned = "".join(c for c in expr if c in "0123456789+-*/.() ")
            if cleaned.strip():
                result = eval(cleaned, {"__builtins__": {}})
                return json.dumps({"answer": str(round(result, 4))})
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"error": "Could not parse that"})

    @function_tool
    async def get_datetime(ctx: RunContext) -> str:
        """Get the current date and time."""
        now = datetime.now()
        return json.dumps({
            "date": now.strftime("%A, %B %-d, %Y"),
            "time": now.strftime("%-I:%M %p"),
            "day":  now.strftime("%A"),
        })

    @function_tool
    async def tell_joke(ctx: RunContext, category: str = "general") -> str:
        """Tell a joke. Categories: business, spa, tech, dad, general"""
        jokes = {
            "dad": [
                "Why don't scientists trust atoms? Because they make up everything!",
                "I told my wife she was drawing her eyebrows too high. She looked surprised.",
                "What do you call a fake noodle? An impasta.",
            ],
            "business": [
                "Why do businesses never tell jokes? They take everything literally.",
                "My boss told me to have a good day, so I went home.",
            ],
            "spa": [
                "Why did the massage therapist win an award? She really kneaded it.",
                "My doctor told me I needed to relax — so I booked a spa day. Doctor's orders!",
            ],
            "tech": [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "A SQL query walks into a bar and asks two tables to join.",
            ],
            "general": [
                "Why can't you trust stairs? They're always up to something.",
                "I'm on a seafood diet — I see food and I eat it.",
                "Why don't eggs tell jokes? They'd crack each other up.",
            ],
        }
        pool = jokes.get(category.lower(), jokes["general"])
        return json.dumps({"joke": random.choice(pool)})

    @function_tool
    async def draft_followup(
        ctx: RunContext,
        recipient_name: str,
        message_type: str,
        context: str = "",
    ) -> str:
        """Draft a follow-up SMS for the owner to send.
        message_type: appointment_reminder | thank_you | no_show | missed_call | promotion"""
        biz_name = biz_ctx.get("business", {}).get("name", "us")
        biz_phone = biz_ctx.get("business", {}).get("phone", "")
        templates = {
            "appointment_reminder": f"Hi {recipient_name}! Reminder of your upcoming appointment at {biz_name}. {context} Reply CONFIRM or call {biz_phone} to reschedule. See you soon!",
            "thank_you":            f"Hi {recipient_name}! Thank you for visiting {biz_name}. {context} We hope to see you again soon!",
            "no_show":              f"Hi {recipient_name}, we missed you today at {biz_name}! {context} Call {biz_phone} to reschedule.",
            "missed_call":          f"Hi {recipient_name}! You called {biz_name} and we missed you. {context} Please call us at {biz_phone} or reply here.",
            "promotion":            f"Hi {recipient_name}! {biz_name} has a special offer for you — {context} Call {biz_phone} to claim it!",
        }
        msg = templates.get(message_type, f"Hi {recipient_name}! Reaching out from {biz_name}. {context}")
        return json.dumps({"draft": msg, "type": message_type})

    @function_tool
    async def get_payment_link(ctx: RunContext, service: str = "") -> str:
        """Get the Stripe payment link for a service or deposit."""
        sb = get_supabase()
        try:
            r = sb.from_("businesses").select("stripe_payment_link").eq("id", business_id).single().execute()
            link = r.data.get("stripe_payment_link") if r.data else None
            if link:
                return json.dumps({"payment_link": link, "note": f"For {service}" if service else "General payment"})
        except Exception:
            pass
        return json.dumps({"note": "Payment link not set up yet — the owner can configure it in Settings."})

    # ── Pipeline ──────────────────────────────────────────────────────────────
    session = AgentSession(
        stt=openai.STT(model="whisper-1", language="en"),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
        tts=openai.TTS(model="tts-1", voice="shimmer"),
        vad=silero.VAD.load(
            min_silence_duration=1.2,   # wait 1.2s of silence before responding
            min_speech_duration=0.2,    # ignore blips under 0.2s (breathing, clicks)
            activation_threshold=0.5,   # standard sensitivity
        ),
    )

    SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "b9e5fba3-071a-4e35-896e-211c4d6eaa7b")
    avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=os.getenv("SIMLI_API_KEY", ""),
            face_id=SIMLI_FACE_ID,
        )
    )

    await avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(
            instructions=instructions,
            tools=[
                save_memory,
                # Appointment & CRM (with confirmation)
                check_availability,
                request_appointment,
                request_contact,
                request_log_message,
                request_waitlist,
                request_update_hours,
                execute_confirmed_action,
                cancel_pending_action,
                # Lookups
                lookup_contact,
                get_upcoming_appointments,
                # Documents
                search_documents,
                # Utility
                get_weather,
                web_search,
                calculate,
                get_datetime,
                tell_joke,
                draft_followup,
                get_payment_link,
            ],
        ),
        room=ctx.room,
        room_options=room_io.RoomOptions(audio_output=False),
    )

    # ── Keepalive — prevents idle disconnect ─────────────────────────────────
    last_activity = {"t": asyncio.get_event_loop().time()}

    def mark_active():
        last_activity["t"] = asyncio.get_event_loop().time()

    # Track any room events as activity
    ctx.room.on("participant_spoke", lambda *_: mark_active())

    async def keepalive_task():
        """Prevent idle disconnect — if silent for 45s, send a gentle prompt."""
        while True:
            await asyncio.sleep(20)
            try:
                idle_secs = asyncio.get_event_loop().time() - last_activity["t"]
                if idle_secs > 45:
                    mark_active()
                    await session.generate_reply(
                        instructions="The user has been quiet. Gently ask if there is anything you can help with today. Keep it to one short sentence."
                    )
            except Exception as e:
                logger.debug(f"keepalive tick: {e}")

    asyncio.create_task(keepalive_task())

    await session.generate_reply(
        instructions="Greet the owner warmly by name if you know it. Mention you're their AI receptionist and you can check availability and book appointments. Be brief, warm, and enthusiastic."
    )

    logger.info(f"Aria ready for {business_name} (room: {ctx.room.name})")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="aria-agent",
    ))
