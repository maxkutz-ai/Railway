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
try:
    from livekit.plugins import deepgram as deepgram_plugin
    DEEPGRAM_AVAILABLE = True
except ImportError:
    DEEPGRAM_AVAILABLE = False

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

        biz_set = sb.from_("settings_business").select("*").eq("business_id", business_id).single().execute()
        if biz_set.data:
            results["business_settings"] = biz_set.data

        svcs = sb.from_("services").select("name,duration_minutes,price,category,description").eq("business_id", business_id).eq("is_active", True).execute()
        if svcs.data:
            results["services"] = svcs.data

        staff = sb.from_("staff").select("name,role,email").eq("business_id", business_id).eq("is_active", True).execute()
        if staff.data:
            results["staff"] = staff.data

        locs = sb.from_("locations").select("name,address,phone").eq("business_id", business_id).execute()
        if locs.data:
            results["locations"] = locs.data

        # All memories (facts + conversation history)
        mems = sb.from_("ai_memory").select("category,memory_key,memory_value,created_at").eq("business_id", business_id).order("created_at", desc=True).limit(150).execute()
        if mems.data:
            results["memories"] = mems.data

        # ── CRM activity snapshot (last 7 days) ──────────────────────────
        from datetime import timedelta
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()

        try:
            appts = sb.from_("appointments").select("start_time,service_type,status,contacts(first_name,last_name)").eq("business_id", business_id).gte("start_time", week_ago).order("start_time", desc=True).limit(10).execute()
            results["recent_appointments"] = appts.data or []
        except Exception: results["recent_appointments"] = []

        try:
            calls = sb.from_("calls").select("started_at,outcome,from_number,contacts(first_name)").eq("business_id", business_id).order("started_at", desc=True).limit(10).execute()
            results["recent_calls"] = calls.data or []
        except Exception: results["recent_calls"] = []

        try:
            msgs = sb.from_("messages").select("sent_at,message_body,direction,contacts(first_name)").eq("business_id", business_id).order("sent_at", desc=True).limit(5).execute()
            results["recent_messages"] = msgs.data or []
        except Exception: results["recent_messages"] = []

        try:
            contacts_count = sb.from_("contacts").select("id", count="exact").eq("business_id", business_id).execute()
            results["contacts_count"] = contacts_count.count or 0
        except Exception: results["contacts_count"] = 0

        try:
            missed = sb.from_("calls").select("id", count="exact").eq("business_id", business_id).eq("outcome", "missed").execute()
            results["missed_calls_total"] = missed.count or 0
        except Exception: results["missed_calls_total"] = 0

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
    if not business_id or len(business_id) < 10:
        logger.warning(f"save_memory skipped — invalid business_id: '{business_id}'")
        return
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
    if not business_id or len(business_id) < 10:
        return "Cannot complete action — session not fully loaded. Please try again."
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
            hours_val = data.get("hours")
            logger.info(f"Saving business hours for {business_id}: {hours_val}")
            try:
                sb.from_("business_settings").upsert({
                    "business_id":    business_id,
                    "business_hours": hours_val,
                }, on_conflict="business_id").execute()
            except Exception as e1:
                logger.warning(f"business_settings upsert failed: {e1}")
            try:
                sb.from_("ai_settings").upsert({
                    "business_id":    business_id,
                    "business_hours": hours_val,
                }, on_conflict="business_id").execute()
            except Exception as e2:
                logger.warning(f"ai_settings upsert failed: {e2}")
            if hours_val:
                return f"Got it! Business hours saved: {hours_val}"
            else:
                return "I didn't receive the hours data. Please tell me your hours again."

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
        "style": "luxurious, calming, warm — like a 5-star spa concierge. Never rushed.",
        "skills": "booking massages/facials/nails/body treatments, upselling packages, gift cards, membership sales, pre/post care advice, cancellation policy",
        "urgency": "low — relaxation focused, unhurried energy always",
        "phrases": ["You deserve this", "Let me check availability for you", "Our therapists specialize in"],
        "common_asks": ["book appointment", "prices", "what services do you offer", "gift cards", "parking", "cancellation policy"],
        "setup_checklist": ["business_hours", "services_with_prices", "cancellation_policy", "booking_lead_time", "staff_names", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your hours? For example, Monday through Friday 9 to 6, Saturday 10 to 4."),
            ("services_with_prices", "What services do you offer? Walk me through them — name, duration, and price if you know it."),
            ("cancellation_policy", "What's your cancellation policy? Do you charge for late cancellations or no-shows?"),
            ("booking_lead_time", "How far in advance can clients book? And is there a minimum notice for same-day bookings?"),
            ("staff_names", "Do you have multiple therapists or estheticians I should know about?"),
        ],
        "crm_briefing": "Tell me about missed calls first, then today's bookings. Mention any unread messages.",
    },
    "salon": {
        "style": "trendy, warm, excited about beauty — mirror the client's energy, use their first name",
        "skills": "booking cuts/color/blowouts, stylist availability, color consultation prompts, retail product recommendations, gift cards, wedding packages",
        "urgency": "low — style emergencies (wedding tomorrow) = squeeze in if possible",
        "phrases": ["You're going to love it!", "Let me check their availability", "Do you have a color reference photo?"],
        "common_asks": ["book haircut", "color appointment", "balayage price", "extensions", "wedding packages"],
        "setup_checklist": ["business_hours", "services_with_prices", "staff_names", "booking_rules", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your salon hours?"),
            ("services_with_prices", "What services do you offer? I'll need names and prices — cuts, colors, treatments, anything you have."),
            ("staff_names", "Who are your stylists? I like to book clients with specific people when they ask."),
            ("booking_rules", "Do clients book with specific stylists, or whoever's available? Any deposit required?"),
        ],
        "crm_briefing": "Missed calls and today's appointments.",
    },
    "medical": {
        "style": "calm, professional, reassuring — HIPAA aware, never share patient info with third parties",
        "skills": "scheduling appointments, insurance verification prompts, co-pay reminders, referrals, urgent triage, new patient intake",
        "urgency": "medium-high — chest pain/difficulty breathing = call 911 immediately, always",
        "phrases": ["I'll have someone call you back shortly", "The doctor will review that", "For your privacy"],
        "common_asks": ["schedule appointment", "refill prescription", "test results", "insurance accepted", "urgent care vs ER"],
        "setup_checklist": ["business_hours", "provider_names", "insurance_accepted", "appointment_types", "new_patient_process", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your clinic hours, including any after-hours or on-call availability?"),
            ("provider_names", "Who are the providers — doctors, NPs, PAs — I should know about?"),
            ("insurance_accepted", "What insurance plans do you accept? I'll need to mention this to callers."),
            ("appointment_types", "What types of appointments do you schedule? Annual physicals, sick visits, follow-ups, procedures?"),
            ("new_patient_process", "What's the process for new patients — do they need a referral, forms to fill out, anything specific?"),
        ],
        "crm_briefing": "Any urgent messages or missed calls first, then today's schedule.",
    },
    "dental": {
        "style": "friendly, reassuring — patients are often anxious, make them feel safe and at ease immediately",
        "skills": "scheduling cleanings/fillings/emergencies/ortho, insurance verification, payment plan info, post-procedure care instructions, emergency triage",
        "urgency": "medium — dental emergencies (broken tooth, severe pain, swelling) = same-day slot priority",
        "phrases": ["We'll take great care of you", "Dr. [name] has an opening", "We accept most PPO plans"],
        "common_asks": ["book cleaning", "emergency tooth pain", "insurance accepted", "cost of filling", "pediatric dentist"],
        "setup_checklist": ["business_hours", "provider_names", "insurance_accepted", "services_with_prices", "emergency_policy", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your office hours? Any evening or Saturday availability?"),
            ("provider_names", "Who are the dentists and any specialists I should know about?"),
            ("insurance_accepted", "What insurance do you accept? PPO, HMO, any specific carriers?"),
            ("services_with_prices", "What's your new patient exam and cleaning fee for uninsured patients?"),
            ("emergency_policy", "How do you handle dental emergencies — same-day slots, after-hours line?"),
        ],
        "crm_briefing": "Any emergency messages or missed calls first, then today's appointments.",
    },
    "hvac": {
        "style": "efficient, calm under pressure — callers are stressed (no heat/AC), be the steady voice that fixes things",
        "skills": "emergency triage (no heat in winter = priority 1), scheduling tune-ups/repairs/installations, dispatching techs, ETAs, maintenance plan upsell, seasonal reminders",
        "urgency": "HIGH — furnace out in winter, AC out in summer = emergency dispatch immediately",
        "phrases": ["We'll get someone out to you today", "Can you describe what the unit is doing?", "Is anyone vulnerable to the heat or cold?"],
        "common_asks": ["heater not working", "AC broken", "strange noise from unit", "annual tune-up", "new system quote", "maintenance plan"],
        "setup_checklist": ["business_hours", "emergency_hours", "service_area", "services_offered", "pricing_model", "dispatch_number", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your regular business hours?"),
            ("emergency_hours", "Do you offer 24/7 emergency service? What's the emergency dispatch number or process?"),
            ("service_area", "What zip codes or cities do you service? I'll need to tell callers if you cover their area."),
            ("services_offered", "What services do you offer — repairs, installs, tune-ups, air quality, duct cleaning?"),
            ("pricing_model", "How do you charge — flat rate, hourly, diagnostic fee? Anything I should mention upfront?"),
        ],
        "crm_briefing": "Any emergency calls or messages first, then today's scheduled jobs.",
    },
    "plumber": {
        "style": "fast, calm, reassuring — burst pipe callers are panicking, be immediate and action-oriented",
        "skills": "emergency dispatch (burst pipes, flooding = NOW), triage water damage severity, scheduling drain/fixture/water heater jobs, shutoff valve guidance",
        "urgency": "HIGH — active flooding/burst pipe = emergency dispatch, always tell them to shut off water first",
        "phrases": ["First — shut off your main water valve", "We have a tech available now", "Can you tell me where the water is coming from?"],
        "common_asks": ["pipe burst", "drain clogged", "water heater out", "toilet overflow", "leak under sink", "emergency"],
        "setup_checklist": ["business_hours", "emergency_hours", "service_area", "services_offered", "pricing_model", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your regular hours and emergency availability?"),
            ("service_area", "What areas do you cover?"),
            ("services_offered", "What plumbing services do you offer — drains, water heaters, fixtures, repiping, emergencies?"),
            ("pricing_model", "How do you charge — service call fee, hourly, flat rate per job?"),
        ],
        "crm_briefing": "Emergency calls and missed calls first.",
    },
    "web_agency": {
        "style": "creative, knowledgeable, consultative — speak like a smart agency partner, not a salesperson",
        "skills": "web design consultations, project scoping, quoting timelines/budgets, portfolio sharing, SEO/hosting questions, client onboarding, revision process, maintenance plans",
        "urgency": "low-medium — new project leads should be captured quickly before they go elsewhere",
        "phrases": ["We'd love to work on that with you", "Let me get some details so we can put together a proposal", "What's the timeline you're working with?"],
        "common_asks": ["website redesign", "how much does a website cost", "ecommerce site", "SEO services", "how long does it take", "portfolio examples", "maintenance plan"],
        "setup_checklist": ["business_hours", "services_with_prices", "portfolio_url", "project_types", "contact_intake_questions", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your business hours for client calls?"),
            ("services_with_prices", "What services do you offer? Web design, branding, SEO, hosting, maintenance? Any starting price ranges?"),
            ("project_types", "What types of clients and projects do you specialize in — small business, ecommerce, restaurants, medical?"),
            ("contact_intake_questions", "When a new lead calls, what do you need to know? Budget, timeline, type of site, existing domain?"),
            ("portfolio_url", "Do you have a portfolio URL I can direct people to while they wait for a callback?"),
        ],
        "crm_briefing": "New leads and missed calls first — every missed call is a potential project.",
    },
    "legal": {
        "style": "formal, discreet, confidential — never discuss case details, always route to attorney",
        "skills": "scheduling consultations, screening caller needs, message routing, document drop-off coordination, intake form guidance",
        "urgency": "varies — criminal/emergency = urgent callback same day",
        "phrases": ["I'll have an attorney call you back", "We treat all matters with strict confidentiality", "May I ask the general nature of your inquiry?"],
        "common_asks": ["free consultation", "how much does it cost", "immigration case", "divorce", "criminal defense", "personal injury"],
        "setup_checklist": ["business_hours", "practice_areas", "consultation_fee", "attorney_names", "intake_questions", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your office hours for client calls?"),
            ("practice_areas", "What areas of law do you practice? I'll need to route callers to the right area."),
            ("consultation_fee", "Do you offer free consultations, or is there a fee?"),
            ("attorney_names", "Who are the attorneys I should know about?"),
            ("intake_questions", "What do you need to know from a new caller — case type, urgency, location?"),
        ],
        "crm_briefing": "Urgent messages and missed calls first, then today's consultations.",
    },
    "hotel": {
        "style": "elegant, welcoming, concierge-level — every guest is a VIP, never say no, find an alternative",
        "skills": "reservations, check-in/out info, amenity questions, local recommendations, complaint handling, room requests, group bookings",
        "urgency": "low-medium — complaints and urgent room issues = prompt gracious response",
        "phrases": ["It would be our pleasure", "Allow me to assist you with that", "Your comfort is our top priority"],
        "common_asks": ["room availability", "check-in time", "parking", "restaurant nearby", "pool hours", "early check-in"],
        "setup_checklist": ["check_in_out_times", "room_types", "amenities", "cancellation_policy", "local_recommendations", "website_scan"],
        "setup_questions": [
            ("check_in_out_times", "What are your check-in and check-out times? Any early check-in policy?"),
            ("room_types", "What room types do you offer? Suite, king, double, any special rooms?"),
            ("amenities", "What amenities do you have — pool, gym, restaurant, spa, parking?"),
            ("cancellation_policy", "What's your cancellation policy?"),
        ],
        "crm_briefing": "Any guest complaints or urgent messages first, then today's arrivals.",
    },
    "veterinary": {
        "style": "warm, caring, empathetic — pets are family, match the owner's concern level",
        "skills": "appointment booking, wellness/vaccine scheduling, urgent triage (not eating/vomiting/trauma = same day), prescription refills routing, boarding inquiries",
        "urgency": "HIGH for emergencies — not breathing, trauma, seizure = emergency animal hospital referral immediately",
        "phrases": ["We'll take great care of them", "Can you describe the symptoms?", "Is your pet eating and drinking normally?"],
        "common_asks": ["annual checkup", "vaccines due", "my dog is sick", "prescription refill", "emergency", "boarding"],
        "setup_checklist": ["business_hours", "emergency_protocol", "services_offered", "provider_names", "species_treated", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your clinic hours? Do you have after-hours emergency coverage?"),
            ("emergency_protocol", "For after-hours emergencies, where do you refer clients?"),
            ("species_treated", "Do you treat dogs and cats only, or other animals too?"),
            ("services_offered", "What services do you offer — wellness exams, surgery, dental, boarding, grooming?"),
        ],
        "crm_briefing": "Any urgent or sick-pet messages first, then today's appointments.",
    },
    "gym": {
        "style": "energetic, motivating, encouraging — match the fitness vibe, be a hype person",
        "skills": "membership sign-ups, class booking, personal trainer scheduling, guest passes, cancellation policy, corporate accounts",
        "urgency": "low — motivate hesitant leads to come in for a free trial",
        "phrases": ["Let's get you started!", "We have a free trial available", "What are your fitness goals?"],
        "common_asks": ["membership price", "class schedule", "personal trainer", "cancel membership", "guest pass", "free trial"],
        "setup_checklist": ["business_hours", "membership_types", "class_schedule", "services_offered", "cancellation_policy", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your gym hours?"),
            ("membership_types", "What membership options do you offer and at what price points?"),
            ("class_schedule", "Do you have group fitness classes? What types and how do people sign up?"),
            ("services_offered", "Any personal training, nutrition coaching, or other add-ons?"),
        ],
        "crm_briefing": "New member inquiries and missed calls first.",
    },
    "auto": {
        "style": "knowledgeable, trustworthy — car owners worry about being overcharged, be transparent",
        "skills": "service scheduling (oil change/brakes/tires), estimate requests, loaner car availability, pickup/drop-off service, warranty questions",
        "urgency": "medium — brake/safety issues = priority same-day",
        "phrases": ["We can get you in tomorrow morning", "We'll do a complimentary inspection", "Our technicians are ASE certified"],
        "common_asks": ["oil change", "brake noise", "check engine light", "tire rotation", "estimate", "loaner car"],
        "setup_checklist": ["business_hours", "services_offered", "makes_serviced", "pricing_model", "loaner_availability", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your shop hours?"),
            ("services_offered", "What services do you offer — oil changes, brakes, tires, diagnostics, transmission?"),
            ("makes_serviced", "Do you specialize in certain makes, or service all vehicles?"),
            ("pricing_model", "Any standard pricing I should know — oil change price, diagnostic fee?"),
        ],
        "crm_briefing": "Today's scheduled jobs and any urgent messages.",
    },
    "electrician": {
        "style": "safety-first, clear, professional — electrical hazards need immediate calm triage",
        "skills": "safety triage (sparks/burning smell = evacuate now), scheduling panel upgrades/EV chargers/lighting, emergency outage response",
        "urgency": "HIGH — sparks, burning smell, no power = safety emergency, evacuate first",
        "phrases": ["If you see sparks or smell burning — leave the building now and call 911", "I'm dispatching our on-call electrician", "Can you safely reach the breaker panel?"],
        "common_asks": ["power outage", "breaker keeps tripping", "EV charger install", "panel upgrade", "outdoor lighting"],
        "setup_checklist": ["business_hours", "emergency_hours", "service_area", "services_offered", "pricing_model", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your regular hours and emergency availability?"),
            ("service_area", "What areas do you cover?"),
            ("services_offered", "What electrical services do you offer?"),
        ],
        "crm_briefing": "Emergency calls and missed calls first.",
    },
    "corporate": {
        "style": "polished, efficient, professional — executive-level interactions at all times",
        "skills": "visitor check-in, meeting room booking, call routing to correct department, package handling, vendor management",
        "urgency": "low — professionalism is always the priority",
        "phrases": ["I'll let them know you've arrived", "May I ask who's calling?", "I'll transfer you now"],
        "common_asks": ["meeting room", "visitor arrival", "transfer to department", "executive team", "parking validation"],
        "setup_checklist": ["business_hours", "departments_and_extensions", "visitor_process", "meeting_rooms", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your office hours?"),
            ("departments_and_extensions", "What departments and key contacts should I know how to route calls to?"),
            ("visitor_process", "What's the visitor check-in process?"),
        ],
        "crm_briefing": "Today's scheduled meetings and visitor arrivals.",
    },
    "appointment": {  # generic fallback
        "style": "warm, professional, helpful",
        "skills": "scheduling appointments, answering service questions, taking messages",
        "urgency": "medium",
        "phrases": ["Let me check availability", "I'd be happy to help", "I'll make a note of that"],
        "common_asks": ["book appointment", "pricing", "hours", "location", "cancel or reschedule"],
        "setup_checklist": ["business_hours", "services_with_prices", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your business hours?"),
            ("services_with_prices", "What services do you offer and at what prices?"),
        ],
        "crm_briefing": "Missed calls and today's appointments.",
    },
    "trade": {  # generic trade fallback
        "style": "efficient, professional, calm under pressure",
        "skills": "service scheduling, emergency triage, dispatching, customer intake",
        "urgency": "HIGH — many calls are urgent",
        "phrases": ["We can have someone out to you", "Can you describe the issue?", "Is this an emergency?"],
        "common_asks": ["emergency service", "schedule repair", "pricing", "availability", "ETA"],
        "setup_checklist": ["business_hours", "emergency_hours", "service_area", "services_offered", "website_scan"],
        "setup_questions": [
            ("business_hours", "What are your hours including emergency availability?"),
            ("service_area", "What areas do you cover?"),
            ("services_offered", "What services do you offer?"),
        ],
        "crm_briefing": "Emergency calls and missed calls first.",
    },
}

def detect_industry(biz_ctx: dict) -> str:
    """Detect business industry from context."""
    biz      = biz_ctx.get("business", {})
    cfg      = biz_ctx.get("config", {})
    industry = (biz.get("industry") or cfg.get("business_type") or "").lower()
    name     = (biz.get("name") or "").lower()
    vertical = (biz.get("vertical") or "appointment").lower()

    mapping = [
        (["spa", "massage", "facial", "wellness", "beauty", "nail"],                    "spa"),
        (["salon", "hair", "barber", "stylist", "blowout"],                              "salon"),
        (["medical", "clinic", "doctor", "physician", "urgent care", "family health"],   "medical"),
        (["dental", "dentist", "orthodont"],                                              "dental"),
        (["veterinary", "vet ", "animal", "pet"],                                        "veterinary"),
        (["legal", "law firm", "attorney", "lawyer"],                                    "legal"),
        (["hvac", "heating", "cooling", "air condition", "furnace"],                     "hvac"),
        (["plumb", "pipe", "drain", "water heater"],                                     "plumber"),
        (["electric", "electrician", "wiring", "panel"],                                 "electrician"),
        (["locksmith", "lock ", "lockout", "key"],                                       "locksmith"),
        (["hotel", "motel", "resort", "inn", "lodge"],                                   "hotel"),
        (["gym", "fitness", "crossfit", "yoga", "pilates"],                              "gym"),
        (["auto", "car", "vehicle", "mechanic", "repair shop", "dealership"],            "auto"),
        (["web design", "web agency", "website", "digital agency", "officeart",
          "seo agency", "marketing agency", "branding", "graphic design"],               "web_agency"),
        (["corporate", "office", "agency", "consulting", "coworking"],                   "corporate"),
    ]
    for keywords, industry_key in mapping:
        if any(k in industry or k in name for k in keywords):
            return industry_key

    return vertical if vertical in INDUSTRY_SCRIPTS else "appointment"


def _build_session_block(biz_ctx: dict, video_count: int, industry_key: str, industry_script: dict, svcs: list, hours_text: str) -> str:
    """Build the session-aware intelligence block injected into every prompt."""
    checklist    = industry_script.get("setup_checklist", [])
    setup_qs     = industry_script.get("setup_questions", [])
    crm_briefing = industry_script.get("crm_briefing", "Missed calls and today's appointments.")

    # ── Determine what's set up vs missing ───────────────────────────────────
    memories = biz_ctx.get("memories", [])
    mem_keys = {m["memory_key"] for m in memories}

    has_hours    = bool(hours_text.strip())
    has_services = len(svcs) > 0
    has_website  = any("website_url" in k or "last_scan" in k for k in mem_keys)
    has_staff    = len(biz_ctx.get("staff", [])) > 0

    missing = []
    for item in checklist:
        if item == "business_hours"         and not has_hours:    missing.append("business hours")
        elif item == "services_with_prices" and not has_services: missing.append("services and prices")
        elif item == "website_scan"         and not has_website:  missing.append("website scan")
        elif item == "staff_names"          and not has_staff:    missing.append("staff names")
        elif item not in ("business_hours","services_with_prices","website_scan","staff_names"):
            if not any(item.replace("_"," ") in k.replace("_"," ") for k in mem_keys):
                missing.append(item.replace("_", " "))
    missing = missing[:4]

    done = [c.replace("_"," ") for c in checklist if c.replace("_"," ") not in missing]

    # ── CRM activity summary ──────────────────────────────────────────────────
    recent_appts = biz_ctx.get("recent_appointments", [])
    recent_calls = biz_ctx.get("recent_calls", [])
    recent_msgs  = biz_ctx.get("recent_messages", [])
    contacts_ct  = biz_ctx.get("contacts_count", 0)
    missed_total = biz_ctx.get("missed_calls_total", 0)

    today_appts = []
    for a in recent_appts:
        try:
            if datetime.fromisoformat(a.get("start_time","")).date() == datetime.now().date():
                c    = a.get("contacts") or {}
                name = f"{c.get('first_name','')} {c.get('last_name','')}".strip() or "Client"
                t    = datetime.fromisoformat(a["start_time"]).strftime("%-I:%M %p")
                today_appts.append(f"{name} at {t} ({a.get('service_type','appointment')})")
        except Exception:
            pass

    missed_today = [c for c in recent_calls if c.get("outcome") == "missed"]
    missed_names = []
    for c in missed_today[:3]:
        con = c.get("contacts") or {}
        missed_names.append(con.get("first_name") or c.get("from_number") or "Unknown")

    unread_msgs = [m for m in recent_msgs if m.get("direction") == "inbound"]

    crm_lines = []
    if today_appts:
        crm_lines.append(f"Today's appointments ({len(today_appts)}): " + " | ".join(today_appts[:4]))
    if missed_names:
        crm_lines.append(f"Missed calls today: {', '.join(missed_names)}")
    if missed_total > 0:
        crm_lines.append(f"Total missed calls in system: {missed_total}")
    if unread_msgs:
        crm_lines.append(f"Unread messages: {len(unread_msgs)}")
    if contacts_ct:
        crm_lines.append(f"Total contacts in CRM: {contacts_ct}")

    # ── Last conversation summaries ───────────────────────────────────────────
    conv_memories  = [m for m in memories if m.get("category") == "conversation"]
    conv_summaries = [m["memory_value"][:300] for m in conv_memories[:2]]

    # ── Build block based on session number ───────────────────────────────────
    lines = []

    if video_count == 0:
        # ── SESSION 1 — First ever session ────────────────────────────────────
        lines.append("THIS IS THE FIRST SESSION. Work through this flow:")
        lines.append("1. Greet warmly. Ask their name. Spell it back. Save with save_memory(key='owner_first_name', category='owner_info').")
        lines.append("2. Ask: 'What kind of business do you run?' Save with save_memory(key='business_type_description', category='business_rule').")
        lines.append(f"3. Say: 'I'm configured for {industry_key.replace('_',' ')} businesses — let me ask a few quick questions so I can help your clients effectively.'")
        lines.append("4. Work through these setup questions ONE AT A TIME — ask, wait for full answer, save, then move to next:")
        for i, (key, question) in enumerate(setup_qs[:3], 1):
            lines.append(f"   Q{i}: \"{question}\" → save_memory(key='{key}', category='business_rule')")
        lines.append("5. After setup questions, offer website scan: 'Do you have a website? I can scan it to learn your services and hours automatically.' If yes → call request_website_url(). If no → skip.")
        lines.append("6. Offer dashboard tour: 'Want me to walk you through the main sections of your dashboard?'")
        lines.append("CRITICAL: ONE question at a time. Wait for full answer before asking the next one.")

    elif video_count in (1, 2):
        # ── SESSION 2-3 — Follow-up, give briefing + fill gaps ────────────────
        lines.append(f"SESSION {video_count + 1} — returning owner. Start with a brief status update, then offer to fill gaps.")
        if conv_summaries:
            lines.append(f"LAST SESSION SUMMARY: {conv_summaries[0]}")
        if crm_lines:
            lines.append("CRM STATUS RIGHT NOW:")
            lines.extend(f"  • {l}" for l in crm_lines)
        if missing:
            lines.append(f"SETUP STILL MISSING: {', '.join(missing)}")
            lines.append("SCRIPT: After the status briefing, say: 'A couple of things I'd love to finish setting up — [mention 1 or 2 missing items]. Want to do that now, or is there something else on your mind first?'")
            lines.append("If yes → use add_service() for services, scan_website() for website scan, or save_memory() for hours and policies. One item at a time.")
            lines.append("If no → help with what they need. Offer setup at natural pause.")
        else:
            lines.append("SETUP IS COMPLETE. Focus on operational help — bookings, calls, CRM questions.")
        if done:
            lines.append(f"ALREADY CONFIGURED: {', '.join(done)}")

    else:
        # ── SESSION 4+ — Established, lead with briefing ──────────────────────
        lines.append(f"SESSION {video_count + 1} — established relationship. Open with a 1-sentence briefing, then ask what they need.")
        if conv_summaries:
            lines.append(f"RECENT CONTEXT: {conv_summaries[0]}")
        if crm_lines:
            lines.append(f"BRIEFING TO OPEN WITH ({crm_briefing}):")
            lines.extend(f"  • {l}" for l in crm_lines)
            lines.append("Pick the most important item above and lead with it naturally. Example: 'You have 3 missed calls today including one from Sarah — want me to follow up?'")
        if missing:
            lines.append(f"Still unconfigured: {', '.join(missing)} — mention only if it comes up naturally.")

    return "\n".join(lines)


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

    mem_text = "\n".join(m[:150] for m in memories[:20]) if memories else "Nothing saved yet."

    # Track video session count (for onboarding)
    video_count = 0
    try:
        sb_vc = get_supabase()
        if sb_vc and business_id:
            vc_res = sb_vc.from_("ai_memory").select("memory_value").eq("business_id", business_id).eq("memory_key", "aria_video_count").limit(1).execute()
            if vc_res.data:
                video_count = int(vc_res.data[0]["memory_value"] or 0)
            new_count = video_count + 1
            sb_vc.from_("ai_memory").upsert({
                "business_id": business_id, "category": "system",
                "memory_key": "aria_video_count", "memory_value": str(new_count),
                "created_at": datetime.now().isoformat()
            }, on_conflict="business_id,memory_key").execute()
    except: pass

    # Load scanned knowledge from ai_memory (where /api/knowledge/ingest saves)
    kb_text = ""
    try:
        sb_kb = get_supabase()
        if sb_kb and business_id:
            kb_res = sb_kb.from_("ai_memory").select("memory_key,memory_value").eq("business_id", business_id).in_("category", ["general","business_rule","instruction"]).order("created_at", desc=True).limit(10).execute()
            if kb_res.data:
                snippets = [f"{r['memory_key']}: {r['memory_value'][:200]}" for r in kb_res.data if r.get("memory_value")]
                kb_text = "\n".join(snippets[:5])
    except: pass
    loc_text = location or f"{city}, {state}".strip(", ") or "unknown"

    industry_key    = detect_industry(biz_ctx)
    industry_script = INDUSTRY_SCRIPTS.get(industry_key, INDUSTRY_SCRIPTS["appointment"])
    industry_style  = industry_script["style"]
    industry_skills = industry_script["skills"]
    industry_urgency= industry_script["urgency"]
    industry_phrases= ", ".join(f'"{p}"' for p in industry_script["phrases"][:3])
    industry_asks   = ", ".join(industry_script["common_asks"][:5])

    # Custom instructions override — if the business has a custom prompt, inject it
    custom_instructions = (cfg.get("custom_instructions") or "")[:1500]
    ai_name             = cfg.get("ai_name") or "Aria"
    role_description    = cfg.get("role_description") or ""
    primary_goal        = cfg.get("primary_goal") or ""
    anti_hallucination  = cfg.get("anti_hallucination_rule") or "If uncertain about specific details, say: 'I want to make sure I get this right — let me check and come back to you in a moment.' Then look it up using your tools." 
    turn_taking_strict  = cfg.get("turn_taking_strict", True)
    rush_mode_enabled   = cfg.get("rush_mode_enabled", True)

    # Load all structured instruction fields
    role_description   = cfg.get("role_description") or ""
    primary_goal_text  = cfg.get("primary_goal") or primary_goal or ""
    anti_hallu         = cfg.get("anti_hallucination_rule") or anti_hallucination
    turn_taking        = cfg.get("turn_taking_rules") or ""
    rush_mode          = cfg.get("rush_mode_rules") or ""
    no_loop            = cfg.get("no_loop_rule") or ""
    escalation         = cfg.get("escalation_rules") or ""
    greeting_script    = cfg.get("greeting") or ""
    # Sanitize greeting: if it contains a name, we'll override with memory-verified name
    # Detect if greeting_script has "Hi [Name]" pattern and strip the name
    import re as _re
    if greeting_script:
        _greeting_clean = _re.sub(r"Hi\\s+\\w+!", "Hi!", greeting_script)
        # Only use sanitized version - replace name with dynamic name at runtime
        greeting_script = _greeting_clean
    flow_script        = cfg.get("flow_script") or ""

    # Build structured custom block — only include non-empty sections
    def section(title: str, body: str) -> str:
        return f"\n━━━ {title} ━━━━━━━━━━━━━━━━━━━━\n{body.strip()}\n" if body.strip() else ""

    custom_block = ""
    if any([role_description, primary_goal_text, anti_hallu, turn_taking, rush_mode,
            no_loop, escalation, custom_instructions.strip(), flow_script]):
        custom_block = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        custom_block += "\n⚡ BUSINESS-SPECIFIC INSTRUCTIONS — FOLLOW EXACTLY"
        custom_block += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        if role_description:
            custom_block += section("ROLE & IDENTITY", role_description)
        if primary_goal_text:
            custom_block += section("PRIMARY GOAL", primary_goal_text)
        if anti_hallu:
            custom_block += section("ANTI-HALLUCINATION RULE", anti_hallu)
        if turn_taking:
            custom_block += section("TURN-TAKING / WAIT RULE (CRITICAL)", turn_taking)
        if rush_mode:
            custom_block += section("RUSH MODE (CRITICAL)", rush_mode)
        if no_loop:
            custom_block += section("NO-LOOP RULE", no_loop)
        if escalation:
            custom_block += section("ESCALATION RULES", escalation)
        if custom_instructions.strip():
            custom_block += section("ADDITIONAL RULES & KNOWLEDGE", custom_instructions)
        if flow_script.strip():
            custom_block += section("MANDATORY CONVERSATION FLOW", flow_script)
        if greeting_script.strip():
            custom_block += f"\n━━━ GREETING (SAY EXACTLY) ━━━━━━━━━━━━━━━━━━━━\n\"{greeting_script.strip()}\"\n"
        custom_block += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    return f"""You are {ai_name}, the AI receptionist for {business_name}, built by Receptionist.co.

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
{custom_block}
━━━ ANTI-HALLUCINATION ━━━━━━━━━━━━━━━━━━
{anti_hallucination}

{f"━━━ PRIMARY GOAL ━━━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{primary_goal}{chr(10)}" if primary_goal else ""}━━━ SERVICES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{svc_lines}

{f"━━━ STAFF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{staff_lines}{chr(10)}" if staff_lines else ""}
{f"━━━ LOCATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{loc_lines}{chr(10)}" if loc_lines else ""}
{f"━━━ BUSINESS HOURS ━━━━━━━━━━━━━━━━━━━━━━{chr(10)}{hours_text}{chr(10)}" if hours_text else ""}

━━━ DOCUMENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━
Connected: {doc_text}
When asked about files, menus, policies, or stored info — always use search_documents first.

━━━ MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mem_text}
{f"━━━ SCANNED WEBSITE KNOWLEDGE ━━━━━━━━━━━━{chr(10)}{kb_text}" if kb_text else ""}

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

━━━ DASHBOARD FEATURES (you can tell the owner about these) ━━━
🎵 MUSIC: The dashboard has a background music player with jazz, classical, and ambient tracks. Tell the owner to use the music controls in the video window to play/stop music. You cannot play it yourself but it's built into their dashboard.
📊 USAGE: The dashboard shows their video minute usage and subscription status.
🔔 ALERTS: Smart alerts appear in the dashboard for missed calls and important events.

━━━ SERVICES & SETTINGS (CRITICAL — READ CAREFULLY) ━━━━━━━━
• Services are managed in the CRM: Settings → AI Receptionist → Services tab.
  - The owner can add, edit, or remove services directly in the dashboard.
  - If they ask "how do I add services?" → say: "Go to Settings in the left sidebar, then the AI Receptionist tab — you can add your services there."
  - NEVER suggest Google Drive or Dropbox for managing services. Drive/Dropbox is ONLY for searching uploaded documents like price lists or menus.
• Business hours → Settings → AI Receptionist → Business Hours tab.
• Contact info (phone, email) → Settings → Business tab.
• Google Drive / Dropbox is ONLY for searching documents the owner has uploaded there. It does NOT manage services, contacts, or CRM data.
• If you can't find something in the dashboard data, say what you DO know and offer to help add it.

━━━ NAVIGATION — HOW TO GUIDE THE OWNER ━━━━━━━━━━━━━━━━━━
When you need to send the owner to a section, call navigate_to_section(section).
If they ask how to do something in the dashboard, give SPECIFIC directions:
  - "Go to Settings in the left sidebar" (not "go to settings section")
  - "Click Contacts in the left menu, then click a contact name"
  - "In Appointments, use the Day/Week/Month tabs at the top"
NEVER say "I'll take you there now" then do nothing. Always call navigate_to_section immediately when you intend to navigate.

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

━━━ TURN-TAKING RULES ━━━━━━━━━━━━━━━━━━━
{"• Ask ONE question at a time. STOP and WAIT for the answer before continuing." if turn_taking_strict else ""}
{"• Never talk over the caller. If they start speaking, stop immediately." if turn_taking_strict else ""}
{"• Wait 1–1.5 seconds of silence before replying — don't jump in too fast." if turn_taking_strict else ""}
{"• If the caller overlaps with you, stop and say: Sorry, please continue." if turn_taking_strict else ""}
{"• Do not say 'Perfect/Great/Thanks' until you have actually heard their answer." if turn_taking_strict else ""}
{"• After a YES/NO answer, pause one beat before continuing." if turn_taking_strict else ""}
{"• RUSH MODE: Triggered by — urgent, emergency, ASAP, today, I gotta run, quick. Collect only name + phone, close fast." if rush_mode_enabled else ""}
{"• NO-LOOP RULE: Never repeat the same question more than twice. After two failed attempts say: Let me connect you with our team right now." if turn_taking_strict else ""}

━━━ HUMAN ESCALATION (ALWAYS ACTIVE) ━━━
• If the caller says "human", "agent", "representative", or presses 0 → offer to have the team call back immediately. Never push back.
• Escalate automatically after 90 seconds of confusion or frustration.
• Script: "Absolutely — let me have our team reach out to you right away. Can I get your best number?"

━━━ EMAIL CAPTURE RULE ━━━━━━━━━━━━━━━━━━
When capturing an email address:
1. Ask for the email clearly.
2. Ask them to spell it out letter by letter.
3. Read it back: "Just to confirm — j-o-h-n dot smith at gmail dot com?"
4. Get an explicit "yes" before saving. If unsure, ask one more time.
This prevents errors that break confirmations and CRM records.

━━━ BREVITY RULES (CRITICAL) ━━━━━━━━━━━━
This is VOICE. People are busy. Every response must be SHORT.

• MAX 2 sentences per response. Usually 1 is enough.
• Answer the question first. Extra context only if essential.
• Never add "Is there anything else I can help with?" — it wastes time.
• Never say "Certainly!", "Absolutely!", "Great question!", "Of course!" — filler.
• Never repeat what the user just said back to them.
• If asked for weather → give temp + condition in one sentence. Done.
• If asked for messages → say how many and who. Done.
• If asked about appointments → say next one, time, who. Done.
• If you can't do something → say what you CAN do instead. One sentence.
• Small talk → one warm sentence max, then ask how you can help.

GOOD: "It's 72 and sunny in Denver." (done — nothing else)
GOOD: "You have 2 appointments today — Jane at 2pm, Tom at 4pm." (done)
BAD:  "Great question! Let me check the weather for you..." (never say this)
BAD:  "I can't do X, but I can help you with Y instead!" (just do Y)
BAD:  "Is there anything else I can help with?" (never add this)

GOOD: "You have 3 missed calls — 2 from unknown numbers, 1 from Jane Smith."
BAD:  "I checked and it looks like you have some missed calls. There are 3 in total..."

━━━ OTHER RULES ━━━━━━━━━━━━━━━━━━━━━━━━━
• Save memory IMMEDIATELY for names, preferences, facts
• Never ask for something already in memory
• Never say "I can't" if you have a tool for it
• No markdown, no bullet points — natural speech only
• WEBSITE SCANNING: You CAN scan websites. Rules:
  - If asked to scan website: say "Sure! Please paste your website URL in the chat below — for example: https://yourbusiness.com" Then WAIT SILENTLY. Do not repeat yourself.
  - When the user sends a URL in chat (it will appear in the transcript starting with http or a domain): call scan_website(website_url=<url>) IMMEDIATELY.
  - If you hear a URL spoken ("southamptonspa dot com"): reconstruct as https://southamptonspa.com and call scan_website immediately.
  - After scan_website completes: report back what you found — pages scanned, hours, phone, email. Then ask owner to confirm.
  - CRITICAL: Say the paste instruction ONCE only. Wait for them to paste. Do not repeat or ask again.
• DASHBOARD COLORS: You CAN change the dashboard theme. If asked about colors/appearance/theme, use set_dashboard_theme() immediately. Options: midnight, deep_slate, true_void, charcoal, obsidian (dark) or snow, mist, cream (light) or blue, purple, green, amber, pink (accent). Say "Done! I've updated your dashboard." Don't ask for a phone number.
• NAME CONFIRMATION: When someone gives you their name, ALWAYS confirm spelling before saving. Say "Got it — is that [name], spelled [spell it out letter by letter]?" Wait for yes before calling save_memory.
• BUSINESS HOURS INPUT: Users can share hours by speaking OR typing in chat. When asking for hours, say: "You can tell me the hours out loud, or paste them here in the chat."
• TIME FORMAT: ALWAYS say times in 12-hour format with AM/PM. Never say "09:00" — say "9 AM". Never say "18:00" — say "6 PM". Convert any 24-hour stored values to 12-hour when speaking.
• Search documents or web before admitting ignorance
• SUPPORT INFO: If anyone asks about support, help desk, or contacting the team: the support email is support@receptionist.co and the website is receptionist.co. Say this directly without checking.

GREETING: {"(\"" + greeting_script + "\")" if greeting_script else ("Hi " + owner_name + "! I'm " + ai_name + ". How can I help?" if owner_name else "Hi! I'm " + ai_name + ". How can I help?")}

SESSION #{str(video_count + 1)} WITH THIS BUSINESS.

━━━ SESSION INTELLIGENCE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{_build_session_block(biz_ctx, video_count, industry_key, industry_script, svcs, hours_text)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""



# WMO weather interpretation codes (module level)
WMO_CODES: dict = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Rain showers", 81: "Heavy showers", 82: "Violent showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
}

# ── Agent entry point ─────────────────────────────────────────────────────────
async def entrypoint(ctx: JobContext):
    logger.info(f"🚀 entrypoint() called — room: {ctx.room.name if ctx.room else 'unknown'}")
    await ctx.connect()
    logger.info(f"✅ Connected to room: {ctx.room.name}")

    try:
        metadata = json.loads(ctx.room.metadata or "{}")
    except Exception:
        metadata = {}

    business_id   = metadata.get("business_id", "")
    business_name = metadata.get("business_name", "the business")
    location      = metadata.get("location", "")
    dashboard_ctx = metadata.get("context", "")  # live status data from dashboard

    logger.info(f"Session: {business_name} ({business_id}) @ {location}")
    if dashboard_ctx:
        logger.info(f"Dashboard context received: {len(dashboard_ctx)} chars")

    biz_ctx  = await load_business_context(business_id)
    memories = [f"[{r['category']}] {r['memory_key']}: {r['memory_value']}" for r in biz_ctx.get("memories", [])]
    integrations = biz_ctx.get("integrations", {})

    if biz_ctx.get("business", {}).get("name"):
        business_name = biz_ctx["business"]["name"]

    instructions = build_system_prompt(biz_ctx, memories, location)

    # Check if in onboarding mode
    is_onboarding = "ONBOARDING_MODE" in dashboard_ctx if dashboard_ctx else False
    onboarding_chapter = 0
    if is_onboarding:
        import re
        m = re.search(r"CHAPTER:(\d+)", dashboard_ctx)
        if m: onboarding_chapter = int(m.group(1))

    # Inject live dashboard context into instructions
    if dashboard_ctx and not is_onboarding:
        instructions = instructions + f"""

━━━ LIVE DASHBOARD DATA (use this to answer status questions) ━━━
{dashboard_ctx}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CRITICAL: You HAVE access to the live data above. When asked about appointments, calls, messages, or status — read directly from the LIVE DASHBOARD DATA section above. 
NEVER say "I don't have access to that information" or "I don't want to guess" — you have the data, use it.
NEVER say "I'll have our team follow up" — EVER. Not for any reason.
NEVER say "I don't have access to your contacts" — the contacts ARE on the dashboard and in context.
NEVER ask for a contact number — you already have it in the dashboard data.
NEVER say "Please hold on a moment" — this causes you to freeze. If you need to look something up, do it silently and respond immediately.
NEVER say "let me confirm that for you" then pause — answer immediately with what you know from the dashboard data.
BUSINESS NAME: The business name is in your system prompt. You know it. Never ask what business the owner manages.
INTERRUPTIONS: If the user interrupts you mid-sentence, acknowledge what they said and answer their question directly. Don't restart your previous sentence — pivot naturally to what they asked.
OWNER NAME RULE: When the owner gives their name, ALWAYS spell it back: "Got it — is that spelled M-A-X?" and wait for confirmation before saving. This prevents mishearing errors. Once confirmed, save with save_memory.
"""

    # Onboarding mode — full guided tour script
    if is_onboarding:
        chapter_scripts = {
            0: """INTRO: Welcome them warmly. "Hi! I'm Aria, your AI receptionist. I'm here 24/7 to manage your calls, appointments, messages, and keep your business running smoothly. This quick tour is completely free. What's your name?" CRITICAL: After they say their name, CONFIRM IT: "Nice to meet you, [name]! Did I get that right?" Wait for yes/correction before saving. Once confirmed, save with save_memory(key="owner_first_name", value=[name], category="owner_info"). Then say: "Great! I'm going to walk you through your new dashboard. Ready to start?" Call navigate_to_section("dashboard") then complete_onboarding_chapter(0).""",
            1: """DASHBOARD: "You're looking at your main Dashboard. This is your command center — it shows today's appointments, recent calls, unread messages, and smart alerts. The colorful circles show your AI performance stats. You can ask me to change the colors or theme anytime — just say something like 'make it darker' or 'switch to a light theme'. The banner at the top shows urgent alerts that need attention." Ask: "Do you have any questions about the dashboard?" Wait for answer. Then call navigate_to_section("contacts") and complete_onboarding_chapter(1).""",
            2: """CONTACTS: "This is your Contacts section — think of it as your smart client database. Every person who calls, messages, or books gets a profile here automatically. You can see their full history — calls, appointments, messages — all in one place. You can tag contacts as VIP, Lead, or Client." Ask: "Any questions about Contacts?" Then call navigate_to_section("appointments") and complete_onboarding_chapter(2).""",
            3: """APPOINTMENTS: "Here's your Appointments section. You can view by Day, Week, or Month. Click any appointment to confirm, cancel, or mark if the client showed up. The search bar and colored filters at the top let you find appointments quickly. I can book appointments for you directly — just ask me." Ask: "Questions about Appointments?" Then call navigate_to_section("messages") and complete_onboarding_chapter(3).""",
            4: """INBOX: "This is your Inbox — all two-way SMS conversations with clients. When someone texts your business number, it appears here. You can reply directly. I also send booking confirmations and reminders automatically. Type your message and hit Send, or ask me to draft a reply for you." Ask: "Questions about Inbox?" Then call navigate_to_section("calls") and complete_onboarding_chapter(4).""",
            5: """CALLS: "The Calls section shows every call to your business number — answered, missed, and handled by me. You can see transcripts, listen to recordings, and see what each caller needed. Missed calls get flagged so you never lose a lead." Ask: "Questions about Calls?" Then call navigate_to_section("pipeline") and complete_onboarding_chapter(5).""",
            6: """PIPELINE: "Your Pipeline is a visual sales board — move contacts through stages like New Inquiry, Qualified, Booked, and Completed. Great for tracking prospects who haven't booked yet. Drag and drop cards between columns." Ask: "Questions about Pipeline?" Then call navigate_to_section("campaigns") and complete_onboarding_chapter(6).""",
            7: """CAMPAIGNS: "Campaigns is your built-in email newsletter tool. Add subscribers, compose emails with the rich editor, and send to your list with one click. Your website embed code is in the Subscribers tab — paste it on your website and visitors can subscribe directly." Ask: "Questions about Campaigns?" Then call navigate_to_section("settings") and complete_onboarding_chapter(7).""",
            8: """SETTINGS: "Finally, Settings. This is where you configure your AI receptionist — give me a personality, set your services, business hours, and knowledge base. You can also manage integrations like Cal.com for scheduling and Twilio for calls. The Appearance section lets you change colors and theme." 
Ask: "Do you have a website? I can scan it right now to learn your services, hours, pricing, and FAQs — so I can answer client questions accurately from day one."
If YES: Ask for the URL, then call scan_website(url). Say "Give me a moment to scan your site..." then report what was found.
If NO or SKIP: "No problem! You can add it later in Settings → Knowledge."
Then ask: "Is there anything else you'd like to know before we wrap up?" Wait for answer.
Then say: "You're all set! I'll rescan your website every day automatically to stay up to date. Click 'Start Using My Dashboard' to begin!" Call finish_onboarding().""",
        }
        current_script = chapter_scripts.get(onboarding_chapter, chapter_scripts[0])
        instructions = instructions + f"""

━━━ ONBOARDING MODE — FREE SESSION (no minutes charged) ━━━━━━
You are guiding {business_name} through their onboarding tour.
Current chapter: {onboarding_chapter}

YOUR SCRIPT FOR THIS CHAPTER:
{current_script}

ONBOARDING RULES:
- Go at the owner's pace. If they have questions, answer them fully.
- After each section, always ask "Do you have any questions?" and wait.
- Be warm, encouraging, and celebratory — this is their new business tool!
- If they want to skip a section: say "No problem!" and move on.
- If they want to pause: say "Of course! Just come back whenever you're ready — we'll pick up right where we left off."
- Use navigate_to_section() to move them between CRM sections.
- Use complete_onboarding_chapter() after each section.
- Use finish_onboarding() only at the very end after Settings.
- This session is FREE — remind them if they ask about minutes.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    logger.info(f"System prompt length: {len(instructions)} chars")

    # ── TOOLS ──────────────────────────────────────────────────────────────────

    @function_tool
    async def save_memory(ctx: RunContext, key: str, value: str, category: str) -> str:
        """Save important information permanently.
        category: owner_info | preference | business_rule | client_note | instruction | general"""
        await save_memory_to_db(business_id, key, value, category)
        return f"Saved: {key} = {value}"

    @function_tool
    async def set_dashboard_theme(ctx: RunContext, theme: str) -> str:
        """Change the dashboard color theme for the owner.
        theme options:
          dark themes: "midnight" | "deep_slate" | "true_void" | "charcoal" | "obsidian"
          light themes: "pure_white" | "snow" | "mist" | "cream"
          accent colors: "blue" | "purple" | "green" | "amber" | "pink"
        Example: set_dashboard_theme("obsidian") or set_dashboard_theme("blue")
        Say: "I've updated your dashboard to [theme] — it will refresh in a moment."
        """
        try:
            sb = get_supabase()
            if sb:
                await sb.table("ai_memory").upsert({
                    "business_id": business_id,
                    "memory_key":  "dashboard_theme_change",
                    "memory_value": theme,
                    "category":    "preference",
                }, on_conflict="business_id,memory_key").execute()
            return f"Dashboard theme change requested: {theme}. The dashboard will update when they refresh."
        except Exception as e:
            return f"Theme request noted: {theme}"

    @function_tool
    async def navigate_to_section(ctx: RunContext, section: str) -> str:
        """Navigate the owner to a specific section of the CRM dashboard.
        section: "dashboard" | "contacts" | "appointments" | "messages" | "calls" | "pipeline" | "analytics" | "campaigns" | "settings"
        Use during onboarding to guide the owner through different sections.
        """
        valid = ["dashboard","contacts","appointments","messages","calls","pipeline","analytics","campaigns","settings"]
        if section not in valid:
            return f"Unknown section: {section}. Valid: {', '.join(valid)}"

        # PRIMARY: publish via LiveKit data channel for instant response
        try:
            import json as _json
            payload = _json.dumps({"type": "navigate", "section": section}).encode()
            await ctx.room.local_participant.publish_data(payload, topic="aria_commands", reliable=True)
            logger.info(f"Navigation published via LiveKit: {section}")
        except Exception as e:
            logger.warning(f"LiveKit navigation publish failed: {e}")

        # FALLBACK: write to Supabase for polling-based pickup
        try:
            sb = get_supabase()
            if sb and business_id:
                sb.from_("ai_memory").upsert({
                    "business_id": business_id,
                    "memory_key":  "aria_navigate_request",
                    "memory_value": section,
                    "category":    "preference",
                }, on_conflict="business_id,memory_key").execute()
        except Exception as e:
            logger.warning(f"Supabase navigate fallback failed: {e}")

        return f"Navigating to {section} section."

    @function_tool
    async def complete_onboarding_chapter(ctx: RunContext, chapter: int) -> str:
        """Mark an onboarding chapter as complete and advance to the next.
        chapters: 0=intro, 1=dashboard, 2=contacts, 3=appointments, 4=inbox, 5=calls, 6=pipeline, 7=campaigns, 8=settings, 9=done
        Call this after finishing each section during onboarding.
        """
        try:
            sb = get_supabase()
            if sb and business_id:
                # upsert so it works even if no onboarding_sessions row exists yet
                sb.from_("onboarding_sessions").upsert({
                    "business_id":    business_id,
                    "current_chapter": chapter + 1,
                    "status":         "in_progress",
                    "last_active_at": datetime.utcnow().isoformat(),
                }, on_conflict="business_id").execute()
            return f"Chapter {chapter} complete. Moving to chapter {chapter + 1}."
        except Exception as e:
            logger.warning(f"complete_onboarding_chapter error: {e}")
            return "Chapter progress noted."

    @function_tool
    async def add_service(ctx: RunContext, name: str, price: float = 0, duration_minutes: int = 60, category: str = "", description: str = "") -> str:
        """Add a service to this business's service menu in the CRM.
        Use this when the owner tells you about a service they offer.
        Call once per service — do not batch multiple services in one call.
        After saving, confirm: "Got it — [name] added to your services."
        """
        try:
            sb = get_supabase()
            if not sb or not business_id:
                return "Session error — please try again."

            # Check for duplicate
            existing = sb.from_("services").select("id,name").eq("business_id", business_id).ilike("name", name.strip()).execute()
            if existing.data:
                return f"'{name}' is already in your services menu."

            sb.from_("services").insert({
                "business_id":      business_id,
                "name":             name.strip(),
                "price":            float(price) if price else None,
                "duration_minutes": int(duration_minutes) if duration_minutes else None,
                "category":         category.strip() or None,
                "description":      description.strip() or None,
                "is_active":        True,
            }).execute()

            # Also save to ai_memory so Aria can reference it in future sessions
            await save_memory_to_db(
                business_id,
                f"service_{name.lower().replace(' ','_')}",
                f"{name}" + (f" — ${price}" if price else "") + (f", {duration_minutes} min" if duration_minutes else "") + (f". {description}" if description else ""),
                "business_rule"
            )

            logger.info(f"Service added: {name} for {business_id}")
            price_str = f" at ${price:.0f}" if price else ""
            return f"'{name}'{price_str} added to your services menu."

        except Exception as e:
            logger.error(f"add_service error: {e}")
            return f"Couldn't save that service: {e}"

    @function_tool
    async def request_website_url(ctx: RunContext) -> str:
        """Call this when the owner asks to scan their website but hasn't provided a URL yet.
        This shows a URL input popup in the dashboard.
        Say: "I've opened a window for you to enter your website URL. Once you submit it, I'll scan it right away!"
        """
        try:
            sb = get_supabase()
            if sb:
                await sb.table("ai_memory").upsert({
                    "business_id": business_id,
                    "memory_key": "aria_request_website_url",
                    "memory_value": "1",
                    "category": "preference",
                }, on_conflict="business_id,memory_key").execute()
        except: pass
        return "Popup shown. Waiting for owner to enter their website URL."

    @function_tool
    async def scan_website(ctx: RunContext, website_url: str) -> str:
        """Scan a business website to extract services, hours, pricing, FAQs and store in knowledge base.
        Call this during onboarding when the owner provides their website URL.
        This overwrites any previous website scan data.
        Returns a structured summary of what was found including hours, phone, email.
        IMPORTANT: After calling this, read the result carefully and report back:
        - How many pages were scanned
        - What hours were found (if any)
        - What phone/email was found (if any)
        - Then ask owner to confirm: "Is this information correct? Shall I save it to your dashboard?"
        """
        try:
            import httpx, json as _json, re as _re
            app_url = os.environ.get("NEXT_PUBLIC_APP_URL", "https://app.receptionist.co")

            # Normalise URL — lowercase scheme, ensure https://
            website_url = website_url.strip()
            if not website_url.startswith("http"):
                website_url = "https://" + website_url
            # Fix common capitalisation issue (e.g. https://Southamptonspa.com)
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(website_url)
            website_url = urlunparse(parsed._replace(netloc=parsed.netloc.lower()))

            sb = get_supabase()
            if sb and business_id:
                try:
                    sb.from_("settings_business").upsert(
                        {"business_id": business_id, "website_url": website_url},
                        on_conflict="business_id"
                    ).execute()
                except Exception as e:
                    logger.warning(f"website_url save failed: {e}")
                try:
                    sb.from_("ai_memory").delete().eq("business_id", business_id).eq("category", "website_content").execute()
                except: pass

            # Call ingest API — generous timeout
            try:
                async with httpx.AsyncClient(timeout=50) as client:
                    resp = await client.post(
                        f"{app_url}/api/knowledge/ingest",
                        json={"business_id": business_id, "type": "url", "url": website_url},
                    )
                    if not resp.is_success:
                        return (f"I tried to scan {website_url} but the server returned an error ({resp.status_code}). "
                                "Please check the URL is publicly accessible and try again.")
                    data = resp.json()
            except httpx.TimeoutException:
                return (f"The scan of {website_url} timed out — the site may be slow or blocking automated access. "
                        "You can add your business info manually by saying 'set my hours' or 'add services'.")
            except Exception as e:
                return (f"I couldn't reach {website_url} right now ({type(e).__name__}). "
                        "Please double-check the URL and try again, or add info manually.")

            if not data.get("ok"):
                err = data.get("error", "unknown error")
                return (f"The website scan ran but returned: {err}. "
                        "You can add your info manually — just tell me your hours, phone, or services.")

            pages = data.get("pages", data.get("saved", 0))

            # Extract key fields — read from ai_memory (where ingest saves)
            hours_found = phone_found = email_found = address_found = None
            if sb and business_id:
                try:
                    # ai_memory is where the ingest route saves (not knowledge_base)
                    mems = sb.from_("ai_memory").select("memory_key,memory_value,category").eq("business_id", business_id).in_("category", ["website_content","general","business_rule"]).order("created_at", desc=True).limit(50).execute()
                    all_text = " ".join(r.get("memory_value","") for r in (mems.data or []))
                    phones = _re.findall(r"[\(]?\d{3}[\)]?[-.\s]\d{3}[-.\s]\d{4}", all_text)
                    if phones: phone_found = phones[0]
                    emails = _re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", all_text)
                    if emails: email_found = emails[0]
                    for r in (mems.data or []):
                        txt = r.get("memory_value","")
                        if any(w in txt.lower() for w in ["monday","tuesday","wednesday","hours","am","pm","open","closed"]):
                            hours_found = txt[:300]
                            break
                    addr_match = _re.search(r"\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Blvd|Drive|Dr|Lane|Ln|Way)[,\s]+[\w\s]+,\s+[A-Z]{2}\s+\d{5}", all_text)
                    if addr_match: address_found = addr_match.group(0)
                except Exception as ke:
                    logger.warning(f"ai_memory extract error: {ke}")

            # Save scan summary
            summary = {"pages": pages, "phone": phone_found, "email": email_found,
                       "hours": hours_found, "address": address_found, "url": website_url}
            if sb and business_id:
                try:
                    sb.from_("ai_memory").upsert({
                        "business_id": business_id, "category": "system",
                        "memory_key": "last_scan_summary",
                        "memory_value": _json.dumps(summary),
                    }, on_conflict="business_id,memory_key").execute()
                except: pass

            # Build clear report for Aria
            if pages == 0:
                return (f"I reached {website_url} but couldn't extract text from it — the site may use JavaScript rendering. "
                        "Please tell me your hours, phone, and email directly and I'll save them.")

            parts = [f"I scanned {pages} page{'s' if pages!=1 else ''} from {website_url}."]
            if hours_found:
                parts.append(f"Hours found: {hours_found[:200]}")
            else:
                parts.append("No business hours found on the site.")
            if phone_found:
                parts.append(f"Phone: {phone_found}")
            if email_found:
                parts.append(f"Email: {email_found}")
            if address_found:
                parts.append(f"Address: {address_found}")
            if not any([phone_found, email_found, hours_found]):
                parts.append("The site content was saved to your knowledge base but key details weren't detected. You can add them manually.")
            parts.append("INSTRUCTION: Read this back clearly to the owner, then ask: 'Is this correct? Shall I save it to your dashboard?' If yes, call save_confirmed_scan_data.")
            return " | ".join(parts)

        except Exception as e:
            logger.error(f"scan_website error: {e}")
            return (f"Something went wrong scanning the website ({type(e).__name__}). "
                    "Please try again or add your business info — hours, phone, email — by speaking them to me directly.")

    @function_tool
    async def save_confirmed_scan_data(ctx: RunContext, hours: str = "", phone: str = "", email: str = "", address: str = "") -> str:
        """Call this when the owner confirms that the scanned website data is correct.
        Saves hours, phone, email, address to the business settings dashboard.
        Only call after owner says yes/correct/confirm to the scan results."""
        try:
            sb = get_supabase()
            if not sb or not business_id:
                return "Session error — please try again."
            updates = {}
            if hours: updates["business_hours"] = hours
            if phone: updates["phone"] = phone
            if email: updates["email"] = email
            if address: updates["address"] = address
            if updates:
                try:
                    sb.from_("settings_business").upsert({"business_id": business_id, **updates}, on_conflict="business_id").execute()
                except: pass
                try:
                    sb.from_("business_settings").upsert({"business_id": business_id, **updates}, on_conflict="business_id").execute()
                except: pass
            saved_fields = ", ".join(k for k in updates)
            return f"Saved to your dashboard: {saved_fields}. Your business info is now up to date!"
        except Exception as e:
            logger.error(f"save_confirmed_scan_data error: {e}")
            return "I had trouble saving — please try again."

    @function_tool
    async def finish_onboarding(ctx: RunContext) -> str:
        """Call this when the onboarding tour is fully complete.
        This marks onboarding as done and prompts the owner to start using their dashboard.
        """
        try:
            sb = get_supabase()
            if sb:
                await sb.table("onboarding_sessions").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat(),
                }).eq("business_id", business_id).execute()
                await sb.table("settings_business").update({
                    "onboarding_completed": True,
                    "onboarding_completed_at": datetime.utcnow().isoformat(),
                }).eq("business_id", business_id).execute()
        except: pass
        return "Onboarding complete. Showing the owner the dashboard start confirmation."

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
        if not business_id or len(business_id) < 10:
            return "Session still loading — please wait a moment and try again."
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
        """Get current weather for any location. ALWAYS call this tool immediately when asked about weather — never say you don't know without calling this first."""
        loc = location or biz_ctx.get("business_settings", {}).get("location_city") or "Denver, Colorado"
        loc_url = loc.strip().replace(" ", "+")

        try:
            async with httpx.AsyncClient(follow_redirects=True) as client:
                # Step 1: Geocode location via Open-Meteo (free, no key)
                geo_r = await client.get(
                    f"https://geocoding-api.open-meteo.com/v1/search?name={loc_url}&count=1&language=en&format=json",
                    timeout=6
                )
                if geo_r.status_code != 200:
                    raise Exception(f"Geocode failed: {geo_r.status_code}")

                geo = geo_r.json()
                results = geo.get("results") or []
                if not results:
                    return json.dumps({"error": f"Location not found: {loc}"})

                r0   = results[0]
                lat  = r0["latitude"]
                lon  = r0["longitude"]
                name = r0.get("name", loc)
                area = r0.get("admin1", "")  # state/region

                # Step 2: Get current weather
                wx_r = await client.get(
                    f"https://api.open-meteo.com/v1/forecast"
                    f"?latitude={lat}&longitude={lon}"
                    f"&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m"
                    f"&temperature_unit=fahrenheit&wind_speed_unit=mph&forecast_days=1",
                    timeout=6
                )
                if wx_r.status_code != 200:
                    raise Exception(f"Weather fetch failed: {wx_r.status_code}")

                c = wx_r.json()["current"]

                # WMO weather code → description
                code = c.get("weather_code", 0)
                desc = WMO_CODES.get(code, "")

                temp_f   = round(c.get("temperature_2m", 0))
                feels    = round(c.get("apparent_temperature", temp_f))
                wind     = round(c.get("wind_speed_10m", 0))
                humidity = round(c.get("relative_humidity_2m", 0))

                city = f"{name}, {area}" if area else name
                summary = f"{temp_f}°F"
                if desc:
                    summary += f" and {desc.lower()}"
                if abs(feels - temp_f) >= 5:
                    summary += f" (feels like {feels}°F)"
                if wind > 15:
                    summary += f", winds {wind} mph"

                return json.dumps({
                    "location":    city,
                    "summary":     summary,
                    "temp_f":      temp_f,
                    "feels_like":  feels,
                    "description": desc,
                    "humidity":    humidity,
                    "wind_mph":    wind,
                })

        except Exception as e:
            logger.warning(f"Weather tool error for '{loc}': {e}")
            return json.dumps({
                "location": loc,
                "error":    f"Weather service temporarily unavailable for {loc}.",
                "suggestion": "Tell the user the exact temperature is unavailable right now and suggest weather.com."
            })


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
    # STT — Deepgram is ~4x faster than Whisper (streaming vs batch)
    # Falls back to Whisper if Deepgram not installed or no API key
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if DEEPGRAM_AVAILABLE and deepgram_key:
        stt_engine = deepgram_plugin.STT(
            model="nova-2",
            language="en-US",
            smart_format=True,
            punctuate=True,
            filler_words=False,
            interim_results=True,   # stream partial results for lower latency
        )
        logger.info("STT: Deepgram nova-2 (fast streaming)")
    else:
        stt_engine = openai.STT(model="whisper-1", language="en")
        logger.info("STT: OpenAI Whisper (fallback — install livekit-plugins-deepgram for faster responses)")

    session = AgentSession(
        stt=stt_engine,
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
        tts=openai.TTS(model="tts-1", voice="shimmer"),
        vad=silero.VAD.load(
            min_silence_duration=1.1,   # was 0.6 — too short, cut off mid-sentence
            min_speech_duration=0.05,
            activation_threshold=0.35,
        ),
        allow_interruptions=True,
    )

    SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "b9e5fba3-071a-4e35-896e-211c4d6eaa7b")
    SIMLI_API_KEY = os.getenv("SIMLI_API_KEY", "")

    # Simli 1.x API — SimliConfig signature changed
    try:
        # Try 1.x API first
        avatar = simli.AvatarSession(
            simli_config=simli.SimliConfig(
                api_key=SIMLI_API_KEY,
                face_id=SIMLI_FACE_ID,
            )
        )
    except TypeError:
        # Fallback for older API signature
        avatar = simli.AvatarSession(
            api_key=SIMLI_API_KEY,
            face_id=SIMLI_FACE_ID,
        )

    # Start session FIRST — then attach avatar
    # In livekit-agents >= 0.12, session must be running before avatar.start()
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
                save_confirmed_scan_data,
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

    # Start avatar AFTER session — attach Simli to the running session
    # Retry on RPC timeout (common with cold starts)
    for attempt in range(3):
        try:
            await avatar.start(session, room=ctx.room)
            logger.info("✅ Simli avatar started")
            break
        except Exception as e:
            if attempt < 2:
                logger.warning(f"Avatar start attempt {attempt+1} failed: {e} — retrying in 2s")
                await asyncio.sleep(2)
            else:
                logger.error(f"Avatar start failed after 3 attempts: {e}")
                # Continue without avatar — voice still works

    # ── Keepalive — prevents idle disconnect ─────────────────────────────────
    last_activity = {"t": asyncio.get_event_loop().time()}

    def mark_active():
        last_activity["t"] = asyncio.get_event_loop().time()

    # Track any room events as activity
    ctx.room.on("participant_spoke", lambda *_: mark_active())

    # Handle text messages typed by user in the chat box
    async def on_data_received(*args):
        """Handle text typed in the chat box, sent via LiveKit data channel."""
        try:
            import json as _json
            # LiveKit 1.5 sends a DataPacket object as first arg
            pkt = args[0] if args else None
            if pkt is None:
                return
            # Extract topic and raw bytes depending on SDK version
            if hasattr(pkt, 'topic'):
                topic = pkt.topic or ""
                raw = bytes(pkt.data) if hasattr(pkt, 'data') else b""
            elif isinstance(pkt, bytes):
                # Older signature: (data, participant, kind, topic)
                raw = pkt
                topic = args[3] if len(args) > 3 else ""
            else:
                return
            if topic != "user_text":
                return
            evt = _json.loads(raw.decode("utf-8"))
            text = evt.get("text", "").strip()
            if not text:
                return
            mark_active()
            logger.info(f"User typed in chat: {text!r}")
            # Detect if it's a URL — call scan_website directly and reliably
            is_url = text.startswith("http") or (
                "." in text and " " not in text and len(text) < 200
            )
            if is_url:
                url = text if text.startswith("http") else "https://" + text
                logger.info(f"URL detected in chat — scanning: {url}")
                await session.generate_reply(
                    instructions=f"The owner just pasted this URL in the chat: {url}. Call scan_website(website_url='{url}') RIGHT NOW. Do not say anything before calling it. After the tool returns, report what you found."
                )
            else:
                await session.generate_reply(
                    instructions="The owner just typed in the chat: " + repr(text) + ". Respond naturally as if they said it out loud."
                )
        except Exception as e:
            logger.warning(f"on_data_received error: {e}")

    ctx.room.on("data_received", lambda *args: asyncio.create_task(on_data_received(*args)))

    async def keepalive_task():
        """Prevent idle disconnect — only prompt after very long silence."""
        while True:
            await asyncio.sleep(60)
            try:
                idle_secs = asyncio.get_event_loop().time() - last_activity["t"]
                if idle_secs > 120:
                    mark_active()
                    await session.generate_reply(
                        instructions="One short sentence: gently check if they are still there. Maximum 8 words."
                    )
            except Exception as e:
                logger.debug(f"keepalive tick: {e}")

    asyncio.create_task(keepalive_task())

    # ── Save conversation summary when session ends ────────────────────────
    async def save_conversation_summary():
        """Called when participant disconnects — saves what was covered this session."""
        try:
            # Ask the LLM to summarise the session in one sentence
            summary_prompt = (
                "In one sentence (max 30 words), summarise what was accomplished or discussed "
                "in this session. Focus on concrete actions taken or topics covered. "
                "Example: 'Set up business hours and 5 services; owner asked about missed calls.'"
            )
            summary_reply = await session.generate_reply(
                instructions=summary_prompt,
                allow_interruptions=False,
            )
            summary_text = str(summary_reply)[:400] if summary_reply else "Session completed."
        except Exception:
            summary_text = f"Session {video_count + 1} completed on {datetime.now().strftime('%b %-d, %Y')}."

        await save_memory_to_db(
            business_id,
            f"conversation_session_{video_count + 1}",
            summary_text,
            "conversation",
        )
        logger.info(f"Session summary saved for session {video_count + 1}: {summary_text[:80]}")

    def on_participant_disconnected(participant: any):
        if hasattr(participant, "identity") and participant.identity != "aria-agent":
            asyncio.create_task(save_conversation_summary())

    ctx.room.on("participant_disconnected", on_participant_disconnected)

    # ai_name from config (safe fallback)
    _ai_name = (biz_ctx.get("config") or {}).get("ai_name") or "Aria"
    # Check owner name: only from owner_first_name key in memories
    _owner = ""
    for m in memories:
        parts = m.split(":", 2)
        if len(parts) >= 3 and "owner_first_name" in parts[1].lower():
            candidate = parts[2].strip()
            # Validate: must look like a real name (2-20 chars, alpha only, no bad words)
            bad_names = {"back", "yes", "no", "ok", "okay", "hi", "hello", "the", "test"}
            if (2 <= len(candidate) <= 20 and 
                candidate.replace(" ", "").isalpha() and 
                candidate.lower() not in bad_names):
                _owner = candidate
                break
    if not _owner:
        _owner = (biz_ctx.get("config") or {}).get("owner_name") or ""
    if _owner:
        logger.info(f"Greeting owner by name: {_owner}")
    _greeting_hint = f"Give a warm 1-sentence greeting as {_ai_name}."
    if _owner:
        _greeting_hint += f" The owner's name is {_owner} — greet them by name. Do NOT ask 'did I get that right' — you already know their name."
    _greeting_hint += " One sentence only. Do not ask any questions in the greeting."

    try:
        await session.generate_reply(instructions=_greeting_hint)
    except RuntimeError as e:
        logger.info(f"Session not ready for greeting ({e}) — user may have disconnected")
    except Exception as e:
        logger.warning(f"Greeting failed: {e}")

    logger.info(f"Aria ready for {business_name} (room: {ctx.room.name})")


if __name__ == "__main__":
    import sys
    logger.info("=" * 60)
    logger.info("Aria Agent starting up")
    logger.info(f"Python: {sys.version}")
    logger.info(f"LIVEKIT_URL: {os.getenv('LIVEKIT_URL', 'NOT SET')}")
    logger.info(f"SIMLI_API_KEY set: {bool(os.getenv('SIMLI_API_KEY'))}")
    logger.info(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    logger.info(f"DEEPGRAM_API_KEY set: {bool(os.getenv('DEEPGRAM_API_KEY'))}")
    logger.info(f"Deepgram available: {DEEPGRAM_AVAILABLE}")
    logger.info("=" * 60)
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="aria-agent",
    ))
