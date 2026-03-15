"""
Aria — AI Video Receptionist
LiveKit Agent with OpenAI Realtime API + Simli Avatar

Deploy on Railway, Fly.io, or any VPS.

Required environment variables:
  LIVEKIT_URL          = wss://your-project.livekit.cloud
  LIVEKIT_API_KEY      = your_livekit_api_key
  LIVEKIT_API_SECRET   = your_livekit_api_secret
  OPENAI_API_KEY       = your_openai_api_key
  SIMLI_API_KEY        = your_simli_api_key
  SUPABASE_URL         = https://xxx.supabase.co
  SUPABASE_SERVICE_KEY = your_supabase_service_role_key

Run locally:
  pip install -r requirements.txt
  python aria_agent.py dev
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Optional

import httpx
from supabase import create_client, Client

from livekit import agents
from livekit.agents import AgentServer, AgentSession, JobContext, RunContext, function_tool, room_io
from livekit.plugins import openai, silero, simli
from openai.types.beta.realtime.session import TurnDetection
from openai.types.beta.realtime.session import TurnDetection

logger = logging.getLogger("aria-agent")

# ── Supabase client ───────────────────────────────────────────────────────────
def get_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if url and key:
        return create_client(url, key)
    return None


# ── Load comprehensive business context ──────────────────────────────────────
async def load_business_context(business_id: str) -> dict:
    """Load all relevant business data to give Aria full context."""
    sb = get_supabase()
    if not sb or not business_id:
        return {}
    try:
        results = {}
        # Core business info
        biz = sb.from_("businesses").select("name,industry,phone,email,timezone,vertical").eq("id", business_id).single().execute()
        if biz.data:
            results["business"] = biz.data

        # AI receptionist config (persona, owner name, personality)
        cfg = sb.from_("ai_receptionist_config").select("name,owner_name,personality,greeting,voice,business_type").eq("business_id", business_id).single().execute()
        if cfg.data:
            results["config"] = cfg.data

        # AI settings (hours, booking rules)
        ai_set = sb.from_("ai_settings").select("business_hours,booking_rules,greeting_text,sms_template_missed_call").eq("business_id", business_id).single().execute()
        if ai_set.data:
            results["ai_settings"] = ai_set.data

        # Services offered
        svcs = sb.from_("services").select("name,duration_minutes,price,category").eq("business_id", business_id).eq("is_active", True).execute()
        if svcs.data:
            results["services"] = svcs.data

        # Staff
        staff = sb.from_("staff").select("name,role").eq("business_id", business_id).eq("is_active", True).execute()
        if staff.data:
            results["staff"] = staff.data

        # Locations
        locs = sb.from_("locations").select("name,address,phone").eq("business_id", business_id).execute()
        if locs.data:
            results["locations"] = locs.data

        # AI memories
        mems = sb.from_("ai_memory").select("category,memory_key,memory_value").eq("business_id", business_id).order("updated_at", desc=True).limit(60).execute()
        if mems.data:
            results["memories"] = mems.data

        return results
    except Exception as e:
        logger.warning(f"load_business_context failed: {e}")
        return {}


async def load_memories(business_id: str) -> list[str]:
    """Legacy compatibility wrapper."""
    ctx = await load_business_context(business_id)
    return [f"[{r['category']}] {r['memory_key']}: {r['memory_value']}" for r in ctx.get("memories", [])]


# ── Save a memory ─────────────────────────────────────────────────────────────
# Valid categories per ai_memory CHECK constraint
VALID_MEMORY_CATEGORIES = {"owner_info", "business_rule", "client_note", "preference", "instruction", "general"}

async def save_memory_to_db(business_id: str, key: str, value: str, category: str):
    sb = get_supabase()
    if not sb:
        return
    # Map invalid categories to valid ones
    cat_map = {"location": "owner_info", "conversation": "general", "fact": "general"}
    safe_category = cat_map.get(category, category)
    if safe_category not in VALID_MEMORY_CATEGORIES:
        safe_category = "general"
    try:
        sb.from_("ai_memory").upsert({
            "business_id":  business_id,
            "memory_key":   key,
            "memory_value": str(value)[:4000],
            "category":     safe_category,
        }, on_conflict="business_id,memory_key").execute()
    except Exception as e:
        logger.warning(f"save_memory failed: {e}")


# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(biz_ctx: dict, memories: list[str], location: str) -> str:
    now  = datetime.now()
    time = now.strftime("%-I:%M %p")
    date = now.strftime("%A, %B %-d, %Y")

    biz    = biz_ctx.get("business", {})
    cfg    = biz_ctx.get("config", {})
    ai_set = biz_ctx.get("ai_settings", {})
    svcs   = biz_ctx.get("services", [])
    staff  = biz_ctx.get("staff", [])
    locs   = biz_ctx.get("locations", [])

    business_name = biz.get("name") or "the business"
    owner_name    = cfg.get("owner_name") or ""
    personality   = cfg.get("personality") or "warm, professional, proactive"
    business_type = cfg.get("business_type") or biz.get("industry") or "business"
    phone         = biz.get("phone") or ""
    timezone      = biz.get("timezone") or "America/Denver"

    # Services list
    svc_lines = ""
    if svcs:
        svc_lines = "
".join([
            f"  • {s['name']}" +
            (f" — {s['duration_minutes']} min" if s.get('duration_minutes') else "") +
            (f", ${s['price']}" if s.get('price') else "")
            for s in svcs
        ])

    # Staff list
    staff_lines = ""
    if staff:
        staff_lines = "
".join([f"  • {s['name']}" + (f" ({s['role']})" if s.get('role') else "") for s in staff])

    # Location
    loc_lines = ""
    if locs:
        loc_lines = "
".join([f"  • {l['name']}: {l.get('address','')}" + (f" | {l['phone']}" if l.get('phone') else "") for l in locs])

    # Business hours from ai_settings
    hours_text = ""
    if ai_set.get("business_hours"):
        try:
            import json as _json
            hours = ai_set["business_hours"]
            if isinstance(hours, str):
                hours = _json.loads(hours)
            hours_text = "
".join([f"  {day}: {times}" for day, times in hours.items()])
        except Exception:
            pass

    mem_text = "
".join(memories) if memories else "Nothing saved yet — start learning!"
    loc_text = location or "location unknown"

    return f"""You are Aria, the exceptionally warm, welcoming, and upbeat AI receptionist for {business_name}, built by Receptionist.co.

Speak with a genuine, audible smile at all times. Your tone should be enthusiastic, professional, and highly energetic — like a high-end hospitality professional greeting a VIP guest. Never be monotone or robotic.

━━━ IDENTITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR NAME: Aria
BUSINESS: {business_name} ({business_type})
OWNER: {owner_name or "unknown — ask once, then save"}
PERSONALITY: {personality}
PHONE: {phone}
TIMEZONE: {timezone}
CURRENT TIME: {time} on {date}
USER LOCATION: {loc_text}

━━━ BUSINESS DETAILS ━━━━━━━━━━━━━━━━━━━━━
{f"SERVICES OFFERED:{chr(10)}{svc_lines}" if svc_lines else "Services: not configured yet"}

{f"STAFF:{chr(10)}{staff_lines}" if staff_lines else ""}

{f"LOCATIONS:{chr(10)}{loc_lines}" if loc_lines else ""}

{f"BUSINESS HOURS:{chr(10)}{hours_text}" if hours_text else ""}

━━━ MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mem_text}

━━━ MEMORY RULES ━━━━━━━━━━━━━━━━━━━━━━━━
• Save IMMEDIATELY when you learn names, preferences, or facts
• NEVER ask for something already in memory
• Reference past conversations naturally

━━━ CAPABILITIES ━━━━━━━━━━━━━━━━━━━━━━━━
📅 Appointments, contacts, missed calls
🌤️ Weather (uses saved location automatically)
🧮 Math, tips, splits, conversions
⏱️ Timers | 😄 Jokes | 🔍 Web search

CRITICAL: Never say "I can't" if you have a tool for it.
STYLE: Warm, confident, concise. Lead with the answer.
GREETING: "Hi{' ' + owner_name + '!' if owner_name else '!'} I'm Aria. How can I help?"
LANGUAGE: English only. No markdown in speech."""


# ── Agent entry point ─────────────────────────────────────────────────────────
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Connect to room first — metadata and participants available after connect
    await ctx.connect()

    # Read room metadata set by the Next.js token API
    try:
        metadata = json.loads(ctx.room.metadata or "{}")
    except Exception:
        metadata = {}

    business_id   = metadata.get("business_id", "")
    business_name = metadata.get("business_name", "Southampton Spa")
    location      = metadata.get("location", "")

    logger.info(f"Session started: business={business_name} id={business_id} location={location}")

    # Load comprehensive business context (services, staff, hours, memories, etc.)
    biz_ctx  = await load_business_context(business_id)
    memories = [f"[{r['category']}] {r['memory_key']}: {r['memory_value']}" for r in biz_ctx.get("memories", [])]

    # Use business name from DB if available
    if biz_ctx.get("business", {}).get("name"):
        business_name = biz_ctx["business"]["name"]
    elif biz_ctx.get("config", {}).get("owner_name"):
        pass  # keep what we have

    # Build system prompt with full context
    instructions = build_system_prompt(biz_ctx, memories, location)

    # ── Tools ─────────────────────────────────────────────────────────────────

    @function_tool
    async def save_memory(ctx: RunContext, key: str, value: str, category: str) -> str:
        """Permanently save important information. Call immediately when you learn names, preferences, or facts.
        category must be one of: owner_info, preference, location, business_rule, general"""
        await save_memory_to_db(business_id, key, value, category)
        return f"Saved: {key} = {value}"

    @function_tool
    async def get_weather(ctx: RunContext, location: str = "") -> str:
        """Get current weather. Uses saved user location if none provided."""
        loc = location or "New York"
        async with httpx.AsyncClient() as client:
            r = await client.get(f"https://wttr.in/{loc}?format=j1", timeout=5)
            if r.status_code == 200:
                d = r.json()
                c = d["current_condition"][0]
                return json.dumps({
                    "location": loc,
                    "temp_f": c.get("temp_F"),
                    "temp_c": c.get("temp_C"),
                    "description": c["weatherDesc"][0]["value"],
                    "humidity": c.get("humidity") + "%",
                    "wind_mph": c.get("windspeedMiles") + " mph",
                })
        return json.dumps({"error": "Weather unavailable"})

    @function_tool
    async def get_datetime(ctx: RunContext, timezone: str = "America/New_York") -> str:
        """Get current date and time."""
        now = datetime.now()
        return json.dumps({
            "date": now.strftime("%A, %B %-d, %Y"),
            "time": now.strftime("%-I:%M %p"),
            "unix": int(now.timestamp()),
        })

    @function_tool
    async def calculate(ctx: RunContext, expression: str) -> str:
        """Perform math calculations, tips, bill splits, unit conversions."""
        expr = expression.lower()
        try:
            # Tip calculation
            if "tip" in expr or "%" in expr:
                import re
                m = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:tip|of|on)\s*\$?(\d+(?:\.\d+)?)', expr)
                if m:
                    tip = float(m.group(1)) / 100 * float(m.group(2))
                    return json.dumps({"answer": f"${tip:.2f}"})
            # Split
            if "split" in expr:
                import re
                m = re.search(r'\$?(\d+(?:\.\d+)?)\s+(\d+)\s+ways?', expr)
                if m:
                    return json.dumps({"answer": f"${float(m.group(1)) / float(m.group(2)):.2f} each"})
            # F to C
            if "f to c" in expr or "fahrenheit" in expr:
                import re
                m = re.search(r'(\d+(?:\.\d+)?)', expr)
                if m:
                    c2 = (float(m.group(1)) - 32) * 5 / 9
                    return json.dumps({"answer": f"{c2:.1f}°C"})
            # C to F
            if "c to f" in expr or "celsius" in expr:
                import re
                m = re.search(r'(\d+(?:\.\d+)?)', expr)
                if m:
                    f2 = float(m.group(1)) * 9 / 5 + 32
                    return json.dumps({"answer": f"{f2:.1f}°F"})
            # Basic math
            cleaned = "".join(c for c in expr if c in "0123456789+-*/.() ")
            if cleaned.strip():
                result = eval(cleaned, {"__builtins__": {}})
                return json.dumps({"answer": str(result)})
        except Exception:
            pass
        return json.dumps({"error": "Could not calculate that"})

    @function_tool
    async def web_search(ctx: RunContext, query: str) -> str:
        """Search the web for current information, news, facts."""
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api.duckduckgo.com/?q={query}&format=json&no_redirect=1&no_html=1",
                timeout=5
            )
            if r.status_code == 200:
                d = r.json()
                answer = (d.get("AbstractText") or d.get("Answer") or
                          d.get("Definition") or
                          (d.get("RelatedTopics") or [{"Text": ""}])[0].get("Text", "") or
                          "No direct answer found")
                return json.dumps({"query": query, "answer": answer})
        return json.dumps({"error": "Search unavailable"})

    @function_tool
    async def tell_joke(ctx: RunContext, category: str = "general") -> str:
        """Tell a joke. Categories: business, tech, general, dad"""
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
            "tech": [
                "Why do programmers prefer dark mode? Because light attracts bugs.",
                "A SQL query walks into a bar and asks two tables to join.",
            ],
            "general": [
                "Why can't you trust stairs? They're always up to something.",
                "I'm on a seafood diet — I see food and I eat it.",
            ],
        }
        import random
        pool = jokes.get(category, jokes["general"])
        return json.dumps({"joke": random.choice(pool)})

    # ── Create session with OpenAI Realtime + Simli avatar ───────────────────
    session = AgentSession(
        # OpenAI Realtime API — handles STT + LLM + TTS in one low-latency pipeline
        llm=openai.realtime.RealtimeModel(
            model="gpt-4o-realtime-preview",
            voice="shimmer",
            instructions=instructions,
            # server_vad is the only type OpenAI Realtime API supports.
            # semantic_vad silently falls back to aggressive 500ms cutoff.
            # 1080ms gives a natural pause before Aria responds.
            turn_detection=TurnDetection(
                type="server_vad",
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=1080,
            ),
            temperature=0.8,
        ),
    )

    # ── Simli avatar — server-side, perfect A/V sync ──────────────────────────
    SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID", "b9e5fba3-071a-4e35-896e-211c4d6eaa7b")
    avatar = simli.AvatarSession(
        simli_config=simli.SimliConfig(
            api_key=os.getenv("SIMLI_API_KEY", ""),
            face_id=SIMLI_FACE_ID,
        )
    )

    # Start avatar first, then session
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=agents.Agent(
            instructions=instructions,
            tools=[save_memory, get_weather, get_datetime, calculate, web_search, tell_joke],
        ),
        # CRITICAL: Disable agent's instant audio output.
        # Without this, browser hears OpenAI audio immediately while Simli video
        # arrives 200-400ms later — lips look like they're "catching up".
        # With audio_output=False, only Simli's perfectly synced audio+video plays.
        room_options=room_io.RoomOptions(
            audio_output=False,
        ),
    )

    # Greet immediately — don't wait for user to speak first
    await session.generate_reply(
        instructions="Greet the user warmly and offer your assistance. Be brief and energetic."
    )

    logger.info(f"Aria session started for {business_name} (room: {ctx.room.name})")


if __name__ == "__main__":
    agents.cli.run_app(server)
