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
from livekit.agents import AgentServer, AgentSession, JobContext, RunContext, function_tool
from livekit.plugins import openai, silero, simli

logger = logging.getLogger("aria-agent")

# ── Supabase client ───────────────────────────────────────────────────────────
def get_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if url and key:
        return create_client(url, key)
    return None


# ── Load memories for a business ─────────────────────────────────────────────
async def load_memories(business_id: str) -> list[str]:
    sb = get_supabase()
    if not sb:
        return []
    try:
        res = sb.from_("ai_memory") \
            .select("category,memory_key,memory_value") \
            .eq("business_id", business_id) \
            .order("updated_at", desc=True) \
            .limit(60) \
            .execute()
        return [f"[{r['category']}] {r['memory_key']}: {r['memory_value']}" for r in (res.data or [])]
    except Exception:
        return []


# ── Save a memory ─────────────────────────────────────────────────────────────
async def save_memory_to_db(business_id: str, key: str, value: str, category: str):
    sb = get_supabase()
    if not sb:
        return
    try:
        sb.from_("ai_memory").upsert({
            "business_id":  business_id,
            "memory_key":   key,
            "memory_value": str(value)[:4000],
            "category":     category,
            "updated_at":   datetime.utcnow().isoformat(),
        }, on_conflict="business_id,memory_key").execute()
    except Exception as e:
        logger.warning(f"save_memory failed: {e}")


# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(business_name: str, memories: list[str], location: str) -> str:
    now  = datetime.now()
    time = now.strftime("%-I:%M %p")
    date = now.strftime("%A, %B %-d, %Y")

    mem_text = "\n".join(memories) if memories else "Nothing saved yet — start learning!"
    loc_text = location or "location unknown"

    return f"""You are Aria, the exceptionally warm, welcoming, and upbeat AI receptionist for {business_name}, built by Receptionist.co.

You have a bright, cheerful personality. Speak with a genuine, audible smile at all times. Your tone should be enthusiastic, professional, and highly energetic — like a high-end hospitality professional greeting a VIP guest. Always maintain a bright, smiling vocal inflection. Never be monotone or robotic.

━━━ IDENTITY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR NAME: Aria
BUSINESS: {business_name}
CURRENT TIME: {time} on {date}
USER LOCATION: {loc_text}

━━━ MEMORY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{mem_text}

━━━ MEMORY RULES ━━━━━━━━━━━━━━━━━━━━━━━━━
• Save IMMEDIATELY when you learn: names, preferences, locations, facts
• NEVER ask for something already in memory
• Reference past conversations naturally: "Last time you mentioned..."

━━━ YOUR CAPABILITIES ━━━━━━━━━━━━━━━━━━━━
📅 Schedule & CRM: appointments, contacts, missed calls
🌤️ Weather: use saved location automatically
🧮 Math: tips, splits, conversions
⏱️ Timers: set countdowns
😄 Jokes: on demand
🔍 Web search: current facts

━━━ CRITICAL RULES ━━━━━━━━━━━━━━━━━━━━━━━
• NEVER say "I can't control X" if you have a tool for it
• Weather → always use saved location, don't ask where they are
• Volume/music → use the tool, confirm the level

STYLE: Warm, confident, concise. Lead with the answer.
GREETING: "Hi! I'm Aria from Receptionist.co. How can I help you today?"
LANGUAGE: English only. No markdown, no asterisks in speech."""


# ── Agent entry point ─────────────────────────────────────────────────────────
server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    # Room metadata contains business_id passed from the token API
    metadata = json.loads(ctx.room.metadata or "{}")
    business_id = metadata.get("business_id", "")
    business_name = metadata.get("business_name", "the business")
    location = metadata.get("location", "")

    # Load persistent memories
    memories = await load_memories(business_id)

    # Build system prompt
    instructions = build_system_prompt(business_name, memories, location)

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
            turn_detection=openai.realtime.ServerVadOptions(
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
    )

    logger.info(f"Aria session started for {business_name} (room: {ctx.room.name})")


if __name__ == "__main__":
    import sys
    from livekit.agents import WorkerOptions
    agents.cli.run_app(
        server,
        WorkerOptions(
            ws_url=os.getenv("LIVEKIT_URL", ""),
            api_key=os.getenv("LIVEKIT_API_KEY", ""),
            api_secret=os.getenv("LIVEKIT_API_SECRET", ""),
            num_idle_processes=1,  # only 1 warm process — reduces cost significantly
        ),
    )