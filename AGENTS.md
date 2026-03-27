# Receptionist Railway Services — Agent Guide

**Stack:** Python 3.11, FastAPI, LiveKit Agents SDK, Twilio  
**Deployment:** Railway (2 services)

## Services

### aria-agent (this repo's `railway/` dir)
- `aria_agent.py` — LiveKit agent worker. Connects to LiveKit Cloud, handles AI voice calls.
- Uses `livekit-agents` SDK with Deepgram STT, OpenAI LLM, Cartesia TTS, Simli video.
- Deployed as: `python aria_agent.py start`

### aria-call-handler (FastAPI)
- `requirements.call_handler.txt` — Dependencies for the call handler service.
- Receives Twilio webhooks, provisions calls, bridges to LiveKit rooms.
- Public endpoint: `aria-call-handler-production.up.railway.app:8080`

## Environment variables
`LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, `SIMLI_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`

## Security rules
- `aria_agent.py` has an IDENTITY LOCK prompt block — do NOT weaken or remove it.
- The SSRF guard blocks `*.railway.internal` and private IP ranges from `scan_website` tool calls.
- Never log full call transcripts to stdout in production.
- `os.getenv()` calls for credentials must have no default value — fail loudly if missing.

## Dependencies
- Pin all versions in `requirements.txt` — do not use `>=` ranges for security-sensitive packages.
- Run `pip install -r requirements.txt --break-system-packages` in Railway builds.
