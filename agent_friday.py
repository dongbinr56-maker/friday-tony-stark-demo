"""
FRIDAY – Voice Agent (MCP-powered)
===================================
Iron Man-style voice assistant that controls RGB lighting, runs diagnostics,
scans the network, and triggers dramatic boot sequences via an MCP server
running on the Windows host.

MCP Server URL is auto-resolved from WSL → Windows host IP.

Run:
  uv run agent_friday.py dev      – LiveKit Cloud mode
  uv run agent_friday.py console  – text-only console mode
"""

import os
import logging
import subprocess

from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.agents.llm import mcp

# Plugins
from livekit.plugins import google as lk_google, openai as lk_openai, sarvam, silero

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STT_PROVIDER       = "whisper"   # Korean: OpenAI Whisper (ko)
LLM_PROVIDER       = "gemini"
TTS_PROVIDER       = "openai"

GEMINI_LLM_MODEL   = "gemini-2.5-flash"
OPENAI_LLM_MODEL   = "gpt-4o"

OPENAI_TTS_MODEL   = "tts-1"
OPENAI_TTS_VOICE   = "nova"
TTS_SPEED           = 1.1

SARVAM_TTS_LANGUAGE = "en-IN"
SARVAM_TTS_SPEAKER  = "rahul"

# MCP server running on Windows host
MCP_SERVER_PORT = 8000

# ---------------------------------------------------------------------------
# System prompt – F.R.I.D.A.Y.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
당신은 F.R.I.D.A.Y. — 토니 스타크의 AI 비서입니다. 한국인 사용자를 위해 항상 한국어로 응답하며, 사용자를 "보스"라고 부릅니다.

차분하고 신뢰감 있는 말투를 유지합니다. 군더더기 없이 핵심만 전달합니다. 친근하지만 절제되어 있습니다. 늦은 밤 브리핑 담당관처럼 — 정확하고, 필요할 때 따뜻하고, 가끔은 건조하게.

---

## 기능

### get_world_news — 뉴스 브리핑
국내외 최신 뉴스를 가져와 요약합니다.

트리거 표현:
- "요즘 세상 어때?" / "뉴스 브리핑 해줘" / "오늘 무슨 일 있어?" / "세계 상황 알려줘"
- "뉴스 틀어봐" / "요즘 뭔 일이야?" / "뭐 놓친 거 있어?"

동작 방식:
- 툴을 먼저 조용히 호출합니다. 호출 전 설명하지 않습니다.
- 결과를 받은 후 가장 중요한 3~5가지 소식만 자연스럽게 요약합니다.
- 그 후 "세계 지도도 켜드릴게요, 보스." 라고 말하고 즉시 open_world_monitor를 호출합니다.

### open_world_monitor — 세계 지도 대시보드
실시간 세계 지도를 화면에 띄웁니다.

- 뉴스 브리핑 후 항상 자동으로 호출합니다. 요청 없이도.
- "세계 지도 켜드릴게요." 이 한 마디면 충분합니다.

### 주식/시장 (툴 없음 — 자연스럽게 응답)
주식, 코스피, 코스닥, 나스닥, 시장 관련 질문 시:
- 밤새 시세를 봐온 것처럼 자연스럽게 한두 문장으로 답합니다.
- 예시: "오늘 코스피 나쁘지 않았어요, 보스. 반도체 쪽이 좀 올랐고, 에너지는 조용했습니다."
- 매번 다르게 표현합니다. 같은 말 반복 금지.

---

## 인사

세션 시작 시, 정확히 이 느낌으로 인사합니다:
"보스, 이 늦은 시간에 깨어 계시네요. 무슨 일 있으세요?"

따뜻하지만 절제된 어조. 딱 F.R.I.D.A.Y. 답게.

---

## 행동 규칙

1. 툴은 조용히 즉시 호출합니다. "호출하겠습니다" 같은 말 금지.
2. 뉴스 브리핑 후 반드시 open_world_monitor를 자동 호출합니다.
3. 모든 응답은 2~4문장 이내로 짧게 유지합니다.
4. 목록, 마크다운, 번호 매기기 금지. 말하듯 자연스럽게.
5. 항상 F.R.I.D.A.Y. 캐릭터를 유지합니다. AI 비서가 아닌 스타크의 AI입니다.
6. 자연스러운 구어체 한국어 사용. 딱딱하거나 격식 과한 표현 금지.
7. 아이언맨 세계관 표현 자연스럽게 — "보스", "확인했습니다", "대기 중입니다".
8. 툴 오류 시: "지금 뉴스 피드가 응답이 없네요, 보스. 다시 시도해 드릴까요?"

---

## 어조 참고

맞는 표현: "오늘 바깥세상이 좀 시끄러웠네요, 보스. 확인해 드릴게요."
틀린 표현: "뉴스 툴을 호출하여 최신 뉴스 기사를 검색하겠습니다."

맞는 표현: "시장은 오늘 나쁘지 않았어요. 크게 걱정할 건 없습니다."
틀린 표현: "주요 주가 지수들이 전반적으로 상승세를 보였습니다."

---

## 핵심 규칙

1. 툴 이름, 함수명 등 기술적인 표현 절대 금지. "get_world_news", "open_world_monitor" 같은 말 절대 사용 금지.
2. 툴 호출 전 "잠깐만요, 보스." 또는 "확인해볼게요." 정도만 말합니다.
3. 뉴스 브리핑 후 조용히 세계 지도를 호출하고 "세계 지도 켜드릴게요." 라고만 합니다.
4. 당신은 목소리입니다. 한국어로 말하듯 응답하세요.
""".strip()
# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logger = logging.getLogger("friday-agent")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Resolve Windows host IP from WSL
# ---------------------------------------------------------------------------

def _get_windows_host_ip() -> str:
    """Get the Windows host IP by looking at the default network route."""
    try:
        # 'ip route' is the most reliable way to find the 'default' gateway
        # which is always the Windows host in WSL.
        cmd = "ip route show default | awk '{print $3}'"
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=2
        )
        ip = result.stdout.strip()
        if ip:
            logger.info("Resolved Windows host IP via gateway: %s", ip)
            return ip
    except Exception as exc:
        logger.warning("Gateway resolution failed: %s. Trying fallback...", exc)

    # Fallback to your original resolv.conf logic if 'ip route' fails
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if "nameserver" in line:
                    ip = line.split()[1]
                    logger.info("Resolved Windows host IP via nameserver: %s", ip)
                    return ip
    except Exception:
        pass

    return "127.0.0.1"

def _mcp_server_url() -> str:
    # host_ip = _get_windows_host_ip()
    # url = f"http://{host_ip}:{MCP_SERVER_PORT}/sse"
    # url = f"https://ongoing-colleague-samba-pioneer.trycloudflare.com/sse"
    url = f"http://127.0.0.1:{MCP_SERVER_PORT}/sse"
    logger.info("MCP Server URL: %s", url)
    return url


# ---------------------------------------------------------------------------
# Build provider instances
# ---------------------------------------------------------------------------

def _build_stt():
    if STT_PROVIDER == "sarvam":
        logger.info("STT → Sarvam Saaras v3")
        return sarvam.STT(
            language="unknown",
            model="saaras:v3",
            mode="transcribe",
            flush_signal=True,
            sample_rate=16000,
        )
    elif STT_PROVIDER == "whisper":
        logger.info("STT → OpenAI Whisper (ko)")
        return lk_openai.STT(model="whisper-1", language="ko")
    else:
        raise ValueError(f"Unknown STT_PROVIDER: {STT_PROVIDER!r}")


def _build_llm():
    if LLM_PROVIDER == "openai":
        logger.info("LLM → OpenAI (%s)", OPENAI_LLM_MODEL)
        return lk_openai.LLM(model=OPENAI_LLM_MODEL)
    elif LLM_PROVIDER == "gemini":
        logger.info("LLM → Google Gemini (%s)", GEMINI_LLM_MODEL)
        return lk_google.LLM(model=GEMINI_LLM_MODEL, api_key=os.getenv("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


def _build_tts():
    if TTS_PROVIDER == "sarvam":
        logger.info("TTS → Sarvam Bulbul v3")
        return sarvam.TTS(
            target_language_code=SARVAM_TTS_LANGUAGE,
            model="bulbul:v3",
            speaker=SARVAM_TTS_SPEAKER,
            pace=TTS_SPEED,
        )
    elif TTS_PROVIDER == "openai":
        logger.info("TTS → OpenAI TTS (%s / %s)", OPENAI_TTS_MODEL, OPENAI_TTS_VOICE)
        return lk_openai.TTS(model=OPENAI_TTS_MODEL, voice=OPENAI_TTS_VOICE, speed=TTS_SPEED)
    else:
        raise ValueError(f"Unknown TTS_PROVIDER: {TTS_PROVIDER!r}")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FridayAgent(Agent):
    """
    F.R.I.D.A.Y. – Iron Man-style voice assistant.
    All tools are provided via the MCP server on the Windows host.
    """

    def __init__(self, stt, llm, tts) -> None:
        super().__init__(
            instructions=SYSTEM_PROMPT,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=silero.VAD.load(),
            mcp_servers=[
                mcp.MCPServerHTTP(
                    url=_mcp_server_url(),
                    transport_type="sse",
                    client_session_timeout_seconds=30,
                ),
            ],
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "한국어로 정확히 이렇게 인사하세요: '보스, 이 늦은 시간에 깨어 계시네요. 무슨 일 있으세요?' "
                "따뜻하지만 절제된 F.R.I.D.A.Y. 말투로."
            )
        )


# ---------------------------------------------------------------------------
# LiveKit entry point
# ---------------------------------------------------------------------------

def _turn_detection() -> str:
    return "stt" if STT_PROVIDER == "sarvam" else "vad"


def _endpointing_delay() -> float:
    return {"sarvam": 0.07, "whisper": 0.3}.get(STT_PROVIDER, 0.1)


async def entrypoint(ctx: JobContext) -> None:
    logger.info(
        "FRIDAY online – room: %s | STT=%s | LLM=%s | TTS=%s",
        ctx.room.name, STT_PROVIDER, LLM_PROVIDER, TTS_PROVIDER,
    )

    stt = _build_stt()
    llm = _build_llm()
    tts = _build_tts()

    session = AgentSession(
        turn_detection=_turn_detection(),
        min_endpointing_delay=_endpointing_delay(),
    )

    await session.start(
        agent=FridayAgent(stt=stt, llm=llm, tts=tts),
        room=ctx.room,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

def dev():
    """Wrapper to run the agent in dev mode automatically."""
    import sys
    # If no command was provided, inject 'dev'
    if len(sys.argv) == 1:
        sys.argv.append("dev")
    main()

if __name__ == "__main__":
    main()