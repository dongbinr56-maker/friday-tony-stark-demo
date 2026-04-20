"""
로컬 Whisper STT — Apple MLX 기반, API 비용 없음.

모델: openai/whisper-large-v3-turbo (기본)
  - M4 Pro에서 실시간 대비 약 5~8배 빠른 속도
  - 최초 실행 시 Hugging Face에서 자동 다운로드 (~800MB)
"""
from __future__ import annotations

import asyncio
import logging
from math import gcd

import numpy as np
from livekit import rtc
from livekit.agents import stt
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import merge_frames

logger = logging.getLogger("friday.whisper-stt")


class LocalWhisperSTT(stt.STT):
    """Apple MLX로 가속되는 로컬 Whisper STT. 인터넷 연결 불필요."""

    def __init__(
        self,
        *,
        model: str = "openai/whisper-large-v3-turbo",
        language: str = "ko",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
                offline_recognize=True,
            )
        )
        self._model_name = model
        self._default_language = language
        self._mlx = None

    def _load(self) -> None:
        if self._mlx is not None:
            return
        import mlx_whisper  # type: ignore[import]
        self._mlx = mlx_whisper
        logger.info("mlx-whisper 로드 완료: %s", self._model_name)

    async def _recognize_impl(
        self,
        buffer: stt.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        self._load()

        merged: rtc.AudioFrame = merge_frames(buffer)
        pcm = np.frombuffer(merged.data, dtype=np.int16).astype(np.float32) / 32768.0

        # Whisper는 16kHz 입력 필요 — 필요 시 리샘플링
        if merged.sample_rate != 16_000:
            from scipy.signal import resample_poly
            g = gcd(merged.sample_rate, 16_000)
            pcm = resample_poly(pcm, 16_000 // g, merged.sample_rate // g).astype(np.float32)

        lang = language if language is not NOT_GIVEN else self._default_language

        loop = asyncio.get_event_loop()
        result: dict = await loop.run_in_executor(
            None,
            lambda: self._mlx.transcribe(  # type: ignore[union-attr]
                pcm,
                path_or_hf_repo=self._model_name,
                language=lang,
                verbose=False,
            ),
        )

        text = result.get("text", "").strip()
        logger.debug("인식 결과: %r", text)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=text, language=lang, confidence=1.0)
            ],
        )
