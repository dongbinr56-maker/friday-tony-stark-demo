"""
로컬 MeloTTS — 한국어 TTS, Apple MPS 가속, API 비용 없음.

모델: MeloTTS KR (MyShell AI)
  - 최초 실행 시 Hugging Face에서 자동 다운로드 (~200MB)
  - 출력: 44100 Hz, mono, int16 PCM
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("friday.melo-tts")

_SAMPLE_RATE = 44100
_NUM_CHANNELS = 1


class MeloTTSAdapter(tts.TTS):
    """MeloTTS 기반 로컬 한국어 TTS. Apple MPS로 가속."""

    def __init__(
        self,
        *,
        language: str = "KR",
        speed: float = 1.0,
        device: str = "mps",
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
        )
        self._language = language
        self._speed = speed
        self._device = device
        self._model = None
        self._speaker_id: int | None = None

    def prewarm(self) -> None:
        self._load()

    def _load(self) -> None:
        if self._model is not None:
            return
        from melo.api import TTS  # type: ignore[import]
        self._model = TTS(language=self._language, device=self._device)
        self._speaker_id = self._model.hps.data.spk2id[self._language]
        logger.info("MeloTTS 로드 완료 — 언어=%s 디바이스=%s", self._language, self._device)

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_MeloChunkedStream":
        return _MeloChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _MeloChunkedStream(tts.ChunkedStream):

    def __init__(
        self,
        *,
        tts: MeloTTSAdapter,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._adapter: MeloTTSAdapter = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        self._adapter._load()

        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=_SAMPLE_RATE,
            num_channels=_NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._adapter._model.tts_to_file(
                    self._input_text,
                    self._adapter._speaker_id,
                    tmp_path,
                    speed=self._adapter._speed,
                    quiet=True,
                ),
            )

            import soundfile as sf  # type: ignore[import]
            data, _ = sf.read(tmp_path, dtype="int16")
            if data.ndim > 1:
                data = data[:, 0]  # 스테레오 → 모노

            output_emitter.push(data.tobytes())
            output_emitter.flush()

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
