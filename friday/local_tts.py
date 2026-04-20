"""
로컬 TTS — macOS 내장 Yuna 한국어 음성

say 명령어 사용, 설치 불필요, Apple Silicon 완전 호환.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from math import gcd

import numpy as np
import soundfile as sf  # type: ignore[import]
from livekit.agents import tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("friday.macos-tts")

_TARGET_SR = 22050
_NUM_CHANNELS = 1


class MacOSTTS(tts.TTS):
    """macOS 내장 한국어 TTS — Yuna 음성, 설치 불필요, 완전 무료."""

    def __init__(self, *, voice: str = "Yuna", rate: int = 190) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=_TARGET_SR,
            num_channels=_NUM_CHANNELS,
        )
        self._voice = voice
        self._rate = rate

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> "_MacOSChunkedStream":
        return _MacOSChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _MacOSChunkedStream(tts.ChunkedStream):

    def __init__(
        self,
        *,
        tts: MacOSTTS,
        input_text: str,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=_TARGET_SR,
            num_channels=_NUM_CHANNELS,
            mime_type="audio/pcm",
            stream=False,
        )

        tts_inst: MacOSTTS = self._tts  # type: ignore[assignment]

        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as f:
            tmp = f.name

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["say", "-v", tts_inst._voice, "-r", str(tts_inst._rate), "-o", tmp, self._input_text],
                    check=True,
                ),
            )

            data, src_sr = sf.read(tmp, dtype="float32")
            if data.ndim > 1:
                data = data[:, 0]

            if src_sr != _TARGET_SR:
                from scipy.signal import resample_poly
                g = gcd(src_sr, _TARGET_SR)
                data = resample_poly(data, _TARGET_SR // g, src_sr // g).astype("float32")

            data_int16 = (data * 32767).clip(-32768, 32767).astype(np.int16)
            output_emitter.push(data_int16.tobytes())
            output_emitter.flush()

        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
