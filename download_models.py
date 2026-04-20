"""
모델 사전 다운로드 스크립트
uv run python download_models.py

최초 1회 실행 후 캐시에 저장되므로 이후 실행부터는 즉시 로드됩니다.
"""

print("=== F.R.I.D.A.Y. 모델 다운로드 시작 ===\n")

# 0. unidic (MeloTTS 일본어 모듈 의존성 — 한국어 사용 시에도 필요)
print("[0/2] unidic 사전 다운로드 중...")
import subprocess, sys
subprocess.run([sys.executable, "-m", "unidic", "download"], check=True)
print("      unidic 완료 ✓\n")

# 1. Whisper (STT) — ~800MB
print("[1/2] Whisper large-v3-turbo 다운로드 중... (~800MB)")
import mlx_whisper
import numpy as np
dummy = np.zeros(16000, dtype=np.float32)
mlx_whisper.transcribe(dummy, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="ko", verbose=False)
print("      Whisper 완료 ✓\n")

# 2. MeloTTS (TTS) — ~200MB
print("[2/2] MeloTTS KR 다운로드 중... (~200MB)")
from melo.api import TTS
tts = TTS(language="KR", device="mps")
print("      MeloTTS 완료 ✓\n")

print("=== 다운로드 완료. 이제 서비스를 실행하세요. ===")
