"""
LiveKit 로컬 접속용 토큰 생성기
uv run python get_token.py
"""

from livekit.api import AccessToken, VideoGrants

API_KEY    = "devkey"
API_SECRET = "friday-livekit-dev-secret-key-for-local"
ROOM       = "friday-room"
IDENTITY   = "user1"

token = (
    AccessToken(API_KEY, API_SECRET)
    .with_identity(IDENTITY)
    .with_grants(VideoGrants(room_join=True, room=ROOM))
    .to_jwt()
)

print("\n=== 플레이그라운드 접속 정보 ===")
print(f"URL   : ws://localhost:7880")
print(f"Token : {token}")
print("================================\n")
