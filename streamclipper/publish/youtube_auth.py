"""OAuth for the YouTube Data API. One token file per target account in data/tokens/."""
from __future__ import annotations

import logging
import re
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",   # captions + thumbnails + video stats
]


def token_path(account: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", account or "default")
    return get_settings().tokens_dir / f"{safe}.json"


def load_credentials(account: str):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    p = token_path(account)
    if not p.exists():
        return None
    creds = Credentials.from_authorized_user_file(str(p), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            p.write_text(creds.to_json(), encoding="utf-8")
        except Exception as e:
            log.error("token refresh failed for %s: %s (reconnect in the dashboard)", account, e)
            return None
    return creds if creds and creds.valid else None


def authorize(account: str) -> str:
    """Interactive: opens the browser for Google consent. Blocking; run in a thread."""
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = Path(get_settings().publish.client_secret)
    if not secret.exists():
        raise FileNotFoundError(f"OAuth client file not found: {secret.resolve()}")
    flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    token_path(account).write_text(creds.to_json(), encoding="utf-8")
    log.info("YouTube account '%s' connected", account)
    return account


def youtube_client(account: str):
    from googleapiclient.discovery import build

    creds = load_credentials(account)
    if not creds:
        return None
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def connected_accounts() -> list[str]:
    return sorted(p.stem for p in get_settings().tokens_dir.glob("*.json"))
