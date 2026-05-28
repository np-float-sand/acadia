"""
ERCOT B2C authentication helper.

ERCOT's public API uses Azure AD B2C ROPC flow.  Each id_token is valid
for ~1 hour; this module caches it in memory and refreshes automatically.

Credentials (set as env vars or fall back to defaults below):
    ERCOT_USERNAME          : your developer.ercot.com email
    ERCOT_PASSWORD          : your developer.ercot.com password
    ERCOT_SUBSCRIPTION_KEY  : subscription key from the ERCOT API portal

Registration:
    https://developer.ercot.com/applications/pubapi/user-guide/registration-and-authentication/
"""

from __future__ import annotations

import os
import time

import requests

_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
    "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"

# In-memory token cache: {token_str, expires_at_epoch}
_cache: dict = {"token": None, "expires_at": 0.0}

# Token lifetime buffer: refresh 5 minutes before expiry
_REFRESH_BUFFER_SECS = 300


def _credentials() -> tuple[str, str, str]:
    username = os.environ.get("ERCOT_USERNAME", "sand.gh1902@gmail.com")
    password = os.environ.get("ERCOT_PASSWORD", "")
    sub_key  = os.environ.get("ERCOT_SUBSCRIPTION_KEY", "")
    if not password:
        raise EnvironmentError(
            "ERCOT_PASSWORD not set.  "
            "Set it as an env var (see developer.ercot.com for registration)."
        )
    if not sub_key:
        raise EnvironmentError(
            "ERCOT_SUBSCRIPTION_KEY not set.  "
            "Find your key at developer.ercot.com → Profile → Subscriptions."
        )
    return username, password, sub_key


def get_id_token(force_refresh: bool = False) -> str:
    """
    Return a valid ERCOT id_token, refreshing from B2C if expired.
    Token is cached in-process for up to 55 minutes.
    """
    now = time.time()
    if not force_refresh and _cache["token"] and now < _cache["expires_at"] - _REFRESH_BUFFER_SECS:
        return _cache["token"]

    username, password, _ = _credentials()
    print("[ercot_auth] Obtaining new id_token from ERCOT B2C…")

    resp = requests.post(
        _TOKEN_URL,
        data={
            "username":      username,
            "password":      password,
            "grant_type":    "password",
            "scope":         f"openid {_CLIENT_ID} offline_access",
            "client_id":     _CLIENT_ID,
            "response_type": "id_token",
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    token = payload.get("id_token")
    if not token:
        raise ValueError(f"No id_token in ERCOT B2C response: {payload}")

    # B2C doesn't always return expires_in; default to 3600s (1 hour)
    expires_in = int(payload.get("expires_in", 3600))
    _cache["token"]      = token
    _cache["expires_at"] = now + expires_in

    print(f"  [ercot_auth] Token obtained (valid ~{expires_in // 60} min).")
    return token


def make_headers() -> dict:
    """Return auth headers for an ERCOT API request."""
    _, _, sub_key = _credentials()
    return {
        "Authorization":          f"Bearer {get_id_token()}",
        "Ocp-Apim-Subscription-Key": sub_key,
    }
