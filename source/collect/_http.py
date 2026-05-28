"""
Felles HTTP-helper for alle collect_*-moduler.

Tre ting denne wrapperen gjør som standard `requests`-kall ikke gjør:

1. Retry med exponential backoff på transiente feil (5xx, 429, connection-feil,
   timeout). Pipelinen kjører lenge nok — særlig Matrikkelen-fetchen — at
   sjansen for at minst én HTTP-kall flopper en gang er reell.

2. Hard øvre grense på respons-størrelsen. Vi leser i stream-modus og avbryter
   nedlastingen midtveis hvis Content-Length eller faktiske bytes overstiger
   grensen. Holder en kompromittert eller feilkonfigurert kilde fra å spise
   alt RAM-et.

3. En User-Agent som identifiserer prosjektet. Offentlige norske API-er (SSB,
   Kartverket) liker å se hvor trafikken kommer fra og kan kontakte oss
   hvis vi gjør noe rart.

Helpere for cache-validering ligger også her siden de er små og brukes på
samme ringnivå (rett før eller etter et nedlast-kall).
"""

import json
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

USER_AGENT = "boligdata-pipeline (https://github.com/simen/boligdata-sammenslaaing)"

# Status-koder som er verdt å prøve igjen på. 408/429/5xx er midlertidige
# feil etter HTTP-spec. 502/503/504 er proxy/gateway-feil som ofte løses
# ved retry.
_RETRY_STATUSES = (408, 429, 500, 502, 503, 504)
_RETRY_TOTAL = 5
_BACKOFF_FACTOR = 2.0  # gir ventetider 0, 2, 4, 8, 16 sekunder


class ResponseTooLargeError(RuntimeError):
    """Kastes når en respons overstiger max_bytes-grensen."""


def _build_session() -> requests.Session:
    """Bygger en requests.Session med retry-adapter på begge HTTP-metoder.

    `Retry` håndterer både connection-feil og status-koder, men ikke
    størrelse — det må vi pakke utenpå selv.
    """
    retry = Retry(
        total=_RETRY_TOTAL,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=_RETRY_STATUSES,
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_SESSION = _build_session()


def _read_with_limit(resp: requests.Response, max_bytes: int) -> bytes:
    """Les hele responsen i biter og avbryt hvis vi passerer max_bytes.

    `Content-Length` sjekkes først som rask sanity-check, men vi stoler ikke
    blindt på den (kan være feil eller mangle). Den ekte grensen er counter-en
    under nedlastingsløkken.
    """
    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise ResponseTooLargeError(
                    f"Content-Length {declared} > max_bytes {max_bytes}"
                )
        except ValueError:
            pass  # ugyldig header — ignorer og stol på counter

    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            resp.close()
            raise ResponseTooLargeError(
                f"Respons over {max_bytes} bytes ({total} så langt) — avbryter"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def get(
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 120,
    max_bytes: int = 500_000_000,
) -> requests.Response:
    """GET med retry, timeout, og hard respons-størrelse-grense.

    Returnerer en requests.Response der `.content` og `.text` allerede er
    materialisert (gjennom vår egen stream-leser). Bruk som vanlig requests-
    Response etterpå.
    """
    resp = _SESSION.get(url, params=params, timeout=timeout, stream=True)
    try:
        content = _read_with_limit(resp, max_bytes)
    finally:
        resp.close()
    # Tving requests til å bruke våre innleste bytes som body. Behold
    # encoding/headers/status_code uendret.
    resp._content = content
    return resp


def post(
    url: str,
    *,
    json: dict | None = None,
    timeout: int = 120,
    max_bytes: int = 500_000_000,
) -> requests.Response:
    """POST med JSON-body. Samme retry/streaming-håndtering som `get`."""
    resp = _SESSION.post(
        url,
        json=json,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
        stream=True,
    )
    try:
        content = _read_with_limit(resp, max_bytes)
    finally:
        resp.close()
    resp._content = content
    return resp


# ----- Cache-validering -----
# Idempotens-sjekkene i collect_*-filene var tidligere bare "path.exists()".
# Det betyr at en avbrutt nedlasting (Ctrl-C midt i resp.content) etterlater
# en korrupt fil som påfølgende fase parser og krasjer på. Disse helperne
# brukes til å sjekke at filen faktisk er gyldig før hopp-over.


def is_valid_json(path: Path) -> bool:
    """Returner True hvis filen kan parses som JSON. Brukes ved hopp-over."""
    if not path.exists():
        return False
    try:
        with open(path, encoding="utf-8") as f:
            json.load(f)
        return True
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False


def is_valid_gml(path: Path, min_bytes: int = 500) -> bool:
    """Rask sanity-check: filen finnes, har minimum størrelse, og slutter på
    et avsluttende XML-tag. Vi parser ikke hele filen — det ville kostet
    tilsvarende standardize-fasen.
    """
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    # Sjekk siste 200 bytes for et lukkende rot-tag. Inkomplette WFS-svar
    # mangler typisk dette fordi forbindelsen ble brutt.
    try:
        with open(path, "rb") as f:
            f.seek(-200, 2)
            tail = f.read().decode("utf-8", errors="ignore")
        return "</" in tail and ">" in tail.rsplit("</", 1)[-1]
    except (OSError, ValueError):
        return False


def politely_sleep(seconds: float = 0.2) -> None:
    """Liten pause for å være snill med offentlige API-er. Brukes mellom
    SSB-kall der vi ikke har en naturlig page-loop med innebygd sleep.
    """
    time.sleep(seconds)
