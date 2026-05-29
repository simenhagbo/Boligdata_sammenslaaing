"""
Henter klima-normaler 1991-2020 fra MET Norges Frost-API.

Frost er gratis men krever en klient-ID (registrert på frost.met.no/auth/).
Vi leser den fra miljøvariabelen FROST_CLIENT_ID eller fra en .env-fil i
prosjektrot. Hvis ingen ID er satt, hopper hele kilden gracefully over —
pipelinen fortsetter med dagens storby-baserte klima-proxy.

Strategi:
  1. Hent metadata for alle norske SensorSystem-stasjoner (vekter på ~700-1000)
  2. For hver stasjon, hent årlige klima-normaler (snitt-temp + nedbør) for
     standardperioden 1991-2020
  3. Lagre rådata som JSON; standardize-fasen mapper hvert postnummer til
     nærmeste stasjon med klima-data

Output:
  raw_data/met_frost/stasjoner.json     metadata + koordinater
  raw_data/met_frost/normaler.json      klima-normaler per stasjon
"""

import json
import sys
from pathlib import Path

import requests

# Sørg for at source/ er på sys.path slik at både direktekjøring og import
# via pipeline.py finner _http og _config
_SOURCE_DIR = Path(__file__).parents[1]
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from collect import _http  # noqa: E402
from collect._config import get_secret  # noqa: E402

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "met_frost"
RAW_DIR.mkdir(parents=True, exist_ok=True)

FROST_BASE = "https://frost.met.no"
# Standardperiode for klima-normaler (WMOs anbefaling fra 2020).
NORMAL_PERIOD = "1991/2020"
# To kjernevariabler. Periode-suffikset hører i `period`-query-parameteren,
# IKKE i element-ID-en — det er en lett feil å gjøre.
NORMAL_ELEMENTS = [
    "mean(air_temperature P1Y)",
    "sum(precipitation_amount P1Y)",
]
# Max antall stasjoner per spørring. Frost svarer raskt for moderate chunks,
# men kan time-out hvis vi ber om alle 1500+ samtidig.
SOURCES_CHUNK = 50


class FrostAuthMissing(RuntimeError):
    """Kastes når FROST_CLIENT_ID ikke er satt. main() håndterer det som
    en mild advarsel slik at pipelinen fortsetter."""


def _client_id() -> str:
    """Returner Frost-klient-ID eller kast en spesifikk feil."""
    cid = get_secret("FROST_CLIENT_ID")
    if not cid:
        raise FrostAuthMissing(
            "FROST_CLIENT_ID ikke satt. Registrer en konto på "
            "frost.met.no/auth/requestCredentials og legg ID-en i en "
            ".env-fil i prosjektrot:\n\n  FROST_CLIENT_ID=din-id-her\n"
        )
    return cid


def _frost_get(path: str, params: dict, max_bytes: int = 100_000_000) -> dict:
    """GET mot Frost med Basic Auth (klient-ID som username, tomt password)."""
    cid = _client_id()
    url = f"{FROST_BASE}{path}"
    # Vi bruker _http for retry + max_bytes, men må wrap-e i en session-loop
    # for å sende Basic Auth. Den enkleste tilnærmingen er å bruke requests
    # direkte siden _http-modulens stream-loop ikke tar auth-argument.
    resp = requests.get(url, params=params, auth=(cid, ""), timeout=180, stream=True)
    # Kopier _http sin max_bytes-sjekk i strømmet form
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            resp.close()
            raise RuntimeError(f"Frost-respons > {max_bytes} bytes — avbryter")
        chunks.append(chunk)
    body = b"".join(chunks)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"Frost {path}: HTTP {resp.status_code} — {body[:300]!r}")
    return json.loads(body)


def fetch_stasjoner() -> None:
    """Hent metadata for alle aktive norske værstasjoner.

    `country=NO` begrenser til Norge. `types=SensorSystem` filtrerer ut
    aggregerte/virtuelle stasjoner. Vi tar med id, navn, koordinater og
    validitets-periode slik at vi kan filtrere ut nedlagte stasjoner i
    standardize.
    """
    out_path = RAW_DIR / "stasjoner.json"
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        out_path.unlink()

    print("  Henter Frost-stasjonsregister...")
    payload = _frost_get("/sources/v0.jsonld", {
        "country": "NO",
        "types": "SensorSystem",
        "fields": "id,name,geometry,validFrom,validTo,county,municipality",
    })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    n = len(payload.get("data", []))
    print(f"  Lagret: {out_path.name} ({n} stasjoner, "
          f"{out_path.stat().st_size / 1024:.0f} KB)")


def fetch_normaler() -> None:
    """Hent klima-normaler (1991-2020) for alle stasjoner.

    Frost krever en eksplisitt sources-liste — wildcard fungerer ikke for
    climatenormals-endepunktet. Vi leser stasjons-IDene fra stasjoner.json
    og chunker spørringer i bolker à 50 for å unngå timeout. Stasjoner uten
    normaldata for 1991-2020 returnerer 404 eller tomt svar — vi logger og
    fortsetter.
    """
    out_path = RAW_DIR / "normaler.json"
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        out_path.unlink()

    # Last stasjons-IDene fra det vi nettopp hentet
    stasjoner_path = RAW_DIR / "stasjoner.json"
    with open(stasjoner_path, encoding="utf-8") as f:
        stasjoner_data = json.load(f)
    alle_ids = [s["id"] for s in stasjoner_data.get("data", []) if s.get("id")]
    print(f"  Henter klima-normaler for {len(alle_ids)} stasjoner i bolker à {SOURCES_CHUNK}...")

    all_rows: list[dict] = []
    n_chunks = (len(alle_ids) + SOURCES_CHUNK - 1) // SOURCES_CHUNK
    for i in range(0, len(alle_ids), SOURCES_CHUNK):
        chunk = alle_ids[i:i + SOURCES_CHUNK]
        chunk_num = i // SOURCES_CHUNK + 1
        try:
            payload = _frost_get("/climatenormals/v0.jsonld", {
                "sources": ",".join(chunk),
                "elements": ",".join(NORMAL_ELEMENTS),
                "period": NORMAL_PERIOD,
            })
            rows = payload.get("data", [])
            all_rows.extend(rows)
            if chunk_num % 5 == 0 or chunk_num == n_chunks:
                print(f"    {chunk_num}/{n_chunks} bolker — totalt {len(all_rows)} rader så langt")
        except requests.HTTPError as e:
            # 404 = ingen stasjoner i bolken har normaldata. Det er normalt
            # for små lokale målepunkter — bare fortsett.
            if "404" in str(e):
                continue
            print(f"    ADVARSEL: bolk {chunk_num} feilet: {e}")
        _http.politely_sleep(0.1)

    # Lagre samlet i samme format som ett enkelt svar — gjør parsingen i
    # standardize-fasen identisk uavhengig av om vi chunket eller ikke.
    payload = {"@context": "https://frost.met.no/schema", "data": all_rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({len(all_rows)} normal-rader, "
          f"{out_path.stat().st_size / 1024:.0f} KB)")


def main() -> None:
    # Validér auth opp-front så vi ikke kaster bort tid på en delvis kjøring
    # uten gyldig klient-ID.
    print("=== Innsamling: MET Frost ===")
    try:
        _client_id()  # validér tidlig
    except FrostAuthMissing as e:
        print(f"  HOPPET OVER: {e}")
        print("=== MET Frost ferdig (hoppet over) ===\n")
        return

    # Kjør begge nedlastingene; logg feil som advarsel slik at hele
    # pipelinen kan fortsette selv om en av delene feiler.
    try:
        fetch_stasjoner()
        fetch_normaler()
    except Exception as e:
        print(f"  ADVARSEL: {type(e).__name__}: {e}")
    print("=== MET Frost ferdig ===\n")


if __name__ == "__main__":
    main()
