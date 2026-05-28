"""
Henter postnummer-polygoner fra Kartverkets WFS og postnummer-registeret fra Bring.

Brukte først Kartverkets adresse-API, men det ville krevd å paginere gjennom
2,5 millioner adresser bare for å få postnummer→kommune-mappingen. Brings
TSV-fil har det samme i én fil med ~5000 rader.
"""

import json
import sys
from pathlib import Path

import requests

# Sørg for at source/ er på sys.path slik at både direktekjøring av denne
# filen og import via pipeline.py finner _http-modulen
_SOURCE_DIR = Path(__file__).parents[1]
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from collect import _http  # noqa: E402

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "kartverket"
RAW_DIR.mkdir(parents=True, exist_ok=True)

POSTNR_WFS = "https://wfs.geonorge.no/skwms1/wfs.postnummeromrader"
BRING_POSTNR_URL = "https://www.bring.no/postnummerregister-ansi.txt"

# Avvis WFS-svar som er mistenkelig små — typisk en feilmelding eller tom respons
MIN_GML_BYTES = 10_000
# WFS-en kan returnere flere hundre MB hvis noe går galt — sett en romslig
# men endelig grense slik at vi ikke spiser hele RAM-et på en korrupt respons.
MAX_WFS_BYTES = 800_000_000


def fetch_postnummer_wfs() -> None:
    """Last ned postnummer-polygonene som GML.

    To ikke-åpenbare detaljer som tok tid å finne ut av:
    1. WFS-en støtter ikke GeoJSON. Må bruke GML 3.2.1 som outputFormat.
    2. srsName må være på URN-form (urn:ogc:def:crs:EPSG::25833). Kortformen
       "EPSG:25833" avvises med HTTP 400 selv om den er gyldig CRS-syntaks.
    """
    out_path = RAW_DIR / "postnummer_wfs.gml"
    # Idempotent: hopp over hvis cachen er gyldig (filen finnes, har minimum
    # størrelse, og slutter på et lukkende XML-tag). En halv-ferdig nedlasting
    # fra forrige kjøring skal trigge ny henting i stedet for å bli akseptert.
    if out_path.exists() and out_path.stat().st_size >= MIN_GML_BYTES \
            and _http.is_valid_gml(out_path, min_bytes=MIN_GML_BYTES):
        print(f"  Allerede hentet: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
        return

    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "app:Postnummerområde",
        "outputFormat": "text/xml; subtype=gml/3.2.1",
        "srsName": "urn:ogc:def:crs:EPSG::25833",
    }

    print("  Henter postnummer-polygoner fra WFS...")
    resp = _http.get(POSTNR_WFS, params=params, timeout=300, max_bytes=MAX_WFS_BYTES)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"WFS feilet: HTTP {resp.status_code} — {resp.text[:300]}")

    # WFS returnerer noen ganger HTTP 200 med XML-feilmelding i body
    # (ExceptionReport) — ekte data starter med <wfs:FeatureCollection>.
    # Sjekker første 500 tegn for å unngå å laste hele responsen i minne to ganger.
    if "ExceptionReport" in resp.text[:500] or "ServiceException" in resp.text[:500]:
        raise RuntimeError(f"WFS returnerte feilmelding: {resp.text[:300]}")

    out_path.write_bytes(resp.content)
    size_kb = out_path.stat().st_size / 1024
    if size_kb * 1024 < MIN_GML_BYTES:
        raise RuntimeError(f"WFS-respons mistenkelig liten ({size_kb:.0f} KB)")
    print(f"  Lagret: {out_path.name} ({size_kb:.0f} KB)")


def fetch_postnummer_registry() -> None:
    """Last ned Brings postnummer-registry og lagre som JSON.

    Filen er TSV (tab-separert) med ISO-8859-1-encoding (Bring bruker
    Windows-1252-lignende koding for å støtte æøå uten BOM). Kolonner:
      postnummer, poststed, kommunenummer, kommunenavn, kategori
    """
    out_path = RAW_DIR / "postnummer_registry.json"
    # Bruk samme parse-baserte validering som SSB-cachen — en korrupt
    # JSON-fil skal trigge ny nedlasting, ikke bli akseptert.
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return

    print("  Henter postnummer-registry fra Bring...")
    # Bring-filen er ~250 KB i dag. Settes til 10 MB for å gi rom for vekst
    # uten å åpne for at en feilkonfigurert kilde sender oss noe gigantisk.
    resp = _http.get(BRING_POSTNR_URL, timeout=60, max_bytes=10_000_000)
    resp.raise_for_status()
    # requests gjetter encoding feil her — fila er ikke UTF-8. Sett eksplisitt
    # før vi leser .text, ellers blir norske tegn ødelagt.
    resp.encoding = "iso-8859-1"

    rows: list[dict] = []
    ignored = 0
    total_lines = 0
    for line in resp.text.splitlines():
        total_lines += 1
        # Hopp over tomme linjer og eventuelle kommentarer (Bring har sjelden
        # noen, men vi er defensive for å unngå parse-feil hvis det dukker opp)
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        # Minst 4 kolonner kreves — kategori er valgfri og noen rader mangler den
        if len(parts) < 4:
            ignored += 1
            continue
        rows.append({
            "postnummer": parts[0].strip(),
            "poststedsnavn": parts[1].strip(),
            "kommunenummer": parts[2].strip(),
            "kommunenavn": parts[3].strip(),
            "kategori": parts[4].strip() if len(parts) > 4 else "",
        })

    if not rows:
        raise RuntimeError("Tomt postnummer-registry fra Bring — sjekk format")

    # Advarsel hvis mistenkelig mange linjer ble droppet — signaliserer at
    # Bring har endret formatet og at parsing er ute av sync med kilden.
    if total_lines > 0 and ignored / total_lines > 0.05:
        print(f"  ADVARSEL: {ignored}/{total_lines} linjer ignorert "
              f"({ignored / total_lines:.1%}) — sjekk om Bring har endret format")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({len(rows)} postnummer)")


def main() -> None:
    print("=== Innsamling: Kartverket + Bring ===")
    fetch_postnummer_wfs()
    fetch_postnummer_registry()
    print("=== Kartverket ferdig ===\n")


if __name__ == "__main__":
    main()
