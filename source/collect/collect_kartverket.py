"""
Henter postnummer-polygoner fra Kartverkets WFS og postnummer-registeret fra Bring.

Brukte først Kartverkets adresse-API, men det ville krevd å paginere gjennom
2,5 millioner adresser bare for å få postnummer→kommune-mappingen. Brings
TSV-fil har det samme i én fil med ~5000 rader.
"""

import json
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "kartverket"
RAW_DIR.mkdir(parents=True, exist_ok=True)

POSTNR_WFS = "https://wfs.geonorge.no/skwms1/wfs.postnummeromrader"
BRING_POSTNR_URL = "https://www.bring.no/postnummerregister-ansi.txt"
MIN_GML_BYTES = 10_000


def fetch_postnummer_wfs() -> None:
    """Last ned postnummer-polygonene som GML.

    WFS-en støtter ikke GeoJSON, så vi må bruke GML 3.2.1. srsName må være
    på URN-form (urn:ogc:def:crs:EPSG::25833) — kortformen avvises med 400.
    """
    out_path = RAW_DIR / "postnummer_wfs.gml"
    if out_path.exists() and out_path.stat().st_size >= MIN_GML_BYTES:
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
    resp = requests.get(POSTNR_WFS, params=params, timeout=300)
    if resp.status_code >= 400:
        raise requests.HTTPError(f"WFS feilet: HTTP {resp.status_code} — {resp.text[:300]}")

    # WFS-feil kommer som <ExceptionReport>, ekte data starter med <wfs:FeatureCollection>
    if "ExceptionReport" in resp.text[:500] or "ServiceException" in resp.text[:500]:
        raise RuntimeError(f"WFS returnerte feilmelding: {resp.text[:300]}")

    out_path.write_bytes(resp.content)
    size_kb = out_path.stat().st_size / 1024
    if size_kb * 1024 < MIN_GML_BYTES:
        raise RuntimeError(f"WFS-respons mistenkelig liten ({size_kb:.0f} KB)")
    print(f"  Lagret: {out_path.name} ({size_kb:.0f} KB)")


def fetch_postnummer_registry() -> None:
    """Last ned Brings postnummer-registry og lagre som JSON.

    Bring serverer TSV i ISO-8859-1 med kolonnene:
    postnummer, poststed, kommunenummer, kommunenavn, kategori.
    """
    out_path = RAW_DIR / "postnummer_registry.json"
    if out_path.exists():
        print(f"  Allerede hentet: {out_path.name}")
        return

    print("  Henter postnummer-registry fra Bring...")
    resp = requests.get(BRING_POSTNR_URL, timeout=60)
    resp.raise_for_status()
    resp.encoding = "iso-8859-1"

    rows: list[dict] = []
    for line in resp.text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
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
