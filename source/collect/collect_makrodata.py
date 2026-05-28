"""
Henter åpne nasjonale makrodata for investeringsavkastnings-modellen.

Tre kilder, alle åpne og uten autentisering:

  Norges Bank   styringsrente (årssnitt av daglige observasjoner)
  SSB 03013     konsumprisindeks (KPI, månedlig — vi tar årssnitt i standardize)
  SSB 10748     gjennomsnittlig rente på nye boliglån (månedlig fra 2013)

Datasettets ML-tidsserie er 2002-2024. KPI og styringsrente dekker hele
perioden; boliglånsrenten begynner i desember 2013 og er NaN for tidligere
år (bevisst valg — ingen ekstrapolering).
"""

import json
import sys
from pathlib import Path

import requests

# Sørg for at source/ er på sys.path slik at både direktekjøring og import
# via pipeline.py finner _http-modulen
_SOURCE_DIR = Path(__file__).parents[1]
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from collect import _http  # noqa: E402

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "makrodata"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SSB_API = "https://data.ssb.no/api/v0/no/table"
# Norges Bank SDMX-JSON. IR=Interest Rates, A=Annual, KPRA=Key Policy Rate Average,
# SD=Sight Deposit (= styringsrenten siden 2011), R=Rate.
NORGES_BANK_STYRINGSRENTE = (
    "https://data.norges-bank.no/api/data/IR/A.KPRA.SD.R"
    "?format=sdmx-json&startPeriod=2002&endPeriod=2030"
)

MAX_BYTES = 50_000_000


def fetch_styringsrente() -> None:
    """Hent årssnitt av Norges Banks styringsrente fra 2002 og framover.

    SDMX-JSON-format. Vi lagrer den hele responsen og parser den i
    standardize-fasen. `endPeriod=2030` slik at framtidige år blir hentet
    automatisk når API-et får nye observasjoner.
    """
    out_path = RAW_DIR / "styringsrente_norgesbank.json"
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        out_path.unlink()

    print("  Henter styringsrente fra Norges Bank...")
    resp = _http.get(NORGES_BANK_STYRINGSRENTE, timeout=120, max_bytes=MAX_BYTES)
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"Norges Bank HTTP {resp.status_code}: {resp.text[:300]}"
        )

    payload = resp.json()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def _ssb_fetch(table_id: str, query: dict, out_name: str) -> None:
    """Felles SSB-fetch — samme mønster som collect_ssb.fetch_table men i lokal
    raw_data/makrodata-mappe slik at makrodata-kildene er gruppert sammen.
    """
    out_path = RAW_DIR / out_name
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        print(f"  Korrupt cache slettet: {out_path.name}")
        out_path.unlink()

    print(f"  Henter SSB tabell {table_id}...")
    resp = _http.post(
        f"{SSB_API}/{table_id}",
        json=query,
        timeout=180,
        max_bytes=MAX_BYTES,
    )
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"SSB tabell {table_id}: HTTP {resp.status_code} — {resp.text[:300]}"
        )
    payload = resp.json()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def fetch_kpi() -> None:
    """03013: Konsumprisindeks (2015=100). Vi tar totalindeks for alle måneder.

    `KpiIndMnd` er indeksen (2015=100) — vi kan utlede både årsendring og
    realprisjusterte verdier fra den. Vi bruker filter=all på Tid slik at
    nye måneder hentes automatisk uten å endre koden.
    """
    query = {
        "query": [
            {"code": "Konsumgrp", "selection": {"filter": "item", "values": ["TOTAL"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["KpiIndMnd"]}},
            {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    _ssb_fetch("03013", query, "kpi_03013.json")


def fetch_boliglaansrente() -> None:
    """10748: Gjennomsnittlig rente på nye boliglån, månedlig fra 2013M12.

    Vi tar `Utlanstype=04` (totale utlån med pant i bolig), `Sektor=04b`
    (husholdninger) og `Rentebinding=99` (totalt — alle bindingstider veiet).
    Det gir oss én rad per måned som er den representative husholdnings-
    boliglånsrenten. Pre-2014-år blir NaN i sluttdatasettet siden tabellen
    ikke har data — det er bevisst (ingen ekstrapolering).
    """
    query = {
        "query": [
            {"code": "Utlanstype", "selection": {"filter": "item", "values": ["04"]}},
            {"code": "Sektor", "selection": {"filter": "item", "values": ["04b"]}},
            {"code": "Rentebinding", "selection": {"filter": "item", "values": ["99"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["RenterNyeBolig"]}},
            {"code": "Tid", "selection": {"filter": "all", "values": ["*"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    _ssb_fetch("10748", query, "boliglaansrente_10748.json")


FETCHERS = [
    ("Styringsrente (Norges Bank)", fetch_styringsrente),
    ("KPI (SSB 03013)", fetch_kpi),
    ("Boliglånsrente (SSB 10748)", fetch_boliglaansrente),
]


def main() -> None:
    print("=== Innsamling: Makrodata ===")
    failed: list[str] = []
    for label, fn in FETCHERS:
        try:
            fn()
        except Exception as e:
            print(f"  ADVARSEL: {label} feilet: {e}")
            failed.append(label)
        _http.politely_sleep(0.2)
    if failed:
        print(f"  Fullført med feil i: {failed}")
    print("=== Makrodata ferdig ===\n")


if __name__ == "__main__":
    main()
