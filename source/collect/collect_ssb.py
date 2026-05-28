"""
Henter tre SSB-tabeller via JSON-stat API-et.

  06265 — antall boliger per type per kommune
  07459 — folkemengde per kommune
  12558 — inntekt etter skatt per kommune

SSB leverer på kommunenivå; kobling til postnummer skjer i standardize-fasen.
"""

import json
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SSB_API = "https://data.ssb.no/api/v0/no/table"


def fetch_table(table_id: str, query: dict, out_name: str) -> None:
    """Hent én SSB-tabell. POST fordi query-objektet kan bli stort."""
    out_path = RAW_DIR / out_name
    if out_path.exists():
        print(f"  Allerede hentet: {out_path.name}")
        return

    print(f"  Henter SSB tabell {table_id}...")
    resp = requests.post(
        f"{SSB_API}/{table_id}",
        json=query,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if resp.status_code >= 400:
        # SSB returnerer beskrivende feilmelding ved feil query — vis den
        raise requests.HTTPError(
            f"SSB tabell {table_id}: HTTP {resp.status_code} — {resp.text[:300]}"
        )

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"SSB tabell {table_id} returnerte ikke JSON: {resp.text[:200]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def fetch_boliger_per_type() -> None:
    # BygnType: 01=enebolig, 02=tomanns, 03=rekkehus, 04=blokk, 05=bofellesskap, 999=annet
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BygnType", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("06265", query, "boliger_per_type_06265.json")


def fetch_folkemengde() -> None:
    # Summerer over Kjonn og Alder i std-fasen for å få totalbefolkning per kommune
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kjonn", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Alder", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Personer1"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("07459", query, "folkemengde_07459.json")


def fetch_inntekt() -> None:
    # InntektSkatt "00S" = inntekt etter skatt
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "InntektSkatt", "selection": {"filter": "item", "values": ["00S"]}},
            {"code": "Tid", "selection": {"filter": "top", "values": ["1"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("12558", query, "inntekt_12558.json")


SSB_FETCHERS = [
    ("Boliger per type (06265)", fetch_boliger_per_type),
    ("Folkemengde (07459)", fetch_folkemengde),
    ("Inntekt (12558)", fetch_inntekt),
]


def main() -> None:
    print("=== Innsamling: SSB ===")
    failed: list[str] = []
    for label, fn in SSB_FETCHERS:
        try:
            fn()
        except Exception as e:
            # En enkelt tabell-feil skal ikke stoppe de andre
            print(f"  ADVARSEL: {label} feilet: {e}")
            failed.append(label)
    if failed:
        print(f"  Fullført med feil i: {failed}")
    print("=== SSB ferdig ===\n")


if __name__ == "__main__":
    main()
