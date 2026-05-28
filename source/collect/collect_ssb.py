"""
Henter SSB-tabeller som tidsserie 2002-2024.

Hver tabell har sin egen start-år (noen begynner først i 2006 eller 2008),
så vi tar overlapp innenfor 2002-2024. Manglende år ender opp som NaN i
sluttdatasettet — preprosessering tar seg av det.

  06035  priser (kvm-pris + omsetninger per boligtype)
  06265  antall boliger per bygningstype
  06266  byggeår-fordeling (chunked pga størrelse)
  06513  bruksareal-fordeling (chunked pga størrelse)
  06913  folkemengde
  12558  inntekt etter skatt (vi tar bare medianen, desil 5)
  09429  utdanningsnivå
  07984  sysselsetting
"""

import json
from pathlib import Path

import requests

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SSB_API = "https://data.ssb.no/api/v0/no/table"

# Felles ML-tidsspann. Hver tabell henter sitt overlapp innenfor 2002-2024.
AAR_MIN = 2002
AAR_MAX = 2024


def _aar(start: int) -> list[str]:
    """Liste av år fra max(start, AAR_MIN) til AAR_MAX inkludert."""
    return [str(y) for y in range(max(start, AAR_MIN), AAR_MAX + 1)]


def fetch_chunked(table_id: str, base_query: dict, aar_list: list[str],
                  chunk_size: int, out_prefix: str) -> None:
    """Splitt en stor SSB-query på Tid for å unngå 800k-datapunkt-grensa."""
    for i in range(0, len(aar_list), chunk_size):
        chunk = aar_list[i:i + chunk_size]
        # Lag en kopi av query med kun denne chunkens år for Tid
        query = json.loads(json.dumps(base_query))
        for sel in query["query"]:
            if sel["code"] == "Tid":
                sel["selection"]["values"] = chunk
        out_name = f"{out_prefix}_part{i // chunk_size:02d}.json"
        fetch_table(table_id, query, out_name)


def fetch_table(table_id: str, query: dict, out_name: str) -> None:
    """Hent én SSB-tabell. POST fordi query-objektet kan bli stort."""
    out_path = RAW_DIR / out_name
    if out_path.exists():
        print(f"  Allerede hentet: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
        return

    print(f"  Henter SSB tabell {table_id}...")
    resp = requests.post(
        f"{SSB_API}/{table_id}",
        json=query,
        headers={"Content-Type": "application/json"},
        timeout=180,
    )
    if resp.status_code >= 400:
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
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BygnType", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(2006)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("06265", query, "boliger_per_type_06265.json")


def fetch_folkemengde() -> None:
    # Bytter til 06913 fra 07459 fordi sistnevnte blir for stor med alder×kjønn
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Folkemengde"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(AAR_MIN)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("06913", query, "folkemengde_06913.json")


def fetch_inntekt() -> None:
    # 12558 gir verdier per desil 1-10. Vi tar bare desil 5 (medianen) i kroner.
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "InntektSkatt", "selection": {"filter": "item", "values": ["00S"]}},
            {"code": "Desiler", "selection": {"filter": "item", "values": ["05"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["VerdiDesil"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(2005)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("12558", query, "inntekt_12558.json")


def fetch_priser() -> None:
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Boligtype", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(AAR_MIN)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("06035", query, "priser_06035.json")


def fetch_byggeaar() -> None:
    # Må chunkes på Tid — full query ville blitt ~1.5M datapunkter.
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BygnType", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BygnAr", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": []}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_chunked("06266", query, _aar(2006), chunk_size=5, out_prefix="byggeaar_06266")


def fetch_bruksareal() -> None:
    # Samme størrelsesproblem som byggeår — chunkes
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BygnType", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "BruksAreal", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": []}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_chunked("06513", query, _aar(2007), chunk_size=5, out_prefix="bruksareal_06513")


def fetch_utdanning() -> None:
    # Nivaa: 01=grunnskole, 02a=videregående, 11=fagskole, 03a=UH kort,
    # 04a=UH lang, 09=uoppgitt. Vi henter prosent for begge kjønn samlet.
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Nivaa", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["PersonerProsent"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(AAR_MIN)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("09429", query, "utdanning_09429.json")


def fetch_sysselsetting() -> None:
    # Bare totaltall: alle næringer (00-99), alle yrkesaktive (15-74)
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "NACE2007", "selection": {"filter": "item", "values": ["00-99"]}},
            {"code": "Kjonn", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Alder", "selection": {"filter": "item", "values": ["15-74"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["Sysselsatte"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(2008)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("07984", query, "sysselsetting_07984.json")


SSB_FETCHERS = [
    ("Boliger per type (06265)", fetch_boliger_per_type),
    ("Folkemengde (06913)", fetch_folkemengde),
    ("Inntekt (12558)", fetch_inntekt),
    ("Priser (06035)", fetch_priser),
    ("Byggeår (06266)", fetch_byggeaar),
    ("Bruksareal (06513)", fetch_bruksareal),
    ("Utdanning (09429)", fetch_utdanning),
    ("Sysselsetting (07984)", fetch_sysselsetting),
]


def main() -> None:
    print(f"=== Innsamling: SSB ({AAR_MIN}-{AAR_MAX}) ===")
    failed: list[str] = []
    for label, fn in SSB_FETCHERS:
        try:
            fn()
        except Exception as e:
            print(f"  ADVARSEL: {label} feilet: {e}")
            failed.append(label)
    if failed:
        print(f"  Fullført med feil i: {failed}")
    print("=== SSB ferdig ===\n")


if __name__ == "__main__":
    main()
