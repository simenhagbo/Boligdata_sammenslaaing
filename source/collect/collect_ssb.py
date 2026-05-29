"""
Henter SSB-tabeller som tidsserie 2002-2024.

Hver tabell har sin egen start-år (noen begynner først i 2006 eller 2008),
så vi tar overlapp innenfor 2002-2024. Manglende år ender opp som NaN i
sluttdatasettet — preprosessering tar seg av det.

Kjernetabeller (alltid med):
  06035  priser (kvm-pris + omsetninger per boligtype)
  06265  antall boliger per bygningstype
  06266  byggeår-fordeling (chunked pga størrelse)
  06513  bruksareal-fordeling (chunked pga størrelse)
  06913  folkemengde
  12558  inntekt etter skatt (vi tar bare medianen, desil 5)
  09429  utdanningsnivå
  07984  sysselsetting

Utvidelses-tabeller (lagt til i trinn 1 av utvidelse):
  10540  registrerte arbeidsledige (månedlig, 1999-2020 — kapper i 2020)
  09588  flytting mellom kommuner (nettoinnflytting)
  06070  husholdninger (antall + enslige-andel)
  05940  boligbygging (fullførte + igangsatte)
  14674  eiendomsskatt (kommunal sats i promille)
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

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SSB_API = "https://data.ssb.no/api/v0/no/table"
# SSB-svar er typisk <10 MB. 200 MB-grensen er et stort sikkerhetsmargin
# uten å være vid nok til å la en feilet API spise alt RAM-et.
MAX_SSB_BYTES = 200_000_000

# Felles ML-tidsspann. Hver tabell henter sitt overlapp innenfor 2002-2024.
AAR_MIN = 2002
AAR_MAX = 2024


def _aar(start: int, end: int = AAR_MAX) -> list[str]:
    """Liste av år fra max(start, AAR_MIN) til min(end, AAR_MAX) inkludert.

    Hver SSB-tabell har sin egen tidligste år (priser starter 2002, byggeår
    starter 2006 osv). Vi kapper alltid mot AAR_MIN/AAR_MAX slik at vi ikke
    spør om år som ligger utenfor vår valgte ML-tidsperiode. `end` brukes
    for tabeller som ble avsluttet før AAR_MAX (typisk 10540 som stoppet i
    2020) — uten denne får man HTTP 400 for år som ikke finnes i kilden.
    """
    return [str(y) for y in range(max(start, AAR_MIN), min(end, AAR_MAX) + 1)]


def _aar_maaneder(start: int, end: int = AAR_MAX, maaned: str = "11") -> list[str]:
    """Liste av Tid-koder for én bestemt måned per år (SSB-format YYYYMmm).

    10540 publiseres månedlig. Vi tar bare november-tall siden det er den
    tradisjonelle årlige referansemåneden for ledighet og gir oss én
    observasjon per år uten å hente 12x dataene.
    """
    return [f"{y}M{maaned}" for y in range(max(start, AAR_MIN), min(end, AAR_MAX) + 1)]


def fetch_chunked(table_id: str, base_query: dict, aar_list: list[str],
                  chunk_size: int, out_prefix: str) -> None:
    """Splitt en stor SSB-query på Tid for å unngå 800k-datapunkt-grensa.

    SSBs API avviser spørringer som returnerer for mange datapunkter (~800 000).
    For tabeller som 06266 (byggeår) og 06513 (bruksareal) overskrider vi
    grensa hvis vi tar alle år på én gang, så vi kjører flere mindre queries
    og lagrer hver chunk som egen fil. Standardize-fasen leser alle delene
    og slår dem sammen.
    """
    for i in range(0, len(aar_list), chunk_size):
        chunk = aar_list[i:i + chunk_size]
        # Deep copy av query slik at vi ikke muterer den opprinnelige
        # mellom iterasjonene (json-roundtrip er en enkel måte å klone et
        # nested dict på).
        query = json.loads(json.dumps(base_query))
        # Skriv inn denne chunkens årsvalg i Tid-dimensjonen
        for sel in query["query"]:
            if sel["code"] == "Tid":
                sel["selection"]["values"] = chunk
        out_name = f"{out_prefix}_part{i // chunk_size:02d}.json"
        fetch_table(table_id, query, out_name)


def fetch_table(table_id: str, query: dict, out_name: str) -> None:
    """Hent én SSB-tabell og lagre rådata som JSON.

    Bruker POST (ikke GET) fordi query-objektet kan bli stort (flere kilobyte
    med eksplisitte item-lister). SSB returnerer beskrivende feilmeldinger ved
    ugyldige queries — vi viser de første 300 tegnene for debugging.
    """
    out_path = RAW_DIR / out_name
    # Idempotent: hopp over bare hvis cachen er gyldig JSON. En halv-ferdig
    # fil fra forrige Ctrl-C-avbrutt kjøring skal trigge ny henting i stedet
    # for å bli akseptert (og krasje i standardize-fasen senere).
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")
        return
    if out_path.exists():
        # Filen finnes men er korrupt — slett før ny nedlasting
        print(f"  Korrupt cache slettet: {out_path.name}")
        out_path.unlink()

    print(f"  Henter SSB tabell {table_id}...")
    resp = _http.post(
        f"{SSB_API}/{table_id}",
        json=query,
        timeout=180,
        max_bytes=MAX_SSB_BYTES,
    )
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"SSB tabell {table_id}: HTTP {resp.status_code} — {resp.text[:300]}"
        )

    # Defensiv sjekk: SSB skal returnere JSON, men hvis noe gikk galt
    # (proxy-feil, edge-case) får vi heller en tydelig RuntimeError enn en
    # mystisk JSONDecodeError senere.
    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"SSB tabell {table_id} returnerte ikke JSON: {resp.text[:200]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({out_path.stat().st_size / 1024:.0f} KB)")


def fetch_boliger_per_type() -> None:
    # Antall boliger fordelt på bygningstype per kommune × år (starter 2006).
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
    # Kjernetabellen for målet: pris per kvm + antall omsetninger per boligtype.
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


def fetch_arbeidsledighet() -> None:
    """10540: Registrerte arbeidsledige som prosent av arbeidsstyrken.

    Vi tar bare november-tall per år (10540 publiserer månedlig, men november
    er SSBs tradisjonelle årlige referansemåned). Tabellen ble avsluttet etter
    november 2020 — nyere år får NaN i sluttdatasettet.
    """
    # Aldersgruppe 15-74 dekker hele arbeidsstyrken
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Alder", "selection": {"filter": "item", "values": ["15-74"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["RegHeltLedige"]}},
            {"code": "Tid", "selection": {"filter": "item",
                                          "values": _aar_maaneder(2002, 2020, "11")}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("10540", query, "arbeidsledighet_10540.json")


def fetch_flytting() -> None:
    """09588: Nettoinnflytting per kommune × år.

    SSB-kode for nettoinnflytting er kort og ikke beskrivende: `Netto`. Vi tar
    bare den ene metrikken siden in/ut + netto er tre tall hvor to er nok.
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["Netto"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(AAR_MIN)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("09588", query, "flytting_09588.json")


def fetch_husholdninger() -> None:
    """06070: Antall husholdninger fordelt etter type per kommune × år.

    Hentes etter type slik at vi kan beregne både totalt antall og andelen
    enslige (én-person-husholdninger) i standardize. Tabellen begynner i 2005.
    Dimensjons-koden er `HushType` (ikke det mer naturlige `HusholdType`).
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "HushType", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["Husholdninger"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(2005)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("06070", query, "husholdninger_06070.json")


def fetch_boligbygging() -> None:
    """05940: Igangsatte og fullførte boliger per kommune × år.

    21 Byggeareal-koder × 994 kommuner × 23 år × 2 ContentsCode = ~960k
    datapunkter, over SSBs 800k-grense. Chunker derfor på Tid i 5-årsbiter
    (samme mønster som 06266/06513).
    """
    query = {
        "query": [
            {"code": "Region", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "Byggeareal", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["Fullforte", "Igangsatte"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": []}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_chunked("05940", query, _aar(AAR_MIN), chunk_size=5, out_prefix="boligbygging_05940")


def fetch_eiendomsskatt() -> None:
    """14674: Kommunal eiendomsskattesats (promille).

    Tabellen begynner i 2007. KOSTRA-tabeller bruker KOK-prefiks på region-
    dimensjonen og lange kryptiske ContentsCode-navn. `KOSgenskatt0000` er
    den generelle eiendomsskattesatsen per 1000 (promille). Kommuner uten
    eiendomsskatt får manglende verdi.
    """
    query = {
        "query": [
            {"code": "KOKkommuneregion0000", "selection": {"filter": "all", "values": ["*"]}},
            {"code": "ContentsCode", "selection": {"filter": "item",
                                                   "values": ["KOSgenskatt0000"]}},
            {"code": "Tid", "selection": {"filter": "item", "values": _aar(2007)}},
        ],
        "response": {"format": "json-stat2"},
    }
    fetch_table("14674", query, "eiendomsskatt_14674.json")


SSB_FETCHERS = [
    ("Boliger per type (06265)", fetch_boliger_per_type),
    ("Folkemengde (06913)", fetch_folkemengde),
    ("Inntekt (12558)", fetch_inntekt),
    ("Priser (06035)", fetch_priser),
    ("Byggeår (06266)", fetch_byggeaar),
    ("Bruksareal (06513)", fetch_bruksareal),
    ("Utdanning (09429)", fetch_utdanning),
    ("Sysselsetting (07984)", fetch_sysselsetting),
    ("Arbeidsledighet (10540)", fetch_arbeidsledighet),
    ("Flytting (09588)", fetch_flytting),
    ("Husholdninger (06070)", fetch_husholdninger),
    ("Boligbygging (05940)", fetch_boligbygging),
    ("Eiendomsskatt (14674)", fetch_eiendomsskatt),
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
        # Liten pause mellom kall — god skikk overfor SSB selv om de ikke
        # har dokumenterte rate-limits. Imperceptibel for brukeren.
        _http.politely_sleep(0.2)
    if failed:
        print(f"  Fullført med feil i: {failed}")
    print("=== SSB ferdig ===\n")


if __name__ == "__main__":
    main()
