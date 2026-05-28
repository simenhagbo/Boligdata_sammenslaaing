"""
Henter togstasjoner fra Entur sin Journey-Planner GraphQL.

Entur eier nasjonalt stoppestedregister (NSR) og har en åpen GraphQL-API
uten autentisering — kun en ET-Client-Name-header som god skikk. Vi henter
alle togstasjoner (transportMode=rail) i Norge ved å dele landet i BBOX-
biter (en enkelt BBOX over hele Norge returnerer for mange stops).

Resultatet er en JSON-fil med tog-stasjons-koordinater som vi i
standardize-fasen bruker til å beregne avstand fra hvert postnummer til
nærmeste togstasjon.
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

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "entur"
RAW_DIR.mkdir(parents=True, exist_ok=True)

ENTUR_GRAPHQL = "https://api.entur.io/journey-planner/v3/graphql"

# Del Norge i 6 BBOX-biter slik at hver query holder seg under ~10k stops.
# Hele Norge i én call returnerer for mange og er sårbar for timeout.
# (min_lat, max_lat, min_lon, max_lon, navn) — overlappende ved grensene er OK
# siden vi dedupliserer på stop-id i standardize.
NORGE_BBOX = [
    (57.5, 60.0,  4.0, 11.5, "sor-vest"),
    (57.5, 60.0, 11.5, 17.0, "sor-ost"),
    (60.0, 63.5,  4.0, 11.5, "vest"),
    (60.0, 63.5, 11.5, 18.0, "ost"),
    (63.5, 67.0,  6.0, 16.0, "midt"),
    (67.0, 71.5, 12.0, 31.5, "nord"),
]


def fetch_togstasjoner() -> None:
    """Hent alle togstasjoner i Norge og lagre som én JSON-fil.

    BBOX-deling sikrer at hver query er liten nok til å fullføre under
    timeout. Vi henter alle stops og filtrerer på rail-modus i standardize
    siden GraphQL-skjemaet ikke har transportMode-filter på BBOX-query.
    """
    out_path = RAW_DIR / "togstasjoner.json"
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        out_path.unlink()

    alle_stops: list[dict] = []
    for min_lat, max_lat, min_lon, max_lon, navn in NORGE_BBOX:
        query = f"""
        {{
          stopPlacesByBbox(
            minimumLatitude: {min_lat}
            maximumLatitude: {max_lat}
            minimumLongitude: {min_lon}
            maximumLongitude: {max_lon}
            filterByInUse: true
          ) {{
            id
            name
            latitude
            longitude
            transportMode
          }}
        }}
        """
        print(f"  Henter Entur BBOX {navn}...")
        resp = _http.post(
            ENTUR_GRAPHQL,
            json={"query": query},
            timeout=120,
            max_bytes=100_000_000,
        )
        if resp.status_code >= 400:
            raise requests.HTTPError(f"Entur HTTP {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        if "errors" in data:
            raise RuntimeError(f"Entur GraphQL-feil: {data['errors'][:1]}")

        # Vi filtrerer ut togstasjoner allerede her for å redusere størrelsen
        # på cachen — transportMode er en liste, sjekk om "rail" er med
        stops = data.get("data", {}).get("stopPlacesByBbox", [])
        togstops = [s for s in stops
                    if s.get("transportMode") and "rail" in s["transportMode"]]
        print(f"    {len(stops)} stops, {len(togstops)} togstasjoner")
        alle_stops.extend(togstops)

    # Deduplisér på stop-id (BBOX-bitene overlapper litt ved grensene)
    seen: set[str] = set()
    unike: list[dict] = []
    for s in alle_stops:
        if s["id"] not in seen:
            seen.add(s["id"])
            unike.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unike, f, ensure_ascii=False, indent=2)
    print(f"  Lagret: {out_path.name} ({len(unike)} unike togstasjoner)")


def main() -> None:
    print("=== Innsamling: Entur ===")
    fetch_togstasjoner()
    print("=== Entur ferdig ===\n")


if __name__ == "__main__":
    main()
