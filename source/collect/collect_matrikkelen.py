"""
Henter Matrikkelen-Bygningspunkt (4,4M bygg) fra Kartverkets WFS.

To begrensninger som styrer designet her:

1. WFS-en støtter bare GML 3.2.1 (ikke GeoJSON), og srsName må være på URN-form.
2. CQL_FILTER blir ignorert. Det er ikke dokumentert noe sted, men man ser det
   ved å filtrere på en kommune og likevel få bygg fra andre kommuner i svaret.
   Konsekvensen er at man ikke kan be om "bare fylke X" — man må paginere
   gjennom hele datasettet.

Datasettet gir kun bygningstype + kommunenummer (ikke BRA eller byggeår — det
ligger bak den lisensierte Matrikkelen-tjenesten). Det er derfor dette er
opt-in i pipelinen; vi får tilsvarende info fra SSB tabell 06265.
"""

import re
import sys
import time
from pathlib import Path

import requests

# Sørg for at source/ er på sys.path slik at både direktekjøring og import
# via pipeline.py finner _http-modulen
_SOURCE_DIR = Path(__file__).parents[1]
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

from collect import _http  # noqa: E402

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "matrikkelen"
RAW_DIR.mkdir(parents=True, exist_ok=True)

WFS_BASE = "https://wfs.geonorge.no/skwms1/wfs.matrikkelen-bygningspunkt"
# Hver side er typisk <50 MB. Grensen er romslig nok til at faktiske svar
# alltid passerer, men holder en kompromittert kilde fra å spise alt minne.
MAX_PAGE_BYTES = 200_000_000

# Beholdt fra tidligere — brukes ikke som filter (CQL ignoreres), men noterer
# fylkesinndelingen for fremtidig referanse
FYLKER = ["03", "11", "15", "18", "31", "32", "33", "34", "39", "40", "42", "46", "50", "55", "56"]

PAGE_SIZE = 5000  # faller tilbake til 1000 ved 400/413
MIN_GML_BYTES = 500


def _count_returned(text: str) -> int:
    """Hent numberReturned-attributtet fra root-elementet i WFS-responsen.

    WFS 2.0 setter dette attributtet på <wfs:FeatureCollection>-elementet og
    det forteller hvor mange features som faktisk ble returnert i denne
    sidens respons. Vi bruker det til å vite når vi har nådd slutten av
    paginering (returned < page_size = siste side).
    """
    m = re.search(r'numberReturned="(\d+)"', text[:4000])
    return int(m.group(1)) if m else 0


def _is_exception(text: str) -> bool:
    """Sjekk om responsen er en ExceptionReport (WFS-feil) i stedet for data."""
    head = text[:500]
    return "ExceptionReport" in head or "ServiceException" in head


def _fetch_page(fylke_nr: str, start_index: int, page_size: int) -> bytes:
    """Hent én side bygninger som GML-bytes. Kaster ved XML-feilmelding.

    propertyName begrenser feltene som returneres — vi trenger bare
    bygningstype og kommunenummer. Det reduserer responsstørrelsen
    betraktelig sammenlignet med å få alle 15+ feltene per bygning.
    """
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeName": "app:Bygning",
        "outputFormat": "text/xml; subtype=gml/3.2.1",
        "srsName": "urn:ogc:def:crs:EPSG::25833",
        "propertyName": "bygningstype,kommunenummer",
        "count": page_size,
        "startIndex": start_index,
        # CQL_FILTER beholdes selv om den ignoreres i denne WFS-en — hvis
        # Kartverket fikser bug-en, kjører scriptet effektivt ut av boksen
        "CQL_FILTER": f"strStartsWith(kommunenummer,'{fylke_nr}')=true",
    }
    resp = _http.get(WFS_BASE, params=params, timeout=300, max_bytes=MAX_PAGE_BYTES)
    resp.raise_for_status()
    if _is_exception(resp.text):
        raise RuntimeError(f"WFS returnerte feilmelding: {resp.text[:400]}")
    return resp.content


def fetch_bygninger_for_fylke(fylke_nr: str) -> None:
    """Paginer gjennom bygninger og lagre hver side som egen .gml-fil.

    Vi lagrer sidene som separate filer fordi de samlede GML-dataene blir
    flere GB hvis vi samler alt i én fil. Sidene merges igjen i standardize-
    fasen. En "_done"-markørfil markerer at hentingen er fullført — slik
    kan vi avbryte og fortsette uten å miste fremgangen.
    """
    fylke_dir = RAW_DIR / f"fylke_{fylke_nr}"
    fylke_dir.mkdir(exist_ok=True)
    done_marker = fylke_dir / "_done"

    if done_marker.exists():
        print(f"    Allerede hentet: fylke_{fylke_nr}")
        return

    start_index = 0
    page_size = PAGE_SIZE
    page_num = 0
    total = 0

    while True:
        try:
            content = _fetch_page(fylke_nr, start_index, page_size)
        except requests.HTTPError as e:
            # WFS-en kan avvise store sider med 400/413 ved overbelastning.
            # Fall tilbake til mindre page_size (1000) og fortsett.
            if page_size > 1000 and e.response is not None and e.response.status_code in (400, 413):
                print(f"    page_size {page_size} avvist, prøver 1000...")
                page_size = 1000
                continue
            raise

        # Mistenkelig liten respons = sannsynligvis ikke mer data å hente
        if len(content) < MIN_GML_BYTES:
            break

        returned = _count_returned(content.decode("utf-8", errors="ignore"))
        if returned == 0:
            break

        (fylke_dir / f"page_{page_num:04d}.gml").write_bytes(content)
        total += returned

        # Hvis vi fikk færre features enn forespurt, er vi på siste side
        if returned < page_size:
            break

        page_num += 1
        start_index += page_size
        # Liten pause for å være snill mot WFS-tjenesten
        time.sleep(0.1)

        # Progress hvert tiende batch så brukeren ser at noe skjer
        if page_num % 10 == 0:
            print(f"    fylke {fylke_nr}: {total:,} bygninger så langt...")

    done_marker.write_text(f"total={total}\npages={page_num + 1}\n")
    print(f"    Lagret fylke_{fylke_nr}: {total:,} bygninger på {page_num + 1} sider")


def main() -> None:
    # Iterer fylke for fylke. En feil i ett fylke skal ikke stoppe hele kjøringen
    # — vi samler opp og rapporter etterpå slik at brukeren kan retry de som feilet.
    print("=== Innsamling: Matrikkelen-Bygningspunkt ===")
    print("  Dette tar 30-60 minutter — 4,4M bygg paginert i 5000 av gangen")
    failed: list[str] = []
    for fylke in FYLKER:
        print(f"  Fylke {fylke}...")
        try:
            fetch_bygninger_for_fylke(fylke)
        except Exception as e:
            print(f"    ADVARSEL: Feil for fylke {fylke}: {e}")
            failed.append(fylke)
    # Rapporter samlet status på slutten
    if failed:
        print(f"  Fullført med feil i fylker: {failed}")
    print("=== Matrikkelen ferdig ===\n")


if __name__ == "__main__":
    main()
