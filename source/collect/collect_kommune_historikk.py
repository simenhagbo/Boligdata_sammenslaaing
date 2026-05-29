"""
Henter historiske kommune-kodeendringer fra SSBs Klass-API.

Norge har slått sammen og delt kommuner flere ganger (særlig reformen i 2020).
SSB rapporterer eldre år med datidens kommunekoder, mens vi mapper postnummer
til 2024-koder. Resultatet er at eldre SSB-data ikke matcher og faller ut.

Denne kilden henter alle kodeendringer 2002-2024 fra Klass-klassifikasjon 131
("Standard for kommuneinndeling"). standardize-fasen bygger en mapping fra
historisk kode til 2024-kode og re-aggregerer SSB-dataene.

Output:
  raw_data/kommune_historikk/kodeendringer.json
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

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "kommune_historikk"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Klass-klassifikasjon 131 = kommuneinndeling. changes-endepunktet gir alle
# gamle→nye kodeendringer i perioden, med dato for hver endring.
KLASS_CHANGES = "https://data.ssb.no/api/klass/v1/classifications/131/changes"
FROM_DATE = "2002-01-01"
# TO_DATE må være 2025 (ikke 2024) for å fange reformen 2024-01-01, der
# fylkene Viken/Vestfold-Telemark/Troms-Finnmark ble splittet tilbake og
# kommunekodene endret seg (f.eks. Tønsberg 3803 → 3905). Postnummer-
# mappingen bruker 2024-koder, så SSB-historikken må lande på samme koder.
TO_DATE = "2025-01-01"


def fetch_kodeendringer() -> None:
    """Hent kommune-kodeendringer 2002-2024 og lagre som JSON."""
    # Idempotent: hopp over hvis gyldig cache finnes
    out_path = RAW_DIR / "kodeendringer.json"
    if _http.is_valid_json(out_path):
        print(f"  Allerede hentet: {out_path.name}")
        return
    if out_path.exists():
        out_path.unlink()

    # Query-string-formen (?from=&to=) svarer JSON som default
    print("  Henter kommune-kodeendringer fra Klass-API...")
    resp = _http.get(
        f"{KLASS_CHANGES}?from={FROM_DATE}&to={TO_DATE}",
        timeout=60,
        max_bytes=10_000_000,
    )
    if resp.status_code >= 400:
        raise requests.HTTPError(
            f"Klass HTTP {resp.status_code}: {resp.text[:300]}"
        )
    payload = resp.json()

    # Lagre rådata uendret; mapping bygges i standardize
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    n = len(payload.get("codeChanges", []))
    print(f"  Lagret: {out_path.name} ({n} kodeendringer)")


def main() -> None:
    # Én enkel fetch — Klass-API er åpent og krever ingen nøkkel
    print("=== Innsamling: Kommune-historikk (Klass) ===")
    try:
        fetch_kodeendringer()
    except Exception as e:
        # Ikke kritisk: uten historikk faller pipelinen tilbake til 2024-koder
        print(f"  ADVARSEL: {type(e).__name__}: {e}")
    print("=== Kommune-historikk ferdig ===\n")


if __name__ == "__main__":
    main()
