"""
Kjører hele datapipelinen.

Bruk:
  python source/pipeline.py                       alle faser
  python source/pipeline.py --collect             bare hent rådata
  python source/pipeline.py --standardize         bare vask
  python source/pipeline.py --merge               bare slå sammen
  python source/pipeline.py --include-matrikkelen tar med Matrikkelen (treg)
  python source/pipeline.py --export-csv          eksporter også til CSV
"""

import argparse
import sys
import time
from pathlib import Path

# Legg til source/-mappen i Python-stien slik at vi kan importere fra
# undermodulene (collect/, standardize/, merge/) når dette skriptet kjøres
# direkte, ikke som en pakke
sys.path.insert(0, str(Path(__file__).parent))

from collect.collect_kartverket import main as collect_kartverket
from collect.collect_matrikkelen import main as collect_matrikkelen
from collect.collect_ssb import main as collect_ssb
from collect.collect_entur import main as collect_entur
from collect.collect_makrodata import main as collect_makrodata
from collect.collect_met_frost import main as collect_met_frost
from standardize.standardize_kartverket import main as std_kartverket
from standardize.standardize_ssb import main as std_ssb
from standardize.standardize_matrikkelen import main as std_matrikkelen
from standardize.standardize_geofeatures import main as std_geofeatures
from standardize.standardize_entur import main as std_entur
from standardize.standardize_makrodata import main as std_makrodata
from standardize.standardize_met_frost import main as std_met_frost
from merge.merge_and_quality import main as merge_all
from export_csv import export as export_to_csv


def _run(label: str, fn, critical: bool = False) -> bool:
    """Kjør én fase med header, tidtaking og feilhåndtering.

    `critical=True` betyr at en feil i denne fasen stopper hele pipelinen
    (typisk Kartverket og merge — uten dem kan vi ikke bygge datasettet).
    Andre faser (SSB, Matrikkelen) er valgfrie — feil logges som ADVARSEL og
    pipelinen fortsetter med færre kolonner.
    """
    t0 = time.time()
    print(f"\n{'='*60}\nFASE: {label}\n{'='*60}")
    try:
        fn()
        print(f"  Tid: {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"FEIL i fasen '{label}': {type(e).__name__}: {e}")
        if critical:
            raise
        print("Pipelinen fortsetter uten denne kilden.")
        return False


def run_collect(include_matrikkelen: bool = False) -> None:
    """Hent rådata fra alle kilder. Kartverket er kritisk siden alle andre
    kilder kobles inn via postnummer-mappingen derfra.
    """
    _run("Innsamling — Kartverket", collect_kartverket, critical=True)
    _run("Innsamling — SSB", collect_ssb)
    _run("Innsamling — Entur", collect_entur)
    _run("Innsamling — Makrodata", collect_makrodata)
    _run("Innsamling — MET Frost (opt-in via FROST_CLIENT_ID)", collect_met_frost)
    if include_matrikkelen:
        _run("Innsamling — Matrikkelen (treg)", collect_matrikkelen)


def run_standardize(include_matrikkelen: bool = False) -> None:
    """Vask og normaliser rådata til Parquet med felles nøkler.
    Kartverket må kjøres først fordi postnummer-kommune-mappingen brukes som
    join-nøkkel av de andre kildene. Geofeatures avhenger av geometri-
    Parquet og må derfor også komme etter Kartverket.
    """
    _run("Standardisering — Kartverket", std_kartverket, critical=True)
    _run("Standardisering — Geofeatures", std_geofeatures)
    _run("Standardisering — SSB", std_ssb)
    _run("Standardisering — Entur", std_entur)
    _run("Standardisering — Makrodata", std_makrodata)
    _run("Standardisering — MET Frost", std_met_frost)
    if include_matrikkelen:
        _run("Standardisering — Matrikkelen", std_matrikkelen)


def run_merge() -> None:
    """Slå alle standardiserte kilder sammen + kvalitetskontroll."""
    _run("Sammenslåing + kvalitetskontroll", merge_all, critical=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Boligdata sammenslåing pipeline")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--include-matrikkelen",
        action="store_true",
        help="Ta med Matrikkelen-Bygningspunkt (legger til ~30-60 min)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Eksporter sluttdatasettet til CSV etter merge (uten geometri)",
    )
    args = parser.parse_args()

    # Hvis ingen fase-flagg er satt, kjør alle fasene i rekkefølge
    run_all = not any([args.collect, args.standardize, args.merge])
    t0 = time.time()

    if run_all or args.collect:
        run_collect(include_matrikkelen=args.include_matrikkelen)
    if run_all or args.standardize:
        run_standardize(include_matrikkelen=args.include_matrikkelen)
    if run_all or args.merge:
        run_merge()

    # CSV-eksport kjøres bare på eksplisitt forespørsel. Parquet er hovedformat,
    # CSV er for når du vil åpne dataene i Excel eller dele med noen som
    # ikke bruker Python.
    if args.export_csv:
        _run("Eksport — CSV", lambda: export_to_csv(med_geometri=False))

    print(f"\n{'='*60}\nPipeline ferdig på {time.time() - t0:.1f}s\n{'='*60}\n")


if __name__ == "__main__":
    main()
