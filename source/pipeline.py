"""
Kjører hele datapipelinen.

Bruk:
  python source/pipeline.py                       alle faser
  python source/pipeline.py --collect             bare hent rådata
  python source/pipeline.py --standardize         bare vask
  python source/pipeline.py --merge               bare slå sammen
  python source/pipeline.py --include-matrikkelen tar med Matrikkelen (treg)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect.collect_kartverket import main as collect_kartverket
from collect.collect_matrikkelen import main as collect_matrikkelen
from collect.collect_ssb import main as collect_ssb
from collect.collect_eiendom_norge import main as collect_eiendom_norge
from standardize.standardize_kartverket import main as std_kartverket
from standardize.standardize_ssb import main as std_ssb
from standardize.standardize_eiendom_norge import main as std_eiendom_norge
from standardize.standardize_matrikkelen import main as std_matrikkelen
from standardize.standardize_geofeatures import main as std_geofeatures
from merge.merge_and_quality import main as merge_all


def _run(label: str, fn, critical: bool = False) -> bool:
    """Kjør én fase. Kritiske faser stopper pipelinen ved feil, andre logger og fortsetter."""
    t0 = time.time()
    print(f"\n{'='*60}\nFASE: {label}\n{'='*60}")
    try:
        fn()
        print(f"  Tid: {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  FEIL i fasen '{label}': {type(e).__name__}: {e}")
        if critical:
            raise
        print("  Pipelinen fortsetter uten denne kilden.")
        return False


def run_collect(include_matrikkelen: bool = False) -> None:
    _run("Innsamling — Kartverket", collect_kartverket, critical=True)
    _run("Innsamling — SSB", collect_ssb)
    _run("Innsamling — Eiendom Norge", collect_eiendom_norge)
    if include_matrikkelen:
        _run("Innsamling — Matrikkelen (treg)", collect_matrikkelen)


def run_standardize(include_matrikkelen: bool = False) -> None:
    # Kartverket må kjøres først — postnummer-mappingen brukes av de andre.
    # Geofeatures avhenger av geometri-Parquet og må derfor komme etter Kartverket.
    _run("Standardisering — Kartverket", std_kartverket, critical=True)
    _run("Standardisering — Geofeatures", std_geofeatures)
    _run("Standardisering — SSB", std_ssb)
    _run("Standardisering — Eiendom Norge", std_eiendom_norge)
    if include_matrikkelen:
        _run("Standardisering — Matrikkelen", std_matrikkelen)


def run_merge() -> None:
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
    args = parser.parse_args()

    run_all = not any([args.collect, args.standardize, args.merge])
    t0 = time.time()

    if run_all or args.collect:
        run_collect(include_matrikkelen=args.include_matrikkelen)
    if run_all or args.standardize:
        run_standardize(include_matrikkelen=args.include_matrikkelen)
    if run_all or args.merge:
        run_merge()

    print(f"\n{'='*60}\nPipeline ferdig på {time.time() - t0:.1f}s\n{'='*60}\n")


if __name__ == "__main__":
    main()
