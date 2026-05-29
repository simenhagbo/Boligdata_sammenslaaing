"""
Bygger mapping fra historiske kommunekoder til 2024-koder, og re-aggregerer
SSB-data slik at eldre år (med datidens koder) matcher dagens kommuner.

Mapping bygges ved å kjede kodeendringene fra Klass: hvis 0716 ble 3801 i
2020 og 3801 ble 3905 i 2024, så mapper 0716 → 3905. De fleste endringene er
sammenslåinger (mange gamle → én ny), som er rett fram. Tre splitt-tilfeller
(én gammel → flere nye) løses ved å velge første etterfølger, med logging.

Re-aggregeringen samler alle historiske koder som peker til samme 2024-kode
innen hvert år, og slår dem sammen med riktig regel per kolonne:
  - Tellinger summeres (befolkning, antall boliger, omsetninger, ...)
  - Priser vektes med antall omsetninger
  - Andeler/medianer vektes med antall boliger eller befolkning
  - Modal boligtype tas fra kommunen med flest boliger

Output:
  processed_data/standardized/kommune_historikk_mapping.parquet
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "kommune_historikk"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)

MAPPING_PATH = STD_DIR / "kommune_historikk_mapping.parquet"


# ----- Bygg mapping fra kodeendringer -----

def _build_redirect(code_changes: list[dict]) -> dict[str, str]:
    """Lag ett-stegs omdirigering gammel→ny fra ekte kodeendringer.

    Rene renavn (oldCode == newCode) hoppes over. Splitt-kilder (én gammel
    kode til flere nye) håndteres etter type:
      - De-merger: hvis én etterfølger beholder kommunenavnet (f.eks. Ålesund
        som beholdt navnet da Haram skilte seg ut), mapper vi dit — den
        videreførende enheten arver historikken.
      - Ekte oppløsning: ingen etterfølger beholder navnet (f.eks. Stokke delt
        mellom Tønsberg og Sandefjord). Da utelater vi koden helt, slik at
        pre-splitt-dataene faller ut i stedet for å oppblåse én tilfeldig
        etterfølger.
    """
    real = [c for c in code_changes if c["oldCode"] != c["newCode"]]

    # Samle etterfølgere per gammel kode, med navn for de-merger-deteksjon
    successors: dict[str, list[dict]] = {}
    for c in real:
        successors.setdefault(c["oldCode"].strip(), []).append(c)

    redirect: dict[str, str] = {}
    ekte_split: list[str] = []
    for old, changes in successors.items():
        ny_koder = {c["newCode"].strip() for c in changes}
        if len(ny_koder) == 1:
            # Entydig endring (merge eller renummerering)
            redirect[old] = next(iter(ny_koder))
            continue
        # Splitt: let etter etterfølger som beholder kommunenavnet
        old_navn = changes[0]["oldName"].strip().lower()
        navne_match = [c["newCode"].strip() for c in changes
                       if c["newName"].strip().lower() == old_navn]
        if len(navne_match) == 1:
            redirect[old] = navne_match[0]  # de-merger: videreførende enhet
        else:
            ekte_split.append(old)  # ekte oppløsning — utelates

    if ekte_split:
        print(f"  {len(ekte_split)} ekte splitt-kilde(r) utelatt (oppløst uten "
              f"videreførende enhet): {sorted(ekte_split)}")
    return redirect


def _resolve_to_2024(code: str, redirect: dict[str, str]) -> str:
    """Følg omdirigerings-kjeden til endelig 2024-kode (med sykkel-vakt)."""
    seen = set()
    current = code
    # Følg kjeden til koden ikke lenger endres, eller vi oppdager en sykkel
    while current in redirect and current not in seen:
        seen.add(current)
        current = redirect[current]
    return current


def build_mapping() -> pd.DataFrame:
    """Bygg mapping-tabell (historisk_kommune_nr → kommune_nr_2024)."""
    src = RAW_DIR / "kodeendringer.json"
    if not src.exists():
        raise FileNotFoundError(
            "kodeendringer.json mangler — kjør collect_kommune_historikk.py"
        )
    with open(src, encoding="utf-8") as f:
        code_changes = json.load(f).get("codeChanges", [])

    # Bygg omdirigering og løs hver historisk kode til sin 2024-ekvivalent
    redirect = _build_redirect(code_changes)
    historiske = sorted(redirect.keys())
    rows = [
        {"historisk_kommune_nr": h, "kommune_nr_2024": _resolve_to_2024(h, redirect)}
        for h in historiske
    ]
    mapping = pd.DataFrame(rows)
    mapping.to_parquet(MAPPING_PATH, index=False)
    print(f"  kommune_historikk_mapping.parquet: {len(mapping)} historiske koder mappet")
    return mapping


def load_mapping() -> dict[str, str]:
    """Les mapping-tabellen som dict historisk→2024. Tom dict hvis fil mangler."""
    if not MAPPING_PATH.exists():
        return {}
    m = pd.read_parquet(MAPPING_PATH)
    return dict(zip(m["historisk_kommune_nr"], m["kommune_nr_2024"]))


# ----- Re-aggreger SSB-tabell til 2024-koder -----

# Kolonner som summeres ved sammenslåing (tellinger / absolutte mengder)
_SUM_COLS = [
    "antall_boliger", "befolkning", "antall_omsetninger_kommune",
    "antall_sysselsatte", "netto_innflytting_kommune",
    "antall_husholdninger_kommune", "fullforte_boliger_kommune",
    "igangsatte_boliger_kommune",
]

# Kolonner som vektes med antall omsetninger (priser)
_WMEAN_OMSETNING = [
    "pris_kvm_alle_kommune", "pris_kvm_enebolig_kommune",
    "pris_kvm_smaahus_kommune", "pris_kvm_blokk_kommune",
]

# Kolonner som vektes med antall boliger (bolig-fordelinger og -medianer)
_WMEAN_BOLIGER = [
    "andel_bygntype_01_kommune", "andel_bygntype_02_kommune",
    "andel_bygntype_03_kommune", "andel_bygntype_04_kommune",
    "andel_bygntype_05_kommune",
    "andel_byggeaar_for1946_kommune", "andel_byggeaar_1946_1970_kommune",
    "andel_byggeaar_1971_1990_kommune", "andel_byggeaar_1991_2010_kommune",
    "andel_byggeaar_etter2010_kommune", "median_byggeaar_kommune",
    "andel_areal_under60_kommune", "andel_areal_60_99_kommune",
    "andel_areal_100_159_kommune", "andel_areal_160_249_kommune",
    "andel_areal_over250_kommune", "median_bruksareal_kommune",
]

# Kolonner som vektes med befolkning (rater og inntekt)
_WMEAN_BEFOLKNING = [
    "inntekt_etter_skatt",
    "andel_utdanning_grunnskole_kommune", "andel_utdanning_videregaaende_kommune",
    "andel_utdanning_uh_kort_kommune", "andel_utdanning_uh_lang_kommune",
    "andel_utdanning_fagskole_kommune", "andel_utdanning_uoppgitt_kommune",
    "andel_arbeidsledige_kommune", "eiendomsskatt_sats_kommune",
]

# Kolonner som vektes med antall husholdninger
_WMEAN_HUSHOLDNING = ["andel_enslige_kommune"]


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Vektet snitt som ignorerer rader der verdi eller vekt mangler."""
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _aggregate_group(g: pd.DataFrame) -> pd.Series:
    """Slå sammen flere historiske kommuner (samme 2024-kode, samme år)."""
    out: dict[str, object] = {}

    # Tellinger summeres (NaN teller som 0 via min_count=1: helt-tomt → NaN)
    for col in _SUM_COLS:
        if col in g.columns:
            out[col] = g[col].sum(min_count=1)

    # Vektede snitt med ulike vektkolonner
    for col in _WMEAN_OMSETNING:
        if col in g.columns:
            out[col] = _weighted_mean(g[col], g.get("antall_omsetninger_kommune", pd.Series(1, index=g.index)))
    for col in _WMEAN_BOLIGER:
        if col in g.columns:
            out[col] = _weighted_mean(g[col], g.get("antall_boliger", pd.Series(1, index=g.index)))
    for col in _WMEAN_BEFOLKNING:
        if col in g.columns:
            out[col] = _weighted_mean(g[col], g.get("befolkning", pd.Series(1, index=g.index)))
    for col in _WMEAN_HUSHOLDNING:
        if col in g.columns:
            out[col] = _weighted_mean(g[col], g.get("antall_husholdninger_kommune", pd.Series(1, index=g.index)))

    # Modal boligtype: ta verdien fra kommunen med flest boliger
    if "modal_boligtype_kommune" in g.columns:
        if "antall_boliger" in g.columns and g["antall_boliger"].notna().any():
            idx = g["antall_boliger"].idxmax()
        else:
            idx = g.index[0]
        out["modal_boligtype_kommune"] = g.loc[idx, "modal_boligtype_kommune"]

    return pd.Series(out)


def reaggregate_to_2024(ssb: pd.DataFrame) -> pd.DataFrame:
    """Map historiske kommunekoder til 2024 og slå sammen per (2024-kode, år).

    Returnerer SSB-tabellen uendret hvis mapping mangler. Ellers re-aggregeres
    alle rader hvis kommune_nr er en historisk kode, slik at eldre år havner
    under dagens kommune og får dekning.
    """
    mapping = load_mapping()
    if not mapping:
        print("  Ingen historikk-mapping funnet — beholder 2024-koder som de er")
        return ssb

    # Map hver kommune_nr til 2024 (ukjente/allerede-2024-koder beholdes)
    work = ssb.copy()
    work["kommune_nr"] = work["kommune_nr"].map(lambda c: mapping.get(c, c))

    # Skille raske enkelt-rad-grupper fra de som faktisk må slås sammen.
    # De fleste (kommune, år) har bare én kilde-rad og trenger ingen aggregering.
    grp_sizes = work.groupby(["kommune_nr", "aar"])["aar"].transform("size")
    enkelt = work[grp_sizes == 1]
    flere = work.drop(enkelt.index)

    n_merged = flere.groupby(["kommune_nr", "aar"]).ngroups if len(flere) else 0
    print(f"  Re-aggregerer {len(flere):,} historiske rader til {n_merged:,} "
          f"(kommune, år)-grupper")

    if len(flere) == 0:
        return work.sort_values(["kommune_nr", "aar"]).reset_index(drop=True)

    # Slå sammen fler-rad-gruppene med kolonne-spesifikke regler
    merged = (
        flere.groupby(["kommune_nr", "aar"], group_keys=True)
        .apply(_aggregate_group, include_groups=False)
        .reset_index()
    )

    # Sett sammen igjen og sorter
    result = pd.concat([enkelt, merged], ignore_index=True)
    result = result.sort_values(["kommune_nr", "aar"]).reset_index(drop=True)
    return result


def main() -> None:
    # Bygg og lagre mapping-tabellen. Re-aggregeringen kalles fra standardize_ssb.
    print("=== Standardisering: Kommune-historikk ===")
    src = RAW_DIR / "kodeendringer.json"
    if not src.exists():
        print("  HOPPET OVER: ingen kodeendringer (kjør collect_kommune_historikk.py)")
        print("=== Kommune-historikk standardisering hoppet over ===\n")
        return
    build_mapping()
    print("=== Kommune-historikk standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
