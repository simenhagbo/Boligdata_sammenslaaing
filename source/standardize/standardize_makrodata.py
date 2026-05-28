"""
Parser makrodata-rådata til én Parquet-fil på årsnivå.

Output: makrodata.parquet med kolonner:
  aar                    int
  kpi_indeks             SSB 03013 KPI årssnitt (2015=100)
  kpi_endring_pct        Årlig endring i KPI (inflasjon)
  styringsrente          Norges Banks årssnitt
  boliglaansrente        SSB 10748 årssnitt (NaN før 2014)

Denne tabellen joines på `aar` i merge-fasen — den er nasjonal og gjelder
for alle kommuner.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "makrodata"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)

# Bruk parse_jsonstat fra standardize_ssb for å unngå duplisering
sys.path.insert(0, str(Path(__file__).parent))
from standardize_ssb import parse_jsonstat  # noqa: E402


def parse_norgesbank_sdmx(path: Path) -> pd.DataFrame:
    """Konverter Norges Banks SDMX-JSON til (aar, styringsrente)-DataFrame.

    SDMX-JSON-strukturen er: én serie (siden vi kun spør om én), med en
    observasjons-ordbok der nøkkelen er indeks i tids-dimensjonen. Tids-
    verdiene står i `structure.dimensions.observation[0].values` i samme
    rekkefølge — så indeks 0 = første år, 1 = andre osv.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    series_dict = data["data"]["dataSets"][0]["series"]
    # Vi spurte om kun én serie, men SDMX returnerer den som en dict med én nøkkel
    series = next(iter(series_dict.values()))
    observations = series["observations"]

    # Hent tids-dimensjons-verdiene (indeks → år-streng)
    time_dim = data["data"]["structure"]["dimensions"]["observation"][0]
    time_values = [v["id"] for v in time_dim["values"]]

    rows = []
    for obs_idx, obs_val in observations.items():
        aar = int(time_values[int(obs_idx)])
        # Verdier er en liste; første element er måleverdien som streng
        verdi = obs_val[0]
        if verdi is None:
            continue
        rows.append({"aar": aar, "styringsrente": float(verdi)})
    return pd.DataFrame(rows).sort_values("aar").reset_index(drop=True)


def monthly_to_annual(df: pd.DataFrame, tid_col: str, value_col: str,
                      new_name: str) -> pd.DataFrame:
    """Aggreger månedlige SSB-rader til årssnitt.

    Tid-verdiene har format `2024M03`. Vi tar første fire tegn som år og
    snitter alle observasjoner innen samme år. Manglende måneder forplanter
    seg ikke — vi krever bare minst én observasjon for å lage et årssnitt.
    """
    df = df.copy()
    df["aar"] = df[tid_col].astype(str).str[:4].astype(int)
    annual = df.groupby("aar")[value_col].mean().rename(new_name).reset_index()
    return annual


def standardize_kpi() -> pd.DataFrame:
    """Årssnitt av KPI fra SSB 03013 (månedlig, totalindeks 2015=100)."""
    src = RAW_DIR / "kpi_03013.json"
    if not src.exists():
        return pd.DataFrame()
    df = parse_jsonstat(src)
    return monthly_to_annual(df, tid_col="Tid", value_col="value", new_name="kpi_indeks")


def standardize_boliglaansrente() -> pd.DataFrame:
    """Årssnitt av boliglånsrente fra SSB 10748 (månedlig fra 2013M12)."""
    src = RAW_DIR / "boliglaansrente_10748.json"
    if not src.exists():
        return pd.DataFrame()
    df = parse_jsonstat(src)
    return monthly_to_annual(df, tid_col="Tid", value_col="value",
                             new_name="boliglaansrente")


def standardize_styringsrente() -> pd.DataFrame:
    """Norges Banks styringsrente — allerede årssnitt fra API-et."""
    src = RAW_DIR / "styringsrente_norgesbank.json"
    if not src.exists():
        return pd.DataFrame()
    return parse_norgesbank_sdmx(src)


def main() -> None:
    print("=== Standardisering: Makrodata ===")

    parts = {
        "kpi": standardize_kpi(),
        "styringsrente": standardize_styringsrente(),
        "boliglaansrente": standardize_boliglaansrente(),
    }
    for name, p in parts.items():
        print(f"  {name}: {len(p):,} år")

    # Start med KPI (lengst tidsserie) og join inn de andre. outer-join slik at
    # alle år bevares — pre-2014 får NaN på boliglånsrente.
    if parts["kpi"].empty:
        raise RuntimeError("KPI mangler — kan ikke bygge makrodata.parquet")
    df = parts["kpi"].copy()

    # Beregn årlig inflasjon (KPI-endring) ved enkel pct_change. Første år får NaN
    # siden vi ikke har forrige år å sammenligne med.
    df["kpi_endring_pct"] = df["kpi_indeks"].pct_change() * 100

    for name in ("styringsrente", "boliglaansrente"):
        if not parts[name].empty:
            df = df.merge(parts[name], on="aar", how="outer")

    # Filtrer til pipelinens ML-tidsserie 2002-... slik at vi ikke drasser
    # med oss pre-2002-rader fra KPI. Slutt-året begrenses ikke — fremtidige
    # år kommer automatisk inn når SSB publiserer.
    df = df[df["aar"] >= 2002].sort_values("aar").reset_index(drop=True)

    out = STD_DIR / "makrodata.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  makrodata.parquet: {len(df)} år × {len(df.columns)} kolonner")
    print(f"  Dekning:")
    for col in df.columns:
        if col == "aar":
            continue
        dekning = df[col].notna().mean()
        print(f"    {col}: {dekning:.0%}")
    print("=== Makrodata standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
