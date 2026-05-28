"""
Slår sammen alle standardiserte kilder til ett Parquet-datasett.

Bruker postnummer fra Kartverket som backbone (alle norske postnummer beholdes),
og left-joiner resten inn. Manglende verdier blir NaN — det er bevisst, slik
at ML-modellen senere kan velge selv hvordan den vil håndtere det.

Join-rekkefølge:
  1. Kartverket geometri + Bring-mapping (backbone)
  2. Geofeatures (areal, sentroide, avstand til storby)
  3. SSB (demografi, priser, byggeår, bruksareal)
  4. Eiendom Norge prisstatistikk
  5. Matrikkelen (kun ved --include-matrikkelen)

Etter SSB-join beregnes `befolkningstetthet` = `befolkning / areal_km2`.

Output:
  boligdata_final.parquet  — selve datasettet
  merge_log.json           — dekningstall, kvalitetsflagg, IQR-outliers
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
FINAL_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

LOG: list[dict] = []


def _log(event: str, details: dict) -> None:
    entry = {"timestamp": datetime.now().isoformat(), "event": event, **details}
    LOG.append(entry)
    print(f"  [{event}] {details}")


def load_standardized() -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Last alle standardiserte Parquet-filer. Manglende filer settes til None."""
    files = {
        "geometri": "postnummer_geometri.parquet",
        "mapping": "postnummer_kommune_mapping.parquet",
        "geofeatures": "postnummer_geofeatures.parquet",
        "matrikkelen": "matrikkelen_aggregert.parquet",
        "ssb": "ssb_bolig_demografi.parquet",
        "eiendom_norge": "eiendom_norge_priser.parquet",
    }
    loaded = {}
    for key, fname in files.items():
        path = STD_DIR / fname
        if not path.exists():
            _log("MANGLER_FIL", {"fil": fname, "handling": "hopper over"})
            loaded[key] = None
            continue
        # Geometri-filen må leses med geopandas for å bevare polygon-kolonnen
        if key == "geometri":
            loaded[key] = gpd.read_parquet(path)
        else:
            loaded[key] = pd.read_parquet(path)
        _log("LASTET", {"kilde": key, "rader": len(loaded[key])})
    return loaded


def merge_all(data: dict) -> gpd.GeoDataFrame:
    """Left-join alle kilder inn i backbone'en. Logger dekning før hver join."""
    backbone = data["geometri"]
    if backbone is None:
        raise RuntimeError("Kartverket geometri mangler — kan ikke bygge datasett")

    _log("BACKBONE", {"postnummer_totalt": len(backbone)})

    if data["mapping"] is not None:
        backbone = backbone.merge(
            data["mapping"][["postnummer", "kommune_nr", "poststedsnavn", "kommunenavn"]],
            on="postnummer",
            how="left",
        )
        _log("MERGE", {"kilde": "kartverket_mapping", "rader_etter": len(backbone)})

    # Geofeatures joines på postnummer — beregnet fra polygonene, 100% dekning forventet
    if data["geofeatures"] is not None:
        before = backbone["postnummer"].isin(data["geofeatures"]["postnummer"]).mean()
        backbone = backbone.merge(data["geofeatures"], on="postnummer", how="left")
        _log("MERGE", {"kilde": "geofeatures", "dekning": f"{before:.1%}"})

    # Matrikkelen joines på kommune_nr — alle postnummer i samme kommune får
    # samme aggregerte verdier
    if data["matrikkelen"] is not None and "kommune_nr" in backbone.columns:
        matr = data["matrikkelen"]
        before = backbone["kommune_nr"].isin(matr["kommune_nr"]).mean()
        backbone = backbone.merge(matr, on="kommune_nr", how="left")
        _log("MERGE", {"kilde": "matrikkelen (per kommune)", "dekning": f"{before:.1%}"})

    if data["ssb"] is not None:
        # Drop kommune_nr fra SSB-tabellen — backbone har den allerede
        ssb = data["ssb"].drop(columns=["kommune_nr"], errors="ignore")
        before = backbone["postnummer"].isin(ssb["postnummer"]).mean()
        backbone = backbone.merge(ssb, on="postnummer", how="left")
        _log("MERGE", {"kilde": "ssb", "dekning": f"{before:.1%}"})

    # Beregn befolkningstetthet på kommunenivå. SSBs befolkning er per kommune,
    # så vi må aggregere postnummer-arealet til kommunearealet før vi deler.
    # (Å bruke postnummer-areal direkte ville gitt kjempetall siden hele kommunens
    # befolkning ville blitt delt på ett enkelt postnummer.)
    if {"befolkning", "areal_km2", "kommune_nr"}.issubset(backbone.columns):
        kommune_areal = backbone.groupby("kommune_nr")["areal_km2"].sum().rename("kommune_areal_km2")
        backbone = backbone.merge(kommune_areal, on="kommune_nr", how="left")
        areal = backbone["kommune_areal_km2"].where(backbone["kommune_areal_km2"] > 0)
        backbone["befolkningstetthet"] = backbone["befolkning"] / areal
        backbone = backbone.drop(columns=["kommune_areal_km2"])
        _log("BEREGNET", {"kolonne": "befolkningstetthet", "dekning": f"{backbone['befolkningstetthet'].notna().mean():.1%}"})

    if data["eiendom_norge"] is not None:
        en = data["eiendom_norge"].drop(columns=["kommune_nr"], errors="ignore")
        before = backbone["postnummer"].isin(en["postnummer"]).mean()
        backbone = backbone.merge(en, on="postnummer", how="left")
        _log("MERGE", {"kilde": "eiendom_norge", "dekning": f"{before:.1%}"})

    return backbone


def quality_check(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Logg dekning per kolonne, flagg dårlige rader, finn IQR-outliers."""
    feature_cols = [
        c for c in df.columns
        if c not in (
            "postnummer", "geometry", "poststedsnavn", "kommunenavn", "kommune_nr",
            "naermeste_storby",  # tekst-kolonne, ikke en numerisk feature
        )
    ]

    coverage = {c: float(df[c].notna().mean()) for c in feature_cols}
    _log("KOLONNE_DEKNING", coverage)

    # Flagg postnummer der over halvparten av kolonnene mangler verdi
    missing_rate = df[feature_cols].isna().mean(axis=1)
    df["data_kvalitet_flagg"] = (missing_rate > 0.5).astype(int)
    n_flagged = int(df["data_kvalitet_flagg"].sum())
    _log("KVALITETSFLAGG", {
        "postnummer_med_>50pct_manglende": n_flagged,
        "pct_av_total": f"{n_flagged / len(df):.1%}",
    })

    # IQR med 3x (ikke standard 1.5x) — boligpriser har naturlig stor spredning
    for col in ["median_pris_m2", "pris_kvm_alle_kommune", "inntekt_etter_skatt", "befolkningstetthet"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        _log("OUTLIER_IQR", {
            "kolonne": col,
            "nedre_grense": round(lower, 1),
            "øvre_grense": round(upper, 1),
            "antall_outliers": int(outliers),
        })

    return df


def main() -> None:
    print("=== Sammenslåing + kvalitetskontroll ===")

    data = load_standardized()
    merged = merge_all(data)
    final = quality_check(merged)

    # Sikkerhetssjekk: noen joins kan i teorien gi duplikater
    n_before = len(final)
    final = final.drop_duplicates("postnummer")
    if len(final) < n_before:
        _log("DUPLIKATER_FJERNET", {"antall": n_before - len(final)})

    out_parquet = FINAL_DIR / "boligdata_final.parquet"
    final.to_parquet(out_parquet, index=False)
    _log("OUTPUT", {
        "fil": str(out_parquet),
        "rader": len(final),
        "kolonner": list(final.columns),
    })

    out_log = FINAL_DIR / "merge_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)

    print(f"\nFerdig: {out_parquet.name} ({len(final)} postnummer, {len(final.columns)} kolonner)")
    print(f"Logg: {out_log.name}")
    print("=== Sammenslåing ferdig ===\n")


if __name__ == "__main__":
    main()
