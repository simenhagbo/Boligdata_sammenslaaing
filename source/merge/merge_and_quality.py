"""
Slår alle standardiserte kilder sammen til ett tidsserie-Parquet.

Hver rad er (postnummer, år). Statiske attributter (geometri, areal, avstand
til storby) repeteres per år. SSB-feltene varierer per år. NaN der en kilde
mangler data for et gitt år — preprosessering kan filtrere eller imputere.

Rekkefølge:
  1. Geometri + Bring-mapping → én rad per postnummer
  2. Geofeatures → joinet på postnummer
  3. Ekspander til postnummer × år (2002-2024)
  4. SSB → joinet på (postnummer, aar)
  5. Beregn befolkningstetthet på kommune-år nivå
  6. Matrikkelen (opt-in, kun statisk per kommune)

Output:
  boligdata_final.parquet
  merge_log.json
"""

import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd

STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
FINAL_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "final"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

AAR_RANGE = list(range(2002, 2025))

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
    }
    loaded = {}
    for key, fname in files.items():
        path = STD_DIR / fname
        if not path.exists():
            _log("MANGLER_FIL", {"fil": fname, "handling": "hopper over"})
            loaded[key] = None
            continue
        # Geometri må leses med geopandas for å bevare polygon-kolonnen
        if key == "geometri":
            loaded[key] = gpd.read_parquet(path)
        else:
            loaded[key] = pd.read_parquet(path)
        _log("LASTET", {"kilde": key, "rader": len(loaded[key])})
    return loaded


def merge_all(data: dict) -> gpd.GeoDataFrame:
    """Bygger tidsserien postnummer × år og fyller inn alle kilder."""
    backbone = data["geometri"]
    if backbone is None:
        raise RuntimeError("Kartverket geometri mangler — kan ikke bygge datasett")

    _log("BACKBONE_BASE", {"postnummer": len(backbone)})

    if data["mapping"] is not None:
        backbone = backbone.merge(
            data["mapping"][["postnummer", "kommune_nr", "poststedsnavn", "kommunenavn"]],
            on="postnummer", how="left",
        )
        _log("MERGE", {"kilde": "kartverket_mapping", "rader": len(backbone)})

    if data["geofeatures"] is not None:
        before = backbone["postnummer"].isin(data["geofeatures"]["postnummer"]).mean()
        backbone = backbone.merge(data["geofeatures"], on="postnummer", how="left")
        _log("MERGE", {"kilde": "geofeatures", "dekning": f"{before:.1%}"})

    if data["matrikkelen"] is not None and "kommune_nr" in backbone.columns:
        matr = data["matrikkelen"]
        before = backbone["kommune_nr"].isin(matr["kommune_nr"]).mean()
        backbone = backbone.merge(matr, on="kommune_nr", how="left")
        _log("MERGE", {"kilde": "matrikkelen", "dekning": f"{before:.1%}"})

    # Ekspander til tidsserie: kryss-produkt med år
    aar = pd.DataFrame({"aar": AAR_RANGE})
    backbone = backbone.merge(aar, how="cross")
    _log("EKSPANDERT_TIDSSERIE", {
        "postnummer": int(backbone["postnummer"].nunique()),
        "aar_range": [AAR_RANGE[0], AAR_RANGE[-1]],
        "rader_totalt": len(backbone),
    })

    if data["ssb"] is not None:
        ssb = data["ssb"]
        # SSB-tabellen har postnummer + aar som nøkkel; vi vil ikke ha dobbelte
        # kolonner for kommune_nr som allerede finnes i backbone
        ssb = ssb.drop(columns=["kommune_nr"], errors="ignore")
        before = backbone.merge(
            ssb[["postnummer", "aar"]].drop_duplicates(),
            on=["postnummer", "aar"], how="left", indicator=True,
        )["_merge"].eq("both").mean()
        backbone = backbone.merge(ssb, on=["postnummer", "aar"], how="left")
        _log("MERGE", {"kilde": "ssb", "dekning": f"{before:.1%}"})

    # Befolkningstetthet på kommune-år nivå. Areal er statisk per postnummer,
    # men vi vil ha kommunenivå-tetthet, så vi summerer postnummer-arealene
    # innen hver kommune først.
    if {"befolkning", "areal_km2", "kommune_nr"}.issubset(backbone.columns):
        # Areal er statisk, så vi kan gruppere på kommune_nr alene (uten år)
        kommune_areal = backbone.drop_duplicates("postnummer").groupby("kommune_nr")["areal_km2"].sum()
        kommune_areal = kommune_areal.rename("kommune_areal_km2").reset_index()
        backbone = backbone.merge(kommune_areal, on="kommune_nr", how="left")
        areal = backbone["kommune_areal_km2"].where(backbone["kommune_areal_km2"] > 0)
        backbone["befolkningstetthet"] = backbone["befolkning"] / areal
        backbone = backbone.drop(columns=["kommune_areal_km2"])
        _log("BEREGNET", {
            "kolonne": "befolkningstetthet",
            "dekning": f"{backbone['befolkningstetthet'].notna().mean():.1%}",
        })

    return backbone


def quality_check(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Logg dekning per kolonne, flagg dårlige rader, finn IQR-outliers."""
    feature_cols = [
        c for c in df.columns
        if c not in (
            "postnummer", "geometry", "poststedsnavn", "kommunenavn",
            "kommune_nr", "naermeste_storby", "aar",
        )
    ]

    coverage = {c: float(df[c].notna().mean()) for c in feature_cols}
    _log("KOLONNE_DEKNING", coverage)

    # Flagg rader der over halvparten av kolonnene mangler verdi.
    # NB: i tidsserie-formatet er disse typisk fra eldre år der SSB-tabellene
    # ikke har data ennå, eller fra kommunesammenslåinger.
    missing_rate = df[feature_cols].isna().mean(axis=1)
    df["data_kvalitet_flagg"] = (missing_rate > 0.5).astype(int)
    n_flagged = int(df["data_kvalitet_flagg"].sum())
    _log("KVALITETSFLAGG", {
        "rader_med_over_50pct_manglende": n_flagged,
        "pct_av_total": f"{n_flagged / len(df):.1%}",
    })

    # IQR med 3x — boligpriser har naturlig stor spredning
    for col in ["pris_kvm_alle_kommune", "inntekt_etter_skatt", "befolkningstetthet"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        _log("OUTLIER_IQR", {
            "kolonne": col,
            "nedre_grense": round(float(lower), 1),
            "ovre_grense": round(float(upper), 1),
            "antall_outliers": int(outliers),
        })

    return df


def main() -> None:
    print("=== Sammenslåing + kvalitetskontroll ===")

    data = load_standardized()
    merged = merge_all(data)
    final = quality_check(merged)

    # Sjekk for duplikater på (postnummer, aar)
    n_before = len(final)
    final = final.drop_duplicates(["postnummer", "aar"])
    if len(final) < n_before:
        _log("DUPLIKATER_FJERNET", {"antall": n_before - len(final)})

    # Sorter for å gjøre filen lett å inspisere
    final = final.sort_values(["postnummer", "aar"]).reset_index(drop=True)

    out_parquet = FINAL_DIR / "boligdata_final.parquet"
    final.to_parquet(out_parquet, index=False)
    _log("OUTPUT", {
        "fil": str(out_parquet),
        "rader": len(final),
        "kolonner": len(final.columns),
        "unike_postnummer": int(final["postnummer"].nunique()),
        "unike_aar": int(final["aar"].nunique()),
    })

    out_log = FINAL_DIR / "merge_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(LOG, f, ensure_ascii=False, indent=2)

    print(f"\nFerdig: {out_parquet.name}")
    print(f"  {len(final):,} rader (postnummer × år)")
    print(f"  {len(final.columns)} kolonner")
    print(f"  {final['postnummer'].nunique()} postnummer × {final['aar'].nunique()} år")
    print(f"Logg: {out_log.name}")
    print("=== Sammenslåing ferdig ===\n")


if __name__ == "__main__":
    main()
