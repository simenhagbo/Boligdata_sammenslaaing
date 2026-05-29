"""
Mapper Frost klima-normaler til hvert postnummer ved nærmeste-stasjon-lookup.

For hver postnummer-sentroide finner vi den værstasjonen som er nærmest og
har klima-normal-data for både temperatur og nedbør. Resultatet er mer
presise klima-features enn dagens "kopier fra nærmeste storby"-proxy.

Output:
  postnummer_met_frost.parquet med kolonner:
    postnummer                  4-sifret
    temperatur_normal_frost     årlig snitt-temperatur (°C, 1991-2020)
    nedbor_normal_frost_mm      årssum nedbør (mm, 1991-2020)
    met_stasjon_id              Frost-stasjons-ID (f.eks. "SN18700")
    met_stasjon_avstand_km      luftlinje fra sentroide til stasjon
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "met_frost"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)


def _parse_stasjoner() -> pd.DataFrame:
    """Les stasjons-metadata. Returnerer DataFrame med (id, lat, lon)."""
    src = RAW_DIR / "stasjoner.json"
    if not src.exists():
        raise FileNotFoundError("stasjoner.json mangler — kjør collect_met_frost.py")

    with open(src, encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    for entry in payload.get("data", []):
        geom = entry.get("geometry")
        if not geom or geom.get("@type") != "Point":
            continue
        coords = geom.get("coordinates", [])
        if len(coords) < 2:
            continue
        rows.append({
            "stasjon_id": entry["id"],
            "lon": float(coords[0]),
            "lat": float(coords[1]),
        })
    return pd.DataFrame(rows)


def _parse_normaler() -> pd.DataFrame:
    """Les klima-normaler. Returnerer wide-format med én rad per stasjon
    og kolonner for hver målestørrelse vi bryr oss om.
    """
    src = RAW_DIR / "normaler.json"
    if not src.exists():
        raise FileNotFoundError("normaler.json mangler — kjør collect_met_frost.py")

    with open(src, encoding="utf-8") as f:
        payload = json.load(f)

    rows: list[dict] = []
    for entry in payload.get("data", []):
        # Forsøk å håndtere både flate-felt og nested-felt-varianter siden
        # Frost har endret responsformatet mellom v0-versjoner
        stasjon = entry.get("sourceId") or entry.get("source")
        element = entry.get("elementId") or entry.get("element")
        value = entry.get("normal") or entry.get("value")
        if value is None or not stasjon or not element:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        rows.append({
            "stasjon_id": str(stasjon).split(":")[0],  # strip ev. ":0" suffix
            "element": element,
            "value": value,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Pivot til wide-format. Vi bryr oss bare om de to elementene vi spurte etter.
    df = df.drop_duplicates(["stasjon_id", "element"])
    wide = df.pivot(index="stasjon_id", columns="element", values="value").reset_index()

    # Rename element-ID-ene til lesbare kolonner. Mappingen tolererer at Frost
    # noen ganger returnerer kortere navn.
    col_map = {}
    for c in wide.columns:
        if "air_temperature" in c:
            col_map[c] = "temperatur_normal_frost"
        elif "precipitation_amount" in c:
            col_map[c] = "nedbor_normal_frost_mm"
    wide = wide.rename(columns=col_map)
    return wide


def map_postnummer_til_stasjon() -> pd.DataFrame:
    """For hver postnummer-sentroide: finn nærmeste stasjon med klima-data."""
    stasjoner = _parse_stasjoner()
    normaler = _parse_normaler()
    if normaler.empty:
        raise RuntimeError("Ingen klima-normaler kunne parses fra Frost-rådata")

    # Behold bare stasjoner som har data for begge elementene vi vil ha
    wanted_cols = [c for c in ("temperatur_normal_frost", "nedbor_normal_frost_mm")
                   if c in normaler.columns]
    klima_stasjoner = normaler.dropna(subset=wanted_cols).merge(
        stasjoner, on="stasjon_id", how="inner",
    )
    print(f"  {len(klima_stasjoner)} stasjoner har komplett klima-normal-data")
    if klima_stasjoner.empty:
        raise RuntimeError(
            "Ingen stasjoner har data for både temperatur og nedbør — sjekk "
            "Frost-rådata"
        )

    # Last postnummer-geometri og reprojiser til UTM 33N for meter-baserte
    # avstander (samme mønster som i standardize_geofeatures)
    geom_src = STD_DIR / "postnummer_geometri.parquet"
    if not geom_src.exists():
        raise FileNotFoundError(
            "postnummer_geometri.parquet mangler — kjør standardize_kartverket.py først"
        )
    gdf = gpd.read_parquet(geom_src).to_crs("EPSG:25833")
    sentroider = gdf.geometry.centroid

    # Reprojiser stasjons-punkter til samme CRS
    stasjon_punkter_4326 = gpd.GeoSeries(
        [Point(lon, lat) for lon, lat in zip(klima_stasjoner["lon"], klima_stasjoner["lat"])],
        crs="EPSG:4326",
    )
    stasjon_xy = np.column_stack([
        stasjon_punkter_4326.to_crs("EPSG:25833").x.values,
        stasjon_punkter_4326.to_crs("EPSG:25833").y.values,
    ])
    sentroide_xy = np.column_stack([sentroider.x.values, sentroider.y.values])

    # Avstandsmatrise i bolker for å unngå å materialisere full N×M-matrise
    # på én gang. ~3400 postnummer × ~500 stasjoner er fortsatt billig, men
    # vi chunker for å holde stilen konsistent med standardize_entur.
    chunk = 500
    nearest_idx = np.empty(len(sentroide_xy), dtype=int)
    min_avstand_m = np.empty(len(sentroide_xy))
    for i in range(0, len(sentroide_xy), chunk):
        block = sentroide_xy[i:i + chunk]
        diff = block[:, None, :] - stasjon_xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        nearest_idx[i:i + chunk] = dist.argmin(axis=1)
        min_avstand_m[i:i + chunk] = dist.min(axis=1)

    out_df = pd.DataFrame({
        "postnummer": gdf["postnummer"].values,
        "temperatur_normal_frost": klima_stasjoner["temperatur_normal_frost"].iloc[nearest_idx].values,
        "nedbor_normal_frost_mm": klima_stasjoner["nedbor_normal_frost_mm"].iloc[nearest_idx].values,
        "met_stasjon_id": klima_stasjoner["stasjon_id"].iloc[nearest_idx].values,
        "met_stasjon_avstand_km": min_avstand_m / 1000,
    })

    out = STD_DIR / "postnummer_met_frost.parquet"
    out_df.to_parquet(out, index=False)
    print(f"  postnummer_met_frost.parquet: {len(out_df)} postnummer")
    print(f"    avstand til stasjon: snitt={out_df['met_stasjon_avstand_km'].mean():.1f} km, "
          f"median={out_df['met_stasjon_avstand_km'].median():.1f} km, "
          f"max={out_df['met_stasjon_avstand_km'].max():.1f} km")
    print(f"    temperatur_normal_frost: snitt={out_df['temperatur_normal_frost'].mean():.1f} °C, "
          f"range=[{out_df['temperatur_normal_frost'].min():.1f}, "
          f"{out_df['temperatur_normal_frost'].max():.1f}]")
    return out_df


def main() -> None:
    print("=== Standardisering: MET Frost ===")
    # Sjekk om rådata finnes — hvis collect-fasen hoppet over (manglende ID),
    # gjør vi det samme her stille
    if not (RAW_DIR / "stasjoner.json").exists():
        print("  HOPPET OVER: ingen Frost-rådata (FROST_CLIENT_ID antakeligvis ikke satt)")
        print("=== MET Frost standardisering hoppet over ===\n")
        return
    map_postnummer_til_stasjon()
    print("=== MET Frost standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
