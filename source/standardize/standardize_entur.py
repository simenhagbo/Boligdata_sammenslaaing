"""
Beregner avstand fra hvert postnummer til nærmeste togstasjon.

Bruker togstasjons-koordinater fra Entur og postnummer-sentroider fra
geofeatures. Avstanden beregnes i EPSG:25833 (UTM 33N, meter) for å gi
korrekt avstand i kilometer over hele Norge.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "entur"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)


def compute_togstasjon_features() -> pd.DataFrame:
    """For hver postnummer-sentroide, finn avstand til nærmeste togstasjon.

    Vi reprojiserer både sentroider og stasjons-punkter til EPSG:25833 før
    avstandsberegning — i grader (4326) er én lengdegrad mye kortere ved
    polene enn ved ekvator, så avstanden i km blir ellers feil.
    """
    src = RAW_DIR / "togstasjoner.json"
    geom_src = STD_DIR / "postnummer_geometri.parquet"
    out = STD_DIR / "postnummer_entur.parquet"

    if not src.exists():
        raise FileNotFoundError(f"{src} finnes ikke — kjør collect_entur.py først")
    if not geom_src.exists():
        raise FileNotFoundError(
            "postnummer_geometri.parquet mangler — kjør standardize_kartverket.py først"
        )

    with open(src, encoding="utf-8") as f:
        stops = json.load(f)
    print(f"  Lastet {len(stops)} togstasjoner fra Entur")

    # Konverter stasjoner til GeoSeries i UTM 33N for meter-basert avstand
    stops_4326 = gpd.GeoSeries(
        [Point(s["longitude"], s["latitude"]) for s in stops], crs="EPSG:4326"
    )
    stops_m = stops_4326.to_crs("EPSG:25833")

    # Last postnummer-sentroider. Reprojiser polygonet til 25833 og ta
    # centroid — samme mønster som i standardize_geofeatures.
    gdf = gpd.read_parquet(geom_src).to_crs("EPSG:25833")
    sentroider = gdf.geometry.centroid

    # For hver sentroide, finn min-avstand til alle stops. Vektorisert via
    # numpy: lag (n_postnummer × n_stops) avstandsmatrise og ta min per rad.
    sentroide_xy = np.column_stack([sentroider.x.values, sentroider.y.values])
    stops_xy = np.column_stack([stops_m.x.values, stops_m.y.values])

    # Avstandsmatrise i bolker for å unngå (3378 × ~700) × 8 byte = ~20 MB
    # — det er greit, men chunker likevel for å være snill med minne på
    # større datasett.
    chunk_size = 500
    min_avstand_m = np.empty(len(sentroide_xy))
    for i in range(0, len(sentroide_xy), chunk_size):
        chunk = sentroide_xy[i:i + chunk_size]
        # Bredkastet differanse: (chunk_size × 1 × 2) - (1 × n_stops × 2)
        diff = chunk[:, None, :] - stops_xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        min_avstand_m[i:i + chunk_size] = dist.min(axis=1)

    out_df = pd.DataFrame({
        "postnummer": gdf["postnummer"].values,
        "avstand_togstasjon_km": min_avstand_m / 1000,
    })
    out_df.to_parquet(out, index=False)
    print(f"  postnummer_entur.parquet: {len(out_df)} postnummer")
    print(f"    avstand_togstasjon_km: snitt={out_df['avstand_togstasjon_km'].mean():.1f} km, "
          f"median={out_df['avstand_togstasjon_km'].median():.1f} km, "
          f"max={out_df['avstand_togstasjon_km'].max():.1f} km")
    return out_df


def main() -> None:
    print("=== Standardisering: Entur ===")
    compute_togstasjon_features()
    print("=== Entur standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
