"""
Beregner geografiske features fra postnummer-polygonene.

Areal og avstander beregnes i EPSG:25833 (UTM 33N, meter), siden geometri
i EPSG:4326 er grader og ikke direkte sammenlignbart for avstand/areal.
Sentroide regnes også i 25833 og reprojiseres tilbake til 4326 for lat/lon.

Avstand til storby brukes som proxy for sentralitet — SSBs offisielle
sentralitetsindeks finnes ikke via API og krever manuell Excel-nedlasting,
men avstand til nærmeste storby + befolkningstetthet (beregnes i merge)
gir mye av samme signal.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"

# (navn, lat, lon). Seks storbyer som dekker hele Norge geografisk.
STORBYER = [
    ("Oslo", 59.9139, 10.7522),
    ("Bergen", 60.3913, 5.3221),
    ("Trondheim", 63.4305, 10.3951),
    ("Stavanger", 58.9700, 5.7331),
    ("Tromsø", 69.6492, 18.9553),
    ("Kristiansand", 58.1599, 8.0182),
]


def compute_geofeatures() -> gpd.GeoDataFrame:
    """Beregn areal, sentroide-koordinater og avstand til nærmeste storby."""
    src = STD_DIR / "postnummer_geometri.parquet"
    out = STD_DIR / "postnummer_geofeatures.parquet"

    if not src.exists():
        raise FileNotFoundError(
            "Kjør standardize_kartverket.py før standardize_geofeatures.py"
        )

    gdf = gpd.read_parquet(src)

    # Reprojiser til UTM 33N for meter-baserte beregninger
    gdf_m = gdf.to_crs("EPSG:25833")

    # Areal i km²
    gdf["areal_km2"] = gdf_m.geometry.area / 1_000_000

    # Sentroide i 25833, så tilbake til 4326 for lat/lon
    sentroider_m = gdf_m.geometry.centroid
    sentroider_grader = gpd.GeoSeries(sentroider_m, crs="EPSG:25833").to_crs("EPSG:4326")
    gdf["sentroide_lon"] = sentroider_grader.x
    gdf["sentroide_lat"] = sentroider_grader.y

    # Konverter storby-koordinater til 25833 for avstandsberegning
    storby_punkter_4326 = gpd.GeoSeries(
        [Point(lon, lat) for _, lat, lon in STORBYER], crs="EPSG:4326"
    )
    storby_punkter_m = storby_punkter_4326.to_crs("EPSG:25833")
    storby_navn = [navn for navn, _, _ in STORBYER]

    # Avstand fra hver sentroide til hver storby — produserer matrise (n_pnr × 6)
    avstander = pd.DataFrame(
        {
            navn: sentroider_m.distance(punkt) / 1000  # meter → km
            for navn, punkt in zip(storby_navn, storby_punkter_m)
        }
    )

    gdf["avstand_oslo_km"] = avstander["Oslo"].values
    gdf["avstand_naermeste_storby_km"] = avstander.min(axis=1).values
    gdf["naermeste_storby"] = avstander.idxmin(axis=1).values

    cols = [
        "postnummer", "areal_km2",
        "sentroide_lat", "sentroide_lon",
        "avstand_oslo_km", "avstand_naermeste_storby_km", "naermeste_storby",
    ]
    out_df = pd.DataFrame(gdf[cols])
    out_df.to_parquet(out, index=False)
    print(f"  postnummer_geofeatures.parquet: {len(out_df)} postnummer")
    print(f"    areal_km2 total: {out_df['areal_km2'].sum():,.0f} km²")
    print(f"    sentroide_lat range: {out_df['sentroide_lat'].min():.2f}-{out_df['sentroide_lat'].max():.2f}")
    return out_df


def main() -> None:
    print("=== Standardisering: Geofeatures ===")
    compute_geofeatures()
    print("=== Geofeatures standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
