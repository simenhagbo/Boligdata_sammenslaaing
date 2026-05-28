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

# Klima-normaler 1991-2020 fra MET Norges representative stasjoner (Blindern,
# Florida, Voll, Sola, Tromsø, Kjevik). Hver postnummer "arver" klima fra sin
# nærmeste storby — det er en grov proxy, men breddegrad og kystavstand er de
# sterkste klimaprediktorene i Norge, og avstand_naermeste_storby_km tar
# delvis hånd om resterende variasjon. Verdier hentet fra MET klimaservice-
# senter sin offisielle 30-årsnormal-rapport.
# (storby_navn, årlig snitt-temperatur °C, årsnedbør mm)
KLIMA_NORMALER = {
    "Oslo":         (6.4,  832),
    "Bergen":       (8.6, 2253),
    "Trondheim":    (5.6,  962),
    "Stavanger":    (8.5, 1268),
    "Tromsø":       (3.2, 1031),
    "Kristiansand": (7.6, 1300),
}


def compute_geofeatures() -> gpd.GeoDataFrame:
    """Beregn areal, sentroide-koordinater og avstand til nærmeste storby."""
    src = STD_DIR / "postnummer_geometri.parquet"
    out = STD_DIR / "postnummer_geofeatures.parquet"

    if not src.exists():
        raise FileNotFoundError(
            "Kjør standardize_kartverket.py før standardize_geofeatures.py"
        )

    gdf = gpd.read_parquet(src)

    # Reprojiser fra EPSG:4326 (grader) til EPSG:25833 (UTM 33N, meter) før
    # vi regner areal og avstand. Grader er ubrukelige for areal-beregning
    # fordi én breddegrad er kortere ved polene enn ved ekvator.
    gdf_m = gdf.to_crs("EPSG:25833")

    # Areal: shapely returnerer i CRS-enhetene (her: m²). Del på 1e6 for km².
    gdf["areal_km2"] = gdf_m.geometry.area / 1_000_000

    # Sentroide regnes i meter-CRS for geometrisk korrekthet, deretter
    # konverteres til lat/lon (EPSG:4326) for ML-features som er lett å tolke.
    sentroider_m = gdf_m.geometry.centroid
    sentroider_grader = gpd.GeoSeries(sentroider_m, crs="EPSG:25833").to_crs("EPSG:4326")
    gdf["sentroide_lon"] = sentroider_grader.x
    gdf["sentroide_lat"] = sentroider_grader.y

    # Bygg storby-punkter. Vi får dem opprinnelig som (lat, lon) i 4326,
    # men shapely Point tar (x, y) = (lon, lat). Pass på rekkefølgen!
    storby_punkter_4326 = gpd.GeoSeries(
        [Point(lon, lat) for _, lat, lon in STORBYER], crs="EPSG:4326"
    )
    # Reprojiser også storbyene til 25833 så avstanden blir i meter
    storby_punkter_m = storby_punkter_4326.to_crs("EPSG:25833")
    storby_navn = [navn for navn, _, _ in STORBYER]

    # Avstandsmatrise: én kolonne per storby, én rad per postnummer-sentroide.
    # GeoSeries.distance er elementvis mellom seriene; vi deler på 1000 for km.
    avstander = pd.DataFrame(
        {
            navn: sentroider_m.distance(punkt) / 1000
            for navn, punkt in zip(storby_navn, storby_punkter_m)
        }
    )

    # Tre features fra matrisen: avstand til Oslo spesifikt (mest brukt som
    # proxy for sentralitet), avstand til nærmeste storby uansett hvilken,
    # og navnet på nærmeste storby (kategorisk feature).
    gdf["avstand_oslo_km"] = avstander["Oslo"].values
    gdf["avstand_naermeste_storby_km"] = avstander.min(axis=1).values
    gdf["naermeste_storby"] = avstander.idxmin(axis=1).values

    # Klima-normaler: enklere proxy fra nærmeste storby. Krever ingen API-
    # nøkkel og er stabil over tid. For mer presis kommune-klima kreves
    # frost.met.no med Frost-stasjon-mapping — se README for fremtidig arbeid.
    gdf["temperatur_normal"] = gdf["naermeste_storby"].map(
        lambda by: KLIMA_NORMALER[by][0]
    )
    gdf["nedbor_normal_mm"] = gdf["naermeste_storby"].map(
        lambda by: KLIMA_NORMALER[by][1]
    )

    # Drop geometri-kolonnen fra output — den finnes allerede i
    # postnummer_geometri.parquet og blir lagt på i merge-fasen.
    cols = [
        "postnummer", "areal_km2",
        "sentroide_lat", "sentroide_lon",
        "avstand_oslo_km", "avstand_naermeste_storby_km", "naermeste_storby",
        "temperatur_normal", "nedbor_normal_mm",
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
