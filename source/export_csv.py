"""
Eksporterer det ferdige datasettet til CSV.

To varianter:
  --med-geometri   beholder geometri-kolonnen som WKT-streng
  --uten-geometri  dropper geometri (standard — gir vanlig CSV som åpnes i Excel)

Bruk:
  python source/export_csv.py
  python source/export_csv.py --med-geometri
"""

import argparse
from pathlib import Path

import geopandas as gpd

FINAL = Path(__file__).parents[1] / "data" / "processed_data" / "final"
SRC = FINAL / "boligdata_final.parquet"


def export(med_geometri: bool) -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} finnes ikke — kjør pipelinen først")

    gdf = gpd.read_parquet(SRC)

    if med_geometri:
        # WKT (Well-Known Text) er en lesbar streng-representasjon av geometri
        gdf["geometry_wkt"] = gdf["geometry"].to_wkt()
        df = gdf.drop(columns=["geometry"])
        out_path = FINAL / "boligdata_final_med_geometri.csv"
    else:
        df = gdf.drop(columns=["geometry"])
        out_path = FINAL / "boligdata_final.csv"

    # Bruker UTF-8 BOM så Excel viser æøå riktig
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    size_kb = out_path.stat().st_size / 1024
    print(f"Skrevet: {out_path}")
    print(f"  {len(df)} rader × {len(df.columns)} kolonner ({size_kb:.0f} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Eksporter datasett til CSV")
    parser.add_argument("--med-geometri", action="store_true",
                        help="Behold polygon som WKT-streng (gir stor fil)")
    args = parser.parse_args()
    export(med_geometri=args.med_geometri)


if __name__ == "__main__":
    main()
