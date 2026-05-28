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

# Tegn som Excel tolker som starten på en formel hvis de står først i en
# celle. Hvis et kommunenavn eller geometri-streng begynner med ett av disse,
# kan Excel utføre formelen i stedet for å vise teksten — en kjent klasse
# av CSV-injection-angrep. Vi prefikser med apostrof for å forhindre det.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _sanitize_cell(v: object) -> object:
    """Prefiks en streng med apostrof hvis den starter med et formel-tegn.

    Påvirker bare strenger (numeriske kolonner går uendret). Bring-registeret
    er trolig trygt i dag, men dette er en billig forsvarslinje hvis en kilde
    endrer seg eller datasettet senere blandes med brukerinput.
    """
    if isinstance(v, str) and v.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + v
    return v


def export(med_geometri: bool) -> None:
    """Eksporter Parquet-datasettet til CSV.

    CSV kan ikke representere geometri-objekter direkte, så vi har to valg:
    droppe geometri-kolonnen (default — gir vanlig tabell), eller konvertere
    til WKT-streng (gir lesbar polygontekst men dobler filstørrelsen).
    """
    if not SRC.exists():
        raise FileNotFoundError(f"{SRC} finnes ikke — kjør pipelinen først")

    gdf = gpd.read_parquet(SRC)

    if med_geometri:
        # WKT (Well-Known Text) er en standard tekstrepresentasjon av geometri.
        # Kan re-parses med shapely.wkt.loads hvis man trenger polygonene
        # tilbake senere.
        gdf["geometry_wkt"] = gdf["geometry"].to_wkt()
        df = gdf.drop(columns=["geometry"])
        out_path = FINAL / "boligdata_final_med_geometri.csv"
    else:
        df = gdf.drop(columns=["geometry"])
        out_path = FINAL / "boligdata_final.csv"

    # Sanitér bare string-kolonner. Numeriske kolonner kan ikke starte med
    # formel-tegn så vi sparer en applymap-runde over hele datasettet.
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].map(_sanitize_cell)

    # utf-8-sig skriver BOM (byte-order-mark) først i filen. Excel trenger
    # dette for å gjenkjenne UTF-8 og vise æøå riktig — uten BOM får man
    # rare tegn for norske bokstaver.
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
