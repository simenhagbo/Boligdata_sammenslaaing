"""
Sjekker join-integritet og dataplausibilitet i tidsserie-datasettet.

Kjør etter pipelinen for å verifisere:
  - korrekt antall rader og struktur (postnummer × år)
  - SSB-verdier er like for alle postnummer i samme (kommune, år)
  - andeler summerer til 1 per rad
  - kjente kommuner har plausible verdier i et gitt år

Bruk:
  python source/verify.py
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

FINAL = Path(__file__).parents[1] / "data" / "processed_data" / "final" / "boligdata_final.parquet"


# Spot-tester på siste år (2024) — der dekningen er best
SPOT_TESTS_2024 = [
    # (postnummer, kommunenavn, min_bef, max_bef, min_pris, max_pris)
    ("0010", "Oslo",        700_000, 750_000, 85_000, 110_000),
    ("5003", "Bergen",      280_000, 300_000, 45_000, 75_000),
    ("7011", "Trondheim",   200_000, 220_000, 45_000, 70_000),
    ("4006", "Stavanger",   140_000, 155_000, 40_000, 65_000),
]


def check(label: str, ok: bool, details: str = "") -> bool:
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}{(': ' + details) if details else ''}")
    return ok


def main() -> None:
    if not FINAL.exists():
        print(f"FEIL: {FINAL} finnes ikke. Kjør 'python source/pipeline.py' først.")
        return

    df = gpd.read_parquet(FINAL)
    print(f"Lastet datasett: {df.shape[0]:,} rader × {df.shape[1]} kolonner")
    print(f"  {df['postnummer'].nunique():,} unike postnummer × {df['aar'].nunique()} år\n")

    failures = 0

    print("1. Struktur:")
    if not check("Ingen duplikater på (postnummer, aar)",
                 not df.duplicated(["postnummer", "aar"]).any()):
        failures += 1
    if not check("aar har 23 unike verdier (2002-2024)",
                 sorted(df["aar"].unique().tolist()) == list(range(2002, 2025))):
        failures += 1
    expected_rows = df["postnummer"].nunique() * df["aar"].nunique()
    if not check(f"Antall rader matcher kryss-produkt ({expected_rows:,})",
                 len(df) == expected_rows, f"{len(df):,}"):
        failures += 1
    if not check("Alle postnummer er 4-sifrede strenger",
                 df["postnummer"].str.match(r"^\d{4}$").all()):
        failures += 1
    if not check("Geometry finnes for alle rader",
                 df["geometry"].notna().all()):
        failures += 1

    print("\n2. Andeler summerer til 1 (per rad):")
    for navn, prefiks in [
        ("BygnType", "andel_bygntype_"),
        ("Byggeår", "andel_byggeaar_"),
        ("Bruksareal", "andel_areal_"),
    ]:
        cols = [c for c in df.columns if c.startswith(prefiks)]
        if not cols:
            continue
        sums = df[cols].sum(axis=1)
        valid = sums[sums > 0]
        ok = ((valid - 1.0).abs() < 0.001).all()
        if not check(f"{navn}: alle rader summerer til 1.0", ok,
                     f"min={valid.min():.4f}, max={valid.max():.4f}"):
            failures += 1

    print("\n3. SSB-verdier konsistente innen (kommune, år):")
    # Alle postnummer i samme kommune & år skal ha identiske SSB-verdier
    kommune_aar_cols = [
        "befolkning", "inntekt_etter_skatt", "antall_boliger",
        "pris_kvm_alle_kommune", "median_byggeaar_kommune",
        "median_bruksareal_kommune", "antall_sysselsatte",
    ]
    inkonsistens = []
    for col in kommune_aar_cols:
        if col not in df.columns:
            continue
        n_unique = df.groupby(["kommune_nr", "aar"])[col].nunique(dropna=True).max()
        if n_unique > 1:
            inkonsistens.append((col, int(n_unique)))
    if not check("Hver (kommune, år) har én unik verdi per SSB-kolonne",
                 not inkonsistens, str(inkonsistens) if inkonsistens else ""):
        failures += 1

    print("\n4. Geo-features statiske over år:")
    # Areal, sentroide og avstand til storby skal være likt for samme postnummer over alle år
    geo_static = ["areal_km2", "sentroide_lat", "avstand_oslo_km"]
    static_ok = True
    for col in geo_static:
        if col not in df.columns:
            continue
        if df.groupby("postnummer")[col].nunique().max() != 1:
            static_ok = False
            break
    if not check("areal_km2, sentroide_lat, avstand_oslo_km er konstante per postnummer",
                 static_ok):
        failures += 1

    print("\n5. Spot-tester for år 2024 (best dekning):")
    df_2024 = df[df["aar"] == 2024]
    for pnr, kommune, bef_min, bef_max, pris_min, pris_max in SPOT_TESTS_2024:
        row = df_2024[df_2024["postnummer"] == pnr]
        if len(row) == 0:
            print(f"  [SKIP] {pnr} ({kommune})")
            continue
        r = row.iloc[0]
        ok_kommune = r["kommunenavn"].lower() == kommune.lower()
        ok_bef = bef_min <= (r["befolkning"] or 0) <= bef_max
        pris = r.get("pris_kvm_alle_kommune")
        ok_pris = pris_min <= (pris or 0) <= pris_max
        details = (f"{r['kommunenavn']}, befolkning={r['befolkning']:,.0f}, "
                   f"pris={pris:,.0f}" if pd.notna(pris) else f"{r['kommunenavn']}, pris=NaN")
        if not check(f"{pnr} ({kommune}) 2024",
                     ok_kommune and ok_bef and ok_pris, details):
            failures += 1

    print("\n6. Geografisk plausibilitet:")
    oslo = df[df["kommunenavn"] == "Oslo"]
    if len(oslo) > 0:
        if not check("Oslo-postnummer har 'Oslo' som nærmeste storby",
                     (oslo["naermeste_storby"] == "Oslo").all()):
            failures += 1
    if not check("Sentroider innen Norge (lat 57-81, lon -10-32)",
                 df["sentroide_lat"].between(57, 81).all() and
                 df["sentroide_lon"].between(-10, 32).all()):
        failures += 1

    print("\n7. Dekning på pris (mål-variabel):")
    if "pris_kvm_alle_kommune" in df.columns:
        cov_2024 = df_2024["pris_kvm_alle_kommune"].notna().mean()
        # ~10% av kommunene (små, få omsetninger) sensureres av SSB selv i 2024
        if not check("pris_kvm_alle_kommune dekning 2024 >= 90%",
                     cov_2024 >= 0.90, f"{cov_2024:.1%}"):
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALLE SJEKKER PASSERTE")
    else:
        print(f"{failures} FEIL — se loggen over")
    print("=" * 60)


if __name__ == "__main__":
    main()
