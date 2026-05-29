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
    """Skriv ut en passert/feilet-rad og returner status for opptelling."""
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}{(': ' + details) if details else ''}")
    return ok


def main() -> None:
    # Avbryt hvis pipelinen ikke er kjørt enda
    if not FINAL.exists():
        print(f"FEIL: {FINAL} finnes ikke. Kjør 'python source/pipeline.py' først.")
        return

    # Last datasettet og rapporter dimensjoner
    df = gpd.read_parquet(FINAL)
    print(f"Lastet datasett: {df.shape[0]:,} rader × {df.shape[1]} kolonner")
    print(f"  {df['postnummer'].nunique():,} unike postnummer × {df['aar'].nunique()} år\n")

    # Teller feilede sjekker for samlet status på slutten
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
    # andel_*_kommune-kolonnene representerer en fordeling og skal summere til
    # 1.0 (per kommune, per år). Vi filtrerer ut rader med sum=0 fordi de
    # mangler data — bare gyldige fordelinger sjekkes.
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
        # Tolererer mindre avvik fra 1.0 pga. flyttalls-presisjon
        ok = ((valid - 1.0).abs() < 0.001).all()
        if not check(f"{navn}: alle rader summerer til 1.0", ok,
                     f"min={valid.min():.4f}, max={valid.max():.4f}"):
            failures += 1

    # Alle andel_*-kolonner skal være brøker i [0, 1] — fanger skala-feil der
    # en kolonne ved en feil ligger i prosent (0-100) i stedet for brøk.
    andel_cols = [c for c in df.columns if c.startswith("andel_")]
    over_en = [c for c in andel_cols if (df[c].dropna() > 1.001).any()]
    if not check("Alle andel_*-kolonner er brøker i [0, 1]",
                 not over_en, str(over_en) if over_en else ""):
        failures += 1

    print("\n3. SSB-verdier konsistente innen (kommune, år):")
    # SSB-data er på kommunenivå, så alle ~10 postnummer i samme kommune
    # skal ha identiske verdier for et gitt år. Hvis en kolonne har flere
    # unike verdier per (kommune, år), har join-logikken gått galt.
    kommune_aar_cols = [
        "befolkning", "inntekt_etter_skatt", "antall_boliger",
        "pris_kvm_alle_kommune", "median_byggeaar_kommune",
        "median_bruksareal_kommune", "antall_sysselsatte",
    ]
    inkonsistens = []
    for col in kommune_aar_cols:
        if col not in df.columns:
            continue
        # nunique med dropna=True ignorerer NaN — kommuner uten data er OK
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
    # Bruker 2024 fordi det er året med høyest data-dekning — eldre år har
    # ofte manglende verdier pga. kommunesammenslåinger
    df_2024 = df[df["aar"] == 2024]
    for pnr, kommune, bef_min, bef_max, pris_min, pris_max in SPOT_TESTS_2024:
        row = df_2024[df_2024["postnummer"] == pnr]
        if len(row) == 0:
            print(f"  [SKIP] {pnr} ({kommune})")
            continue
        r = row.iloc[0]
        # Sjekker at: postnummeret faktisk tilhører forventet kommune,
        # befolkningstallet er i rimelig range, og prisen er plausibel
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
    # Oslo-postnummer skal selvsagt være nærmest Oslo, ikke en av de andre storbyene
    oslo = df[df["kommunenavn"] == "Oslo"]
    if len(oslo) > 0:
        if not check("Oslo-postnummer har 'Oslo' som nærmeste storby",
                     (oslo["naermeste_storby"] == "Oslo").all()):
            failures += 1
    # Bbox-sjekk for hele Norge: lat 57-81, lon -10-32
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

    print("\n8. Makrodata (nasjonale, lik for alle kommuner i et gitt år):")
    # KPI og styringsrente skal ha 100% dekning for alle år 2002-2024
    for col in ("kpi_indeks", "kpi_endring_pct", "styringsrente"):
        if col not in df.columns:
            continue
        cov = df[col].notna().mean()
        if not check(f"{col} dekning 100%", cov >= 0.999, f"{cov:.1%}"):
            failures += 1
    # Hver makro-verdi skal være lik på tvers av postnummer for et gitt år
    # (nasjonal data). Sjekk én år for å være sikker.
    for col in ("kpi_indeks", "styringsrente"):
        if col not in df.columns:
            continue
        unik_per_aar = df.groupby("aar")[col].nunique(dropna=True).max()
        if not check(f"{col} unik per år", unik_per_aar <= 1,
                     f"max {int(unik_per_aar)} verdier per år"):
            failures += 1
    # KPI skal være monotont stigende — Norge har ikke hatt deflasjon over år
    if "kpi_indeks" in df.columns:
        kpi_aarlig = df.drop_duplicates("aar")[["aar", "kpi_indeks"]].sort_values("aar")
        monotont = (kpi_aarlig["kpi_indeks"].diff().dropna() > 0).all()
        if not check("KPI monotont stigende 2002-2024", monotont):
            failures += 1
    # Spot-sjekk: KPI 2024 mot 2015 — Norge har hatt ~30% inflasjon på 9 år
    if "kpi_indeks" in df.columns:
        kpi_2015 = df[df["aar"] == 2015]["kpi_indeks"].iloc[0] if (df["aar"] == 2015).any() else None
        kpi_2024 = df[df["aar"] == 2024]["kpi_indeks"].iloc[0] if (df["aar"] == 2024).any() else None
        if kpi_2015 and kpi_2024:
            if not check("KPI 2015=100 og 2024 i intervall 125-140",
                         abs(kpi_2015 - 100) < 0.01 and 125 <= kpi_2024 <= 140,
                         f"2015={kpi_2015:.1f}, 2024={kpi_2024:.1f}"):
                failures += 1

    print("\n9. Investerings-features (per kommune, lookahead-sikre):")
    if "prisstigning_real_pct" in df.columns:
        # Real prisstigning skal ha plausibelt spenn — typisk -40 til +60% per år
        vals = df["prisstigning_real_pct"].dropna()
        if not check("prisstigning_real_pct i [-60, 100]",
                     vals.between(-60, 100).all() if len(vals) else True,
                     f"min={vals.min():.1f}, max={vals.max():.1f}" if len(vals) else "tom"):
            failures += 1
    if "cagr_5aar_real_pct" in df.columns:
        vals = df["cagr_5aar_real_pct"].dropna()
        # 5-års CAGR er mer stabil — typisk -15 til +20% per år
        if not check("cagr_5aar_real_pct i [-20, 25]",
                     vals.between(-20, 25).all() if len(vals) else True,
                     f"min={vals.min():.1f}, max={vals.max():.1f}" if len(vals) else "tom"):
            failures += 1
    if "volatilitet_5aar" in df.columns:
        vals = df["volatilitet_5aar"].dropna()
        # Volatilitet (log-return-std) skal være ikke-negativ og typisk <0.5
        if not check("volatilitet_5aar i [0, 0.5]",
                     vals.between(0, 0.5).all() if len(vals) else True,
                     f"max={vals.max():.3f}" if len(vals) else "tom"):
            failures += 1

    print("\n" + "=" * 60)
    if failures == 0:
        print("ALLE SJEKKER PASSERTE")
    else:
        print(f"{failures} FEIL — se loggen over")
    print("=" * 60)


if __name__ == "__main__":
    main()
