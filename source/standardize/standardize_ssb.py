"""
Parser SSB-rådata til ett tidsserie-Parquet på (postnummer, år)-format.

Alle postnummer i samme kommune deler SSB-verdiene for et gitt år — det er
en bevisst forenkling siden SSB ikke publiserer på postnummer-nivå.
"""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parents[2] / "data" / "raw_data" / "ssb"
STD_DIR = Path(__file__).parents[2] / "data" / "processed_data" / "standardized"
STD_DIR.mkdir(parents=True, exist_ok=True)


# ----- JSON-stat parsing -----

def parse_jsonstat(path: Path) -> pd.DataFrame:
    """Flat ut JSON-stat2 — én rad per dimensjonskombinasjon.

    JSON-stat lagrer verdier som en flat liste der rekkefølgen er det kartesiske
    produktet av dimensjonene (dvs. siste dimensjon varierer raskest). For å få
    tilbake hver verdi sin (dim1, dim2, ...)-tuple bruker vi itertools.product
    over kategori-kodene i samme rekkefølge som SSB returnerte dem.
    """
    with open(path, encoding="utf-8") as f:
        js = json.load(f)

    dims = js["id"]                       # navn på dimensjoner, f.eks. ["Region", "Tid"]
    labels = js["dimension"]              # kategori-info per dimensjon
    values = js["value"]                  # den flate verdilisten

    # (kode, etikett)-par per dimensjon. dict bevarer insertion order fra Python 3.7+,
    # noe vi er avhengig av for at zip-en under skal matche verdienes rekkefølge.
    cats = [list(labels[d]["category"]["label"].items()) for d in dims]

    rows = []
    for combo, val in zip(itertools.product(*cats), values):
        row = {dim: code for dim, (code, _) in zip(dims, combo)}
        row["value"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def parse_jsonstat_chunks(prefix: str) -> pd.DataFrame:
    """Les alle `{prefix}_partNN.json`-filer og slå sammen.

    Brukes for tabellene som var for store til å hentes i én SSB-query (06266
    og 06513) og som derfor ble splittet på Tid i collect-fasen.
    """
    files = sorted(RAW_DIR.glob(f"{prefix}_part*.json"))
    if not files:
        # Fallback: hvis tabellen likevel ble hentet som enkeltfil (lite range)
        single = RAW_DIR / f"{prefix}.json"
        if single.exists():
            return parse_jsonstat(single)
        return pd.DataFrame()
    return pd.concat([parse_jsonstat(f) for f in files], ignore_index=True)


def _find_col(df: pd.DataFrame, name_lower: str) -> str:
    """Finn første kolonne hvis lowercase-navn matcher. Case-insensitivt oppslag
    fordi SSB av og til varierer casing mellom tabeller (Region vs region).
    """
    for c in df.columns:
        if c.lower() == name_lower:
            return c
    raise KeyError(f"Fant ikke kolonne '{name_lower}' i {list(df.columns)}")


# ----- Backbone: postnummer × år -----

def load_backbone() -> pd.DataFrame:
    """Bygger postnummer × år-grunnlaget (5122 postnummer × 23 år).

    Merk: vi bruker Bring-mappingen (5122 postnummer) som er bredere enn
    geometri-Parquet (3378). I merge-fasen filtreres det til de 3378 med
    polygon-geometri, så de "ekstra" postnumrene her (postboks-postnummer
    uten geografisk område) faller naturlig fra.
    """
    mapping_path = STD_DIR / "postnummer_kommune_mapping.parquet"
    if not mapping_path.exists():
        raise FileNotFoundError("Kjør standardize_kartverket.py før denne")
    mapping = pd.read_parquet(mapping_path)[["postnummer", "kommune_nr"]]

    # Kryss-produkt med år 2002-2024 — gir én rad per (postnummer, år)
    aar = pd.DataFrame({"aar": list(range(2002, 2025))})
    return mapping.merge(aar, how="cross")


# ----- Boliger per type (06265) -----

def standardize_boliger_per_type() -> pd.DataFrame:
    """Totaltall, modal og andeler per BygnType per (kommune, år) fra 06265.

    BygnType-koder: 01=enebolig, 02=tomannsbolig, 03=rekkehus,
    04=blokk, 05=bofellesskap, 999=annet.
    """
    src = RAW_DIR / "boliger_per_type_06265.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "bygntype")
    tid_col = _find_col(df, "tid")

    # Vi dropper 999 (annet) fra andelsberegningen — den er en sekkebetegnelse
    # som forstyrrer fordelingstolkningen
    main_types = ["01", "02", "03", "04", "05"]
    main = df[df[type_col].isin(main_types)].copy()

    # Pivot til wide: én kolonne per BygnType, fill_value=0 for kombinasjoner
    # uten data (typisk små kommuner som ikke har f.eks. blokk)
    wide = main.pivot_table(
        index=[region_col, tid_col], columns=type_col,
        values="value", aggfunc="sum",
    ).fillna(0)
    # Garanter at alle fem typer er kolonner selv om noen mangler i dataene
    for t in main_types:
        if t not in wide.columns:
            wide[t] = 0.0
    wide = wide[main_types]

    totals = wide.sum(axis=1).rename("antall_boliger")
    # Andelene er bare meningsfulle der totalen er > 0 — ellers NaN
    andeler = wide.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"andel_bygntype_{t}_kommune" for t in main_types]
    # Modal = den BygnType-koden med høyest antall i denne kommunen og året
    modal = wide.idxmax(axis=1).rename("modal_boligtype_kommune")

    out = pd.concat([totals, modal, andeler], axis=1).reset_index()
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    # SSB returnerer kommunenummer uten ledende null for 1-3-sifrede koder
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Folkemengde (06913) -----

def standardize_folkemengde() -> pd.DataFrame:
    """Folkemengde per kommune × år fra 06913 (ContentsCode=Folkemengde)."""
    src = RAW_DIR / "folkemengde_06913.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar", "value": "befolkning"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "befolkning"]]


# ----- Inntekt (12558) -----

def standardize_inntekt() -> pd.DataFrame:
    """Median inntekt etter skatt per kommune × år (12558, Desil 5)."""
    src = RAW_DIR / "inntekt_12558.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "inntekt_etter_skatt"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    df = df[["kommune_nr", "aar", "inntekt_etter_skatt"]].dropna()
    return df.drop_duplicates(["kommune_nr", "aar"])


# ----- Priser (06035) -----

# Mapping fra Boligtype-kode til kolonne-suffiks
_PRIS_TYPE_LABEL = {"01": "enebolig", "02": "smaahus", "03": "blokk"}


def standardize_priser() -> pd.DataFrame:
    """Kvm-pris per boligtype + vektet snitt og antall omsetninger per (kommune, år).

    SSB 06035 rapporterer to ContentsCode-verdier per (kommune, boligtype, år):
    KvPris (kr/m²) og Omsetninger (antall salg). Vi vil ha begge som separate
    kolonner, og deretter et vektet snitt over de tre boligtypene (enebolig,
    småhus, blokk) der vekten er antall omsetninger.

    Datakvalitet: små kommuner med få salg får sensurerte verdier (typisk
    omsetninger=0 og pris=NaN). Vi håndterer dette eksplisitt for å unngå
    "0 kr/m²"-falske verdier i sluttdatasettet.
    """
    src = RAW_DIR / "priser_06035.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "boligtype")
    content_col = _find_col(df, "contentscode")
    tid_col = _find_col(df, "tid")

    # Vi vil ha to kolonner (kvpris, omsetninger) per (Region, Boligtype, Tid).
    # pivot_table med aggfunc="sum" konverterer NaN til 0 (sum av NaN-er er 0
    # når skipna=True), som gir oss falske "0 kr/m²"-verdier for sensurerte
    # kommuner. Vi bruker derfor merge på filtrerte deler i stedet — det
    # bevarer NaN korrekt.
    kvpris_df = df[df[content_col] == "KvPris"][
        [region_col, type_col, tid_col, "value"]
    ].rename(columns={"value": "kvpris"})
    oms_df = df[df[content_col] == "Omsetninger"][
        [region_col, type_col, tid_col, "value"]
    ].rename(columns={"value": "omsetninger"})
    pivoted = kvpris_df.merge(oms_df, on=[region_col, type_col, tid_col], how="outer")

    # SSB rapporterer omsetninger=0 og kvpris=NaN for sensurerte små kommuner.
    # Sett kvpris=NaN også der omsetninger=0 for konsistens — det er bedre at
    # alle "ingen data"-tilfeller ser like ut.
    pivoted.loc[pivoted["omsetninger"].fillna(0) == 0, "kvpris"] = np.nan

    # Totalt antall omsetninger per (kommune, år) — uavhengig av om kvm-pris
    # finnes. Brukbart som feature uansett (kommuner med høy omsetning =
    # aktivt boligmarked).
    oms_total = pivoted.groupby([region_col, tid_col])["omsetninger"].sum().reset_index()
    oms_total = oms_total.rename(columns={"omsetninger": "antall_omsetninger_kommune"})

    # Vektet snitt-pris over de tre boligtypene. Logikken:
    #   pris_kvm_alle = sum(kvpris × omsetninger) / sum(omsetninger)
    # men bare for rader der kvpris finnes. Hvis vi tok med rader der
    # kvpris=NaN men omsetninger>0 ville sum(kvpris×omsetninger) blitt
    # liten/null mens nevneren ble stor → falsk lav gjennomsnittspris.
    valid = pivoted.dropna(subset=["kvpris"]).copy()
    valid["pris_x_vekt"] = valid["kvpris"] * valid["omsetninger"]
    alle = valid.groupby([region_col, tid_col]).agg(
        pris_x_vekt_sum=("pris_x_vekt", "sum"),
        omsetninger_med_pris=("omsetninger", "sum"),
    ).reset_index()
    alle["pris_kvm_alle_kommune"] = (
        alle["pris_x_vekt_sum"] / alle["omsetninger_med_pris"].replace(0, np.nan)
    )
    # Sett oms_total tilbake inn — den ble beregnet før vi dropet NaN-radene
    alle = alle.merge(oms_total, on=[region_col, tid_col], how="outer")
    alle = alle.rename(columns={region_col: "kommune_nr", tid_col: "aar"})[
        ["kommune_nr", "aar", "pris_kvm_alle_kommune", "antall_omsetninger_kommune"]
    ]
    alle["kommune_nr"] = alle["kommune_nr"].astype(str).str.zfill(4)
    alle["aar"] = alle["aar"].astype(int)

    # Per-type kvm-pris som egne kolonner. Bruker aggfunc="first" siden vi
    # uansett bare har én rad per (region, type, år) etter merge.
    per_type = pivoted.pivot_table(
        index=[region_col, tid_col], columns=type_col,
        values="kvpris", aggfunc="first",
    )
    per_type = per_type.rename(
        columns={k: f"pris_kvm_{v}_kommune" for k, v in _PRIS_TYPE_LABEL.items()}
    )
    # Sikre at alle tre type-kolonner finnes
    for label in _PRIS_TYPE_LABEL.values():
        col = f"pris_kvm_{label}_kommune"
        if col not in per_type.columns:
            per_type[col] = np.nan
    per_type = per_type.reset_index().rename(
        columns={region_col: "kommune_nr", tid_col: "aar"}
    )
    per_type["kommune_nr"] = per_type["kommune_nr"].astype(str).str.zfill(4)
    per_type["aar"] = per_type["aar"].astype(int)

    return per_type.merge(alle, on=["kommune_nr", "aar"], how="outer")


# ----- Byggeår og bruksareal-fordeling -----

# SSBs 13 BygnAr-kategorier slås sammen til fem perioder. Midtpunkt er omtrentlig
# år for vektet median (sentrum av intervallet, eller estimat for åpen kategori
# som "1900 og tidligere"). Kode 99 = ukjent og ekskluderes.
_BYGGEAAR_BUCKETS = {
    "01": ("for1946", 1880),     # 1900 og tidligere (estimat)
    "02": ("for1946", 1910),     # 1901-1920
    "03": ("for1946", 1930),     # 1921-1940
    "04": ("for1946", 1943),     # 1941-1945
    "06": ("1946_1970", 1953),   # 1946-1960
    "07": ("1946_1970", 1965),   # 1961-1970
    "08": ("1971_1990", 1975),   # 1971-1980
    "09": ("1971_1990", 1985),   # 1981-1990
    "10": ("1991_2010", 1995),   # 1991-2000
    "11": ("1991_2010", 2005),   # 2001-2010
    "12": ("etter2010", 2015),   # 2011-2020
    "13": ("etter2010", 2022),   # 2021 og etter
}
_BYGGEAAR_PERIODER = ["for1946", "1946_1970", "1971_1990", "1991_2010", "etter2010"]

# Tilsvarende: SSBs 15 BruksAreal-grupper slås sammen til fem størrelser.
# Midtpunktene brukes til vektet median-beregning.
_BRUKSAREAL_BUCKETS = {
    "1":  ("under60", 20),   # Under 30 kvm
    "2":  ("under60", 35),   # 30-39 kvm
    "3":  ("under60", 45),   # 40-49 kvm
    "4":  ("under60", 55),   # 50-59 kvm
    "5":  ("60_99", 70),     # 60-79 kvm
    "6":  ("60_99", 90),     # 80-99 kvm
    "50": ("100_159", 110),  # 100-119 kvm
    "51": ("100_159", 130),  # 120-139 kvm
    "52": ("100_159", 150),  # 140-159 kvm
    "53": ("160_249", 180),  # 160-199 kvm
    "54": ("160_249", 225),  # 200-249 kvm
    "55": ("over250", 275),  # 250-299 kvm
    "56": ("over250", 325),  # 300-349 kvm
    "57": ("over250", 400),  # 350 kvm eller større (estimat)
}
_BRUKSAREAL_GRUPPER = ["under60", "60_99", "100_159", "160_249", "over250"]


def _vektet_median(midtpunkter: np.ndarray, vekter: np.ndarray) -> float:
    """Median fra en (midtpunkt, vekt)-histogramfordeling.

    Antar at midtpunktene er sortert stigende. Bruker kumulativ vekt-sum til å
    finne den bøtten der vi passerer halvparten av total vekt — det blir vår
    median. Litt grov siden vi ikke interpolerer innen bøtten, men presis nok
    for SSBs 5-bøtte-fordelinger.
    """
    total = vekter.sum()
    if total == 0:
        return np.nan
    kumulativ = np.cumsum(vekter)
    # searchsorted finner posisjonen der vi krysser total/2-grensa
    idx = min(int(np.searchsorted(kumulativ, total / 2)), len(midtpunkter) - 1)
    return float(midtpunkter[idx])


def _andeler_og_median_tidsserie(
    df: pd.DataFrame, region_col: str, tid_col: str, bucket_col: str,
    buckets: dict, gruppe_navn: list[str], prefix: str, median_col: str,
) -> pd.DataFrame:
    """Beregn andeler + vektet median per (kommune, år) fra en histogram-tabell.

    Felles logikk for byggeår og bruksareal: vi får SSB-rader på (Region, Tid,
    bucket-kode, BygnType, value)-form. Mapper bucket-kodene til våre fem
    grupper, summerer per gruppe, og deler på totalen for å få andeler.
    """
    # Filtrer ut bucket-koder vi ikke bryr oss om (typisk "99 = ukjent")
    work = df[df[bucket_col].isin(buckets)].copy()
    work["gruppe"] = work[bucket_col].map(lambda c: buckets[c][0])
    work["midtpunkt"] = work[bucket_col].map(lambda c: buckets[c][1])

    # Summer antall boliger per (kommune, år, gruppe), så pivot til wide
    grupped = work.groupby([region_col, tid_col, "gruppe"])["value"].sum().unstack(fill_value=0)
    # Sørg for at alle gruppene finnes som kolonner (selv hvis ingen rader matchet)
    for g in gruppe_navn:
        if g not in grupped.columns:
            grupped[g] = 0.0
    grupped = grupped[gruppe_navn]
    totals = grupped.sum(axis=1)
    andeler = grupped.div(totals.replace(0, np.nan), axis=0)
    andeler.columns = [f"{prefix}_{g}_kommune" for g in gruppe_navn]

    # Vektet median: sortér på midtpunkt slik at cumsum gir kumulativ fordeling
    work_sorted = work.sort_values([region_col, tid_col, "midtpunkt"])
    medians = work_sorted.groupby([region_col, tid_col]).apply(
        lambda g: _vektet_median(g["midtpunkt"].values, g["value"].values),
        include_groups=False,
    ).rename(median_col)

    return pd.concat([andeler, medians], axis=1).reset_index()


def standardize_byggeaar() -> pd.DataFrame:
    """Byggeår-fordeling i fem perioder + median per (kommune, år)."""
    df = parse_jsonstat_chunks("byggeaar_06266")
    if df.empty:
        return pd.DataFrame()

    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bygnar")
    tid_col = _find_col(df, "tid")

    out = _andeler_og_median_tidsserie(
        df, region_col, tid_col, bucket_col,
        _BYGGEAAR_BUCKETS, _BYGGEAAR_PERIODER,
        prefix="andel_byggeaar", median_col="median_byggeaar_kommune",
    )
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


def standardize_bruksareal() -> pd.DataFrame:
    """Bruksareal-fordeling i fem grupper + median per (kommune, år)."""
    df = parse_jsonstat_chunks("bruksareal_06513")
    if df.empty:
        return pd.DataFrame()

    region_col = _find_col(df, "region")
    bucket_col = _find_col(df, "bruksareal")
    tid_col = _find_col(df, "tid")

    out = _andeler_og_median_tidsserie(
        df, region_col, tid_col, bucket_col,
        _BRUKSAREAL_BUCKETS, _BRUKSAREAL_GRUPPER,
        prefix="andel_areal", median_col="median_bruksareal_kommune",
    )
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Utdanning (09429) -----

# Mapping fra Nivaa-kode til kolonne-suffiks. "00" (i alt) brukes som nevner.
_UTD_NIVAA_LABEL = {
    "01": "grunnskole",
    "02a": "videregaaende",
    "11": "fagskole",
    "03a": "uh_kort",
    "04a": "uh_lang",
    "09": "uoppgitt",
}


def standardize_utdanning() -> pd.DataFrame:
    """Andel per utdanningsnivå per (kommune, år) fra 09429.

    SSB rapporterer prosent direkte (PersonerProsent), så vi bare deler på 100
    for å få 0-1-skala som matcher de andre andelene i datasettet.
    """
    src = RAW_DIR / "utdanning_09429.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    nivaa_col = _find_col(df, "nivaa")
    tid_col = _find_col(df, "tid")

    df["andel"] = df["value"] / 100.0  # 0-1 i stedet for 0-100

    # Filtrer på de seks utdanningsnivåene vi mapper til kolonner.
    # "00" (i alt) ligger også i rådata men brukes ikke siden alle andelene
    # uansett refererer til samme totalbefolkning.
    main = df[df[nivaa_col].isin(_UTD_NIVAA_LABEL)].copy()
    wide = main.pivot_table(
        index=[region_col, tid_col], columns=nivaa_col,
        values="andel", aggfunc="first",
    )
    wide = wide.rename(columns={k: f"andel_utdanning_{v}_kommune"
                                 for k, v in _UTD_NIVAA_LABEL.items()})
    # Sikre at alle seks kolonner finnes selv om noen mangler i rådata
    for v in _UTD_NIVAA_LABEL.values():
        col = f"andel_utdanning_{v}_kommune"
        if col not in wide.columns:
            wide[col] = np.nan

    out = wide.reset_index().rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Arbeidsledighet (10540) -----

def standardize_arbeidsledighet() -> pd.DataFrame:
    """Andel registrerte arbeidsledige per kommune × år fra 10540.

    Tid-verdiene er månedlig (`2020M11`-format) siden vi tar november-tall.
    Vi parser bare året ut og bruker det direkte — én observasjon per (kommune, år).
    """
    src = RAW_DIR / "arbeidsledighet_10540.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    # `2020M11` → 2020. Tar bare de første 4 tegnene.
    df["aar"] = df[tid_col].astype(str).str[:4].astype(int)
    df = df.rename(columns={region_col: "kommune_nr", "value": "andel_arbeidsledige_kommune"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    return df[["kommune_nr", "aar", "andel_arbeidsledige_kommune"]].drop_duplicates(
        ["kommune_nr", "aar"]
    )


# ----- Flytting (09588) -----

def standardize_flytting() -> pd.DataFrame:
    """Netto innflytting per kommune × år fra 09588."""
    src = RAW_DIR / "flytting_09588.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "netto_innflytting_kommune"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "netto_innflytting_kommune"]].drop_duplicates(
        ["kommune_nr", "aar"]
    )


# ----- Husholdninger (06070) -----

def standardize_husholdninger() -> pd.DataFrame:
    """Antall husholdninger + andel enslige per kommune × år fra 06070.

    HushType-kategoriene er en partisjon (hver husholdning hører til
    nøyaktig én), så vi summerer alle for totalantallet. Kode `001` =
    "Aleneboende" gir oss andel enslige.
    """
    src = RAW_DIR / "husholdninger_06070.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    type_col = _find_col(df, "hushtype")
    tid_col = _find_col(df, "tid")

    # Total per (kommune, år) ved å summere over alle HushType-kategorier.
    # SSB markerer manglende data som 0 (ikke None), så hvis ALLE 11
    # HushType-kategorier er 0 betyr det "ingen rapport" — vi konverterer
    # til NaN slik at andelen ikke blir misvisende.
    total = df.groupby([region_col, tid_col])["value"].sum().rename("antall_husholdninger_kommune")
    total = total.replace(0, np.nan)

    # Aleneboende = HushType-kode "001". Reindeks til samme indeks som total
    # og fill 0 — uten dette får (kommune, år)-par der "001" ikke ble rapportert
    # NaN, og andelen blir NaN selv om vi har totalt antall husholdninger.
    enslige = df[df[type_col] == "001"].groupby([region_col, tid_col])["value"].sum()
    enslige = enslige.reindex(total.index, fill_value=0)

    out = pd.concat([total, enslige.rename("_enslige")], axis=1).reset_index()
    # Andel enslige = enslige / total. Bruker .replace(0, NaN) for å unngå
    # divisjon-med-null der totalen er 0 (typisk små eller manglende rader).
    out["andel_enslige_kommune"] = out["_enslige"] / out["antall_husholdninger_kommune"].replace(0, np.nan)
    out = out.drop(columns=["_enslige"])
    out = out.rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out


# ----- Boligbygging (05940) -----

def standardize_boligbygging() -> pd.DataFrame:
    """Fullførte + igangsatte boliger per kommune × år fra 05940 (chunked).

    Vi summerer over alle Byggeareal-koder for boligbygg (alt unntatt "000"
    som er "andre bygg enn boligbygg"). Resultatet er totalt antall boliger
    i hver kategori.
    """
    df = parse_jsonstat_chunks("boligbygging_05940")
    if df.empty:
        return pd.DataFrame()

    region_col = _find_col(df, "region")
    area_col = _find_col(df, "byggeareal")
    content_col = _find_col(df, "contentscode")
    tid_col = _find_col(df, "tid")

    # Filtrer ut ikke-bolig-kategorien (000 = "Andre bygg enn boligbygg")
    df = df[df[area_col] != "000"]

    # Pivotér på ContentsCode for å få Fullforte og Igangsatte som kolonner.
    # Summen over Byggeareal gir totalt antall boliger per (kommune, år).
    wide = df.pivot_table(
        index=[region_col, tid_col], columns=content_col,
        values="value", aggfunc="sum",
    )
    wide = wide.rename(columns={
        "Fullforte": "fullforte_boliger_kommune",
        "Igangsatte": "igangsatte_boliger_kommune",
    })
    # Sikre at begge kolonner finnes selv om en av ContentsCode-verdiene mangler
    for col in ("fullforte_boliger_kommune", "igangsatte_boliger_kommune"):
        if col not in wide.columns:
            wide[col] = np.nan

    out = wide.reset_index().rename(columns={region_col: "kommune_nr", tid_col: "aar"})
    out["kommune_nr"] = out["kommune_nr"].astype(str).str.zfill(4)
    out["aar"] = out["aar"].astype(int)
    return out[["kommune_nr", "aar",
                "fullforte_boliger_kommune", "igangsatte_boliger_kommune"]]


# ----- Eiendomsskatt (14674) -----

def standardize_eiendomsskatt() -> pd.DataFrame:
    """Generell eiendomsskattesats per kommune × år (promille) fra 14674.

    KOSTRA-tabellen bruker `KOKkommuneregion0000` som region-dimensjon.
    Verdiene er allerede 4-sifret kommunenummer så ingen mapping er nødvendig.
    Kommuner uten eiendomsskatt får NaN — preprosessering kan velge å
    behandle det som 0 eller la det være missing.
    """
    src = RAW_DIR / "eiendomsskatt_14674.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    # KOK-prefiks bryter med det vanlige "region"-mønsteret, men _find_col
    # gjør substring-match så vi finner riktig kolonne uansett
    region_col = next(c for c in df.columns if "kommune" in c.lower())
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "eiendomsskatt_sats_kommune"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "eiendomsskatt_sats_kommune"]].drop_duplicates(
        ["kommune_nr", "aar"]
    )


# ----- Sysselsetting (07984) -----

def standardize_sysselsetting() -> pd.DataFrame:
    """Antall sysselsatte (15-74 år, alle næringer) per (kommune, år)."""
    src = RAW_DIR / "sysselsetting_07984.json"
    if not src.exists():
        return pd.DataFrame()

    df = parse_jsonstat(src)
    region_col = _find_col(df, "region")
    tid_col = _find_col(df, "tid")

    df = df.rename(columns={region_col: "kommune_nr", tid_col: "aar",
                            "value": "antall_sysselsatte"})
    df["kommune_nr"] = df["kommune_nr"].astype(str).str.zfill(4)
    df["aar"] = df["aar"].astype(int)
    return df[["kommune_nr", "aar", "antall_sysselsatte"]].drop_duplicates(
        ["kommune_nr", "aar"]
    )


# ----- Main -----

def main() -> None:
    # Bygg postnummer × år-grunnlaget som alle SSB-delene blir merget på
    print("=== Standardisering: SSB (tidsserie 2002-2024) ===")
    backbone = load_backbone()
    print(f"  Backbone: {len(backbone):,} rader (postnummer × år)")

    # Kjør hver standardize-funksjon og samle resultatene i en dict
    parts = {
        "boliger_per_type": standardize_boliger_per_type(),
        "folkemengde": standardize_folkemengde(),
        "inntekt": standardize_inntekt(),
        "priser": standardize_priser(),
        "byggeaar": standardize_byggeaar(),
        "bruksareal": standardize_bruksareal(),
        "utdanning": standardize_utdanning(),
        "sysselsetting": standardize_sysselsetting(),
        "arbeidsledighet": standardize_arbeidsledighet(),
        "flytting": standardize_flytting(),
        "husholdninger": standardize_husholdninger(),
        "boligbygging": standardize_boligbygging(),
        "eiendomsskatt": standardize_eiendomsskatt(),
    }
    for name, p in parts.items():
        print(f"  {name}: {len(p):,} kommune-år rader, {len(p.columns)} kol")

    # Join alle på (kommune_nr, aar). Hver del er allerede på riktig nivå.
    df = backbone.copy()
    for name, p in parts.items():
        if p.empty:
            continue
        df = df.merge(p, on=["kommune_nr", "aar"], how="left")

    # Skriv samlet output
    out = STD_DIR / "ssb_bolig_demografi.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  ssb_bolig_demografi.parquet: {len(df):,} rader, {len(df.columns)} kolonner")
    # Vis dekning på de viktigste kolonnene
    for col in [
        "antall_boliger", "befolkning", "inntekt_etter_skatt",
        "pris_kvm_alle_kommune", "median_byggeaar_kommune",
        "median_bruksareal_kommune", "andel_utdanning_uh_kort_kommune",
        "antall_sysselsatte",
        "andel_arbeidsledige_kommune", "netto_innflytting_kommune",
        "antall_husholdninger_kommune", "andel_enslige_kommune",
        "fullforte_boliger_kommune", "igangsatte_boliger_kommune",
        "eiendomsskatt_sats_kommune",
    ]:
        if col in df.columns:
            print(f"    {col}: {df[col].notna().mean():.1%}")
    print("=== SSB standardisering ferdig ===\n")


if __name__ == "__main__":
    main()
