# Boligdata sammenslåing

> **NB:** Dette repoet er produsert sammen med
> Claude Code. Alt innhold er gjennomgått, men det kan fortsatt
> inneholde feil eller logiske brister. Ved bruk av datasettet,
> verifiser det selv dersom du baserer noe viktig på det.

Ett prosjekt der jeg henter data fra flere offentlige norske datakilder til
å produsere ett datasett med boligdata egnet for ML-trening. Prosessen ved innhenting, datarensing og kombinering av dataene er gjort sammen med Claude Code. Hver rad representerer en
kombinasjon av postnummer og år, og datasettet dekker 2002-2024.

## Hvorfor

Jeg ønsket å lære meg en skikkelig dataengineering-flyt som å hente data fra ulike API-er,
vaske hvert datasett for seg, og så sy det hele sammen med kvalitetssjekker.
Boligdata er et fint case fordi kildene er åpne, men formatene er helt
forskjellige (WFS/GML, JSON-stat, TSV) og kommunesammenslåinger over tid gjør
joinene mer interessante enn de først ser ut.

## Kildene

| Kilde | Hva derfra | Hvordan |
|---|---|---|
| Kartverket (GeoNorge) | Postnummer-polygoner | WFS, GML 3.2.1 |
| Bring | Postnummer → kommune | TSV med hele registeret |
| SSB 06035 | Kvm-pris og omsetninger per boligtype | JSON-stat |
| SSB 06265 | Antall boliger per bygningstype | JSON-stat |
| SSB 06266 | Byggeår-fordeling | JSON-stat (chunked) |
| SSB 06513 | Bruksareal-fordeling | JSON-stat (chunked) |
| SSB 06913 | Folkemengde | JSON-stat |
| SSB 12558 | Inntekt etter skatt (median) | JSON-stat |
| SSB 09429 | Utdanningsnivå | JSON-stat |
| SSB 07984 | Sysselsetting | JSON-stat |
| SSB 10540 | Registrerte arbeidsledige (1999-2020) | JSON-stat (november-tall per år) |
| SSB 09588 | Nettoinnflytting per kommune | JSON-stat |
| SSB 06070 | Antall husholdninger + andel enslige | JSON-stat |
| SSB 05940 | Fullførte og igangsatte boliger | JSON-stat (chunked) |
| SSB 14674 | Generell eiendomsskattesats (promille) | JSON-stat (KOSTRA) |
| SSB 03013 | Konsumprisindeks (KPI, 2015=100) | JSON-stat (månedlig → årssnitt) |
| SSB 10748 | Boliglånsrente, husholdninger, totalt | JSON-stat (månedlig → årssnitt, fra 2014) |
| Norges Bank | Styringsrente årssnitt | SDMX-JSON (åpen, ingen auth) |
| Entur | Avstand til nærmeste togstasjon | Journey Planner GraphQL (bbox-paginert) |
| MET Norge | Klima-normaler 1991-2020 (snitt-temp + nedbør) | Hardkodet fra MET-rapport, mappet via nærmeste storby |
| MET Frost-API | Klima-normaler per værstasjon | Frost REST-API, mappet til nærmeste stasjon (opt-in, krever gratis klient-ID) |
| Beregnet fra geometri | Areal, sentroide, avstand til storby | UTM 33N reprojection |
| Matrikkelen-Bygningspunkt | Bygningstype per kommune | WFS (opt-in, treg) |

## Om datakvalitet

**Tidsserie.** Datasettet er 3378 postnummer × 23 år = 77 694 rader. SSB
publiserer data på kommunenivå — alle postnummer i samme kommune deler de samme
SSB-verdiene for et gitt år. Geo-features (areal, sentroide, avstand) er
postnummer-spesifikke og statiske over år.

**Eldre år har dårligere dekning.** Norge har gjennomgått flere
kommunesammenslåinger (særlig 2020), og SSB rapporterer eldre år med datidens
kommunekoder mens vi mapper til 2024-koder. Resultatet er at pris- og
inntektsdekning typisk er rundt 30% før 2020 og 95% i 2024. Det går an å fikse
ved å hente SSBs kommune-historikk, men det er et større prosjekt.

**Prisene er kommunenivå, ikke postnummer-nivå.** Åpen SSB-data viser kun kommune nivå. Eiendom Norges abonnementsprodukter tilbyr by/region-nivå data (også
grovere enn kommune for små områder), Finn.no har postnummer, men kan ikke
skrapes lovlig, og SSB Microdata krever forskningssøknad. For ML betyr dette:
**splitt train/test på `kommune_nr`, ikke postnummer** — ellers lekker prisen
via kommunen og R2 blir kunstig høy.

**SSB sensurerer små kommuner.** For 06035 (priser) settes både kvm-pris og
omsetninger til NaN/0 hvis kommunen har for få salg i et år. Vi har håndtert
0-tilfeller eksplisitt slik at du ikke får falske "0 kr/m²"-rader.

## Mappestruktur

```
data/
  raw_data/         hentet fra kildene, røres ikke
  processed_data/
    standardized/   én Parquet per kilde
    final/          ferdig sammenslått datasett
source/
  collect/          fase 1: henter rådata
  standardize/      fase 2: vasker og normaliserer
  merge/            fase 3: slår sammen + kvalitetskontroll
  pipeline.py       kjører alt
  verify.py         spot-tester datasettets integritet
  export_csv.py     konverterer Parquet til CSV
```

## Kom i gang

```bash
pip install -r requirements.txt
python source/pipeline.py
```

Trenger Python 3.10 eller nyere. Første kjøring tar 1-2 minutter (mest tid på
SSB-fetching). Hvis rådataene allerede ligger på disk er pipelinen ferdig på
under 30 sekunder.

Enkeltfaser:

```bash
python source/pipeline.py --collect       # bare hent rådata
python source/pipeline.py --standardize   # bare vask
python source/pipeline.py --merge         # bare slå sammen
```

### Mer presise klima-features (valgfritt)

Dagens klima-features (`temperatur_normal`, `nedbor_normal_mm`) bruker
verdier for nærmeste storby som proxy. For mer presise tall fra MET Norges
Frost-API:

1. Registrer en gratis konto på
   [frost.met.no/auth/requestCredentials](https://frost.met.no/auth/requestCredentials.html)
2. Kopier `.env.example` til `.env` og lim inn klient-ID-en din
3. Kjør pipelinen igjen — den henter klima-normaler (1991-2020) for hver
   norsk værstasjon og mapper hvert postnummer til nærmeste stasjon

Resultatet er fire nye kolonner: `temperatur_normal_frost`,
`nedbor_normal_frost_mm`, `met_stasjon_id`, `met_stasjon_avstand_km`. De
proxy-baserte kolonnene beholdes som fallback for kompatibilitet.

Hvis du ikke setter en ID, hopper pipelinen Frost-fasen gracefully over —
ingenting annet endres.

### Verifiser datasettet

```bash
python source/verify.py
```

Kjører 14 sjekker: ingen duplikater, andeler summerer til 1, SSB-verdier er
konsistente innen kommune-år, geo-features er statiske over år, og spot-tester
mot kjente postnummer (Oslo, Bergen, Trondheim, Stavanger).

### Eksporter til CSV

```bash
python source/export_csv.py                 # uten geometri, åpnes i Excel
python source/export_csv.py --med-geometri  # tar med polygon som WKT

# Eller kjør sammen med pipelinen:
python source/pipeline.py --export-csv
python source/pipeline.py --merge --export-csv  # bare merge + CSV
```

CSV-en er ca 25 MB pga tidsserie-formatet (77 694 rader). Excel kan åpne den,
men det går raskere å bruke Parquet direkte fra Python når du jobber med
dataene. CSV er mest nyttig for deling og enkel inspeksjon i Excel.

### Matrikkelen er opt-in

```bash
python source/pipeline.py --include-matrikkelen
```

Bygningspunkt-WFS-en gir bare bygningstype og kommunenummer (ikke areal eller
byggeår — det krever lisens). Og siden CQL_FILTER blir ignorert må man paginere
gjennom alle 4,4 millioner bygg i Norge.

### Les datasettet

```python
import geopandas as gpd
df = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")
df_2024 = df[df["aar"] == 2024]  # filtrer til ett år hvis ønskelig
```

## Kolonner

Totalt 66 kolonner (70 med MET Frost aktivert). Hver rad er én kombinasjon av postnummer og kommune.

**Identifikatorer (6):** `postnummer`, `aar`, `geometry`, `kommune_nr`,
`poststedsnavn`, `kommunenavn`

**Geografi, statisk per postnummer (9):** `areal_km2`, `sentroide_lat`,
`sentroide_lon`, `avstand_oslo_km`, `avstand_naermeste_storby_km`,
`naermeste_storby`, `temperatur_normal` (årlig snitt °C fra nærmeste
MET-stasjon), `nedbor_normal_mm` (årsnedbør i mm), `avstand_togstasjon_km`
(luftlinje til nærmeste togstasjon fra Entur)

**Boliger og typer, per kommune-år (7):** `antall_boliger`,
`modal_boligtype_kommune`, og fem `andel_bygntype_0X_kommune`-kolonner
(enebolig, tomannsbolig, rekkehus, blokk, bofellesskap)

**Priser, per kommune-år (5):** `pris_kvm_enebolig_kommune`,
`pris_kvm_smaahus_kommune`, `pris_kvm_blokk_kommune`, `pris_kvm_alle_kommune`
(vektet snitt over typene), `antall_omsetninger_kommune`

**Byggeår, per kommune-år (6):** fem `andel_byggeaar_*_kommune`-perioder
(for1946, 1946_1970, 1971_1990, 1991_2010, etter2010) + `median_byggeaar_kommune`

**Bruksareal, per kommune-år (6):** fem `andel_areal_*_kommune`-grupper
(under60, 60_99, 100_159, 160_249, over250) + `median_bruksareal_kommune`

**Utdanning, per kommune-år (6):** seks `andel_utdanning_*_kommune`
(grunnskole, videregaaende, fagskole, uh_kort, uh_lang, uoppgitt)

**Demografi og økonomi (4):** `befolkning`, `inntekt_etter_skatt` (median),
`antall_sysselsatte`, `befolkningstetthet` (kommunens befolkning / kommunens
totale areal)

**Arbeid og flytting (2):** `andel_arbeidsledige_kommune` (NB: data slutter
2020 — SSB-statistikken ble lagt ned), `netto_innflytting_kommune`

**Husholdninger (2):** `antall_husholdninger_kommune`, `andel_enslige_kommune`
(én-person-husholdninger)

**Boligbygging (2):** `fullforte_boliger_kommune`, `igangsatte_boliger_kommune`
(årlig flyt, ikke beholdning — `antall_boliger` er beholdningen)

**Skatt (1):** `eiendomsskatt_sats_kommune` (generell sats i promille,
NaN for kommuner uten eiendomsskatt)

**Makro, nasjonalt per år (4):** `kpi_indeks` (SSB 03013, 2015=100),
`kpi_endring_pct` (årlig inflasjon), `styringsrente` (Norges Bank årssnitt),
`boliglaansrente` (SSB 10748 årssnitt, NaN før 2014)

**Investerings-features, per kommune-år (5):**
`prisstigning_nominal_pct` (årlig endring i kvm-pris),
`prisstigning_real_pct` (nominell minus KPI-inflasjon),
`cagr_5aar_real_pct` (5-års compound annual growth rate, real),
`volatilitet_5aar` (std.avvik på log-returns siste 5 år),
`sharpe_5aar` ((CAGR − styringsrente) / volatilitet)

**Matrikkelen (2, opt-in):** `antall_bygninger_kommune`,
`modal_bygningstype_kommune`

**MET Frost (4, opt-in):** `temperatur_normal_frost`,
`nedbor_normal_frost_mm`, `met_stasjon_id`, `met_stasjon_avstand_km` —
mer presise klima-normaler basert på faktisk værstasjon i stedet for
storby-proxy. Krever gratis Frost-klient-ID (se "Mer presise klima-features").

**Kvalitet (1):** `data_kvalitet_flagg` — 1 hvis raden mangler over halvparten
av verdiene (typisk eldre år eller små kommuner)

### Om investerings-features

`prisstigning_*` og `volatilitet_5aar` er **lookahead-sikre**: en rullende
beregning ved år T bruker bare observasjoner t.o.m. år T, så featuren kan
trygt brukes til å predikere år T+1 uten å lekke fremtidig informasjon.
For å trene en modell som predikerer år T fra features ved T-1, skift
kolonnene én år når du splitter trening-/testdata.

`sharpe_5aar` er en kapital-only Sharpe-aktig ratio — den måler
risikojustert real prisavkastning over 5 år versus styringsrente. Den
inkluderer **ikke** leieinntekt, drift, eller skatt, så den gir et
"hvor var bolig en god verdiappresiering-investering?"-svar, ikke full ROI.
Datasettet kan utvides til full ROI senere ved å legge til antakelser om
leieyield (se LICENSE-fil for fremtidig arbeid).

## Ting som var lærerikt

**WFS-tjenestene til Kartverket støtter ikke GeoJSON.** Du må be om GML 3.2.1
og `srsName` må være på URN-format (`urn:ogc:def:crs:EPSG::25833`), ikke
kortformen `EPSG:25833`. Brukte litt tid på å finne ut av det.

**Matrikkelen-Bygningspunkt-WFS ignorerer CQL_FILTER og BBOX.** Det er ikke
dokumentert noe sted jeg fant — man oppdager det ved at filtrene ikke har
noen effekt. Derfor er Matrikkelen-fetchen opt-in.

**SSB chunker man bare når man må.** API-et har en grense på ~800k
datapunkter per spørring, og 06266/06513 går over hvis du tar 23 år på én
gang. Splitter på Tid i biter á 5 år for å holde seg under grensa.

**`pivot_table(aggfunc="sum")` konverterer NaN til 0.** Det er den slags ting
som ikke synes før du oppdager kommuner med "0 kr/m²" i datasettet. Endte opp
med å bruke `merge` på filtrerte deler i stedet for å bevare NaN.

## Robusthet

Bare Kartverket og Bring er kritiske — uten postnummer-mappingen kan ikke
datasettet bygges, så hvis den feiler stopper pipelinen. Alt det andre
fortsetter ved feil, du får bare færre kolonner i output. Hver fase er
idempotent, og cachede filer valideres (JSON-parses / GML lukker korrekt)
før de godkjennes — så en avbrutt nedlasting blir hentet på nytt i stedet
for å passere som "ferdig" og krasje senere.

HTTP-kallene har retry med exponential backoff på 5xx/429/timeout og en
hard øvre grense på respons-størrelsen (500 MB standard, satt lavere per
kilde). Det holder en feilkonfigurert eller midlertidig nedlagt kilde fra
å spise alt RAM-et.

### Kjent skala-grense

Pipelinen er testet med rundt 77 000 rader (kommune × år, 2002-2024). Hvis du
trenger adresse-nivå (Rundt 58 millioner rader) holder ikke den nåværende
Pandas-baserte implementasjonen — særlig `parse_jsonstat` (bygger Python-
liste før DataFrame) og kryss-produkt-merge må refaktoreres til Polars
eller DuckDB.

Koden er ment for lokal kjøring. Hvis du senere skal eksponere den som
en tjeneste, vil du i tillegg trenge: sentralisert logging, dependency-
pinning, input-validering på CLI-args, og rate-limiting på utgående kall.

## ML-eksempel

```python
import geopandas as gpd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

gdf = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")

# Bruk bare nyere år der dekningen er god
df = gdf[(gdf["aar"] >= 2020) & (gdf["data_kvalitet_flagg"] == 0)]
df = df.dropna(subset=["pris_kvm_alle_kommune"])

features = [
    "befolkning", "inntekt_etter_skatt", "befolkningstetthet",
    "avstand_oslo_km", "avstand_naermeste_storby_km",
    "median_byggeaar_kommune", "median_bruksareal_kommune",
    "andel_bygntype_04_kommune",        # andel blokkleiligheter
    "andel_utdanning_uh_lang_kommune",  # andel med lang UH
    "antall_omsetninger_kommune",
    "aar",
]
X = df[features].fillna(df[features].median())
y = df["pris_kvm_alle_kommune"]

# Splitt på kommune, ikke postnummer — ellers lekker prisen via kommunen
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["kommune_nr"]))

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X.iloc[train_idx], y.iloc[train_idx])
print(f"R² = {model.score(X.iloc[test_idx], y.iloc[test_idx]):.3f}")
```

## Lisens

Koden er MIT — se [LICENSE](LICENSE). Bruk og endre fritt så lenge du beholder
copyright-linjen.

Datakildene har sine egne lisenser. Repoet redistribuerer ikke dataene
(pipelinen henter dem ved kjøring), men hvis du publiserer noe basert på
outputen bør du kreditere kildene:

- Kartverket / GeoNorge — NLOD 2.0 — *"Inneholder data fra Kartverket"*
- Bring — fri bruk med attribusjon — *"Postnummerregister: Bring"*
- SSB — CC BY 4.0 / NLOD 2.0 — *"Kilde: Statistisk sentralbyrå"*
- Norges Bank — NLOD 2.0 — *"Styringsrente: Norges Bank"*
- Entur — NLOD 2.0 — *"Inneholder data fra Entur"*
- Meteorologisk institutt — CC BY 4.0 — *"Klima-normaler fra MET Norge"*
- Matrikkelen — NLOD 2.0 — *"Inneholder data fra Kartverket"*



