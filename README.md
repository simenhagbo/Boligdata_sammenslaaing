# Boligdata sammenslåing

> **NB:** Mye av koden og dokumentasjonen i dette repoet er skrevet med hjelp av
> KI-verktøy (Claude/Anthropic). Jeg har gått igjennom alt, men det kan fortsatt
> finnes feil eller logiske brister. Bruk gjerne datasettet, men ikke uten å
> verifisere det selv hvis du baserer noe viktig på det.

Sommerprosjekt der jeg slår sammen flere offentlige norske datakilder til ett
boligdata-datasett egnet for ML-trening. Hver rad er én (postnummer, år)-
kombinasjon, og datasettet dekker 2002-2024.

## Hvorfor

Jeg ville lære meg en skikkelig dataengineering-flyt: hente fra ulike API-er,
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
| Beregnet fra geometri | Areal, sentroide, avstand til storby | UTM 33N reprojection |
| Matrikkelen-Bygningspunkt | Bygningstype per kommune | WFS (opt-in, treg) |

## Om datakvalitet

**Tidsserie.** Datasettet er 3378 postnummer × 23 år = ~77 700 rader. SSB
publiserer på kommunenivå — alle postnummer i samme kommune deler de samme
SSB-verdiene for et gitt år. Geo-features (areal, sentroide, avstand) er
postnummer-spesifikke og statiske over år.

**Eldre år har dårligere dekning.** Norge har gjennomgått flere
kommunesammenslåinger (særlig 2020), og SSB rapporterer eldre år med datidens
kommunekoder mens vi mapper til 2024-koder. Resultatet er at pris- og
inntektsdekning typisk er ~30% før 2020 og ~95% i 2024. Det går an å fikse
ved å hente SSBs kommune-historikk, men det er et større prosjekt.

**Prisene er kommunenivå, ikke postnummer-nivå.** Open SSB-data går bare så
langt. Eiendom Norges abonnementsprodukter har by/region-nivå data (også
grovere enn kommune for små områder), Finn.no har postnummer men kan ikke
scrapes lovlig, og SSB Microdata krever forskningssøknad. For ML betyr dette:
**splitt train/test på `kommune_nr`, ikke postnummer** — ellers lekker prisen
via kommunen og R² blir kunstig høy.

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
```

### Matrikkelen er opt-in

```bash
python source/pipeline.py --include-matrikkelen
```

Bygningspunkt-WFS-en gir bare bygningstype og kommunenummer (ikke areal eller
byggeår — det krever lisens). Og siden CQL_FILTER blir ignorert må man paginere
gjennom alle 4,4 millioner bygg i Norge, så det tar omtrent en time.

### Les datasettet

```python
import geopandas as gpd
df = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")
df_2024 = df[df["aar"] == 2024]  # filtrer til ett år hvis ønskelig
```

## Kolonner

Totalt 47 kolonner. Hver rad er én (postnummer, år)-kombinasjon.

**Identifikatorer (6):** `postnummer`, `aar`, `geometry`, `kommune_nr`,
`poststedsnavn`, `kommunenavn`

**Geografi, statisk per postnummer (6):** `areal_km2`, `sentroide_lat`,
`sentroide_lon`, `avstand_oslo_km`, `avstand_naermeste_storby_km`,
`naermeste_storby`

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

**Matrikkelen (2, opt-in):** `antall_bygninger_kommune`,
`modal_bygningstype_kommune`

**Kvalitet (1):** `data_kvalitet_flagg` — 1 hvis raden mangler over halvparten
av verdiene (typisk eldre år eller små kommuner)

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
idempotent (filer som finnes hoppes over), så du kan trygt avbryte og
fortsette senere.

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
- Matrikkelen — NLOD 2.0 — *"Inneholder data fra Kartverket"*

## Etisk om publisering

Datakildene er åpne, men de aggregerte dataene per kommune kan i prinsippet
identifisere svært små kommuner med få omsetninger. SSB håndterer dette ved
å sensurere de mest sårbare verdiene (du ser dem som NaN i datasettet). Hvis
du publiserer modeller eller analyser, ikke prøv å rekonstruere de sensurerte
verdiene — det er bevisst skjult for å beskytte personvern.
