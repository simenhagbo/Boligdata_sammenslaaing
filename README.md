# Boligdata sammenslåing

Sommerprosjekt der jeg slår sammen fire offentlige norske datakilder til ett datasett
egnet for å trene en boligprismodell. Nøkkelen er postnummer — én rad per postnummer i Norge.

## Hvorfor

Jeg ville lære meg en skikkelig dataengineering-flyt: hente fra ulike API-er, vaske
hvert datasett for seg, og så sy det hele sammen med kvalitetssjekker. Boligdata
er et fint case fordi kildene er åpne, men formatene er helt forskjellige
(WFS/GML, JSON-stat, Excel, TSV...).

## Kildene

| Kilde | Hva derfra | Hvordan |
|---|---|---|
| Kartverket (GeoNorge) | Postnummer-polygoner | WFS, GML 3.2.1 |
| Bring | Postnummer → kommune | TSV-fil med hele registeret |
| SSB | Boliger, befolkning, inntekt | JSON-stat API (tabell 06265, 07459, 12558) |
| Eiendom Norge | Pris per m² | Excel-fil fra nettsiden |
| Matrikkelen-Bygningspunkt | Bygningstype per kommune | WFS (opt-in, treg) |

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
```

## Kom i gang

```bash
pip install -r requirements.txt
python source/pipeline.py
```

Trenger Python 3.10 eller nyere. Første kjøring tar typisk under et minutt
(Eiendom Norge må evt. lastes ned manuelt — se under). Hvis rådataene allerede
ligger på disk er pipelinen ferdig på et par sekunder.

Vil du kjøre én fase om gangen:

```bash
python source/pipeline.py --collect       # bare hent rådata
python source/pipeline.py --standardize   # bare vask
python source/pipeline.py --merge         # bare slå sammen
```

### Matrikkelen er opt-in

```bash
python source/pipeline.py --include-matrikkelen
```

Bygningspunkt-WFS-en gir bare bygningstype og kommunenummer (ikke areal eller
byggeår — det krever lisens). Og siden CQL_FILTER blir ignorert må man paginere
gjennom alle 4,4 millioner bygg i Norge, så det tar omtrent en time. Vi får
tilsvarende informasjon gratis fra SSB tabell 06265, så jeg har gjort dette
til et flagg du må slå på selv hvis du faktisk vil ha rådataene.

### Les datasettet

```python
import geopandas as gpd
df = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")
print(df.columns.tolist())
```

## Kolonner

- `postnummer`, `kommune_nr`, `poststedsnavn`, `kommunenavn` — fra Kartverket/Bring
- `geometry` — postnummerets polygon (EPSG:4326)
- `antall_boliger`, `modal_boligtype_kommune` — fra SSB 06265
- `befolkning` — fra SSB 07459
- `inntekt_etter_skatt` — fra SSB 12558
- `median_pris_m2` — fra Eiendom Norge (hvis Excel-filen er lastet ned)
- `antall_bygninger_kommune`, `modal_bygningstype_kommune` — fra Matrikkelen (kun med `--include-matrikkelen`)
- `data_kvalitet_flagg` — 1 hvis raden mangler over halvparten av verdiene

## Ting jeg har lært underveis

**Excel-filen fra Eiendom Norge er en pest.** Lenken endrer seg månedlig og
Cloudflare blokkerer ofte direkte nedlasting. Hvis collect-scriptet feiler med
404 eller 403, last ned Excel-filen manuelt fra
[eiendomnorge.no/boligprisstatistikk](https://eiendomnorge.no/boligprisstatistikk/)
og legg den i `data/raw_data/eiendom_norge/prisstatistikk.xlsx`. Pipelinen
fortsetter uten Eiendom Norge-data hvis filen mangler — du får bare ikke
`median_pris_m2`-kolonnen.

**SSB er kommunenivå.** Alle postnummer i samme kommune får identiske SSB-verdier.
Det er en pragmatisk forenkling — alternativet ville vært å oppfinne data på
postnummernivå, og det er verre.

**WFS-tjenestene til Kartverket støtter ikke GeoJSON for disse datasettene.**
Du må be om GML 3.2.1, og `srsName` må være på URN-format
(`urn:ogc:def:crs:EPSG::25833`), ikke kortformen `EPSG:25833`. Brukte en stund
på å finne ut av det.

**Matrikkelen-Bygningspunkt-WFS ignorerer CQL_FILTER og BBOX.** Det er ikke
dokumentert noe sted jeg fant. Konsekvensen er at man ikke kan hente per
fylke — alle forespørsler returnerer fra start av datasettet. Derfor opt-in.

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
from sklearn.model_selection import train_test_split

gdf = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")
df = gdf[gdf["data_kvalitet_flagg"] == 0].dropna(subset=["median_pris_m2"])

X = df[["antall_boliger", "befolkning", "inntekt_etter_skatt"]].fillna(0)
y = df["median_pris_m2"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_tr, y_tr)
print(f"R² = {model.score(X_te, y_te):.3f}")
```

## Lisens

Koden er MIT — se [LICENSE](LICENSE). Det betyr du kan bruke, endre og dele
fritt så lenge du beholder copyright-linjen.

Datakildene har sine egne lisenser. Du redistribuerer ikke dataene fra dette
repoet (pipelinen henter dem ved kjøring), men hvis du bruker outputen til noe
publisert bør du kreditere kildene:

- Kartverket / GeoNorge — NLOD 2.0 — *"Inneholder data fra Kartverket"*
- Bring — fri bruk med attribusjon — *"Postnummerregister: Bring"*
- SSB — CC BY 4.0 / NLOD 2.0 — *"Kilde: Statistisk sentralbyrå"*
- Eiendom Norge — offentlig statistikk — *"Kilde: Eiendom Norge"*
- Matrikkelen — NLOD 2.0 — *"Inneholder data fra Kartverket"*
