"""
Rask inspeksjon av det ferdige datasettet.

Hentet ut som scratchpad for å bla i parquet-filen uten å åpne notebook.
For full validering, kjør `python source/verify.py`.
"""

import zipfile
from pathlib import Path
import geopandas as gpd

# Last datasettet
df = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")

# Skriv ut grunnleggende oversikt: første rader, statistikk, kolonne-info og dimensjon
print(df.head(20))
print(df.describe())
print(df.info())
print(df.shape)