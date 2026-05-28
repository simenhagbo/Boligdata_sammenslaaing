import zipfile
from pathlib import Path
import geopandas as gpd

df = gpd.read_parquet("data/processed_data/final/boligdata_final.parquet")
print(df.head(20))
print(df.describe())
print(df.info())
print(df.shape)