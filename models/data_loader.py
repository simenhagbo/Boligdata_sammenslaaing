"""
Sentralisert datatilgang for ML-modellene.

`boligdata_final.parquet` er eneste sannhetskilde — den har alle avledede
kolonner (befolkningstetthet, makrodata, investerings-features, kvalitetsflagg)
som beregnes i merge-fasen. Vi leser den ett sted slik at alle modeller
bruker nøyaktig samme data, og ingen merge-logikk dupliseres her.

Datasettets to grain-nivåer (begge bygges i source/standardize):
  - kommune × år:   ssb_bolig_demografi.parquet (SSB-data)
  - postnummer:     postnummer_geofeatures / _entur / _met_frost / _geometri

Disse slås sammen til postnummer × år i merge-fasen. Trenger du å inspisere
et enkelt grain-nivå, bruk load_standardized().
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

# Stier til de ferdige og standardiserte filene
_PROCESSED = Path(__file__).parents[1] / "data" / "processed_data"
FINAL_PATH = _PROCESSED / "final" / "boligdata_final.parquet"
STD_DIR = _PROCESSED / "standardized"


def load_dataset() -> gpd.GeoDataFrame:
    """Les det ferdige ML-datasettet (postnummer × år) fra parquet."""
    # Avbryt med tydelig melding hvis pipelinen ikke er kjørt ennå
    if not FINAL_PATH.exists():
        raise FileNotFoundError(
            f"{FINAL_PATH} finnes ikke. Kjør 'python source/pipeline.py' først."
        )
    return gpd.read_parquet(FINAL_PATH)


def load_standardized(name: str) -> pd.DataFrame:
    """Les én standardisert mellomfil ved navn (uten .parquet-suffiks).

    Nyttig for inspeksjon av et enkelt grain-nivå, f.eks.
    load_standardized("ssb_bolig_demografi") for kommune × år-dataene.
    """
    path = STD_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} finnes ikke — kjør pipelinen først")
    return pd.read_parquet(path)
