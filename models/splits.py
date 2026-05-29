"""
Deler datasettet i train/val/test basert på år.

  train: aar <= 2018
  val:   aar 2019-2020
  test:  aar 2021-2024

En tilfeldig split ville lekket fremtiden inn i treningen siden samme
kommune dukker opp flere år. Tidsbasert split unngår det.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Standard tidsgrenser. Hardkodet for reproduserbarhet — hvis datasettet
# utvides senere vil testsettet automatisk få flere år, ikke endre
# treningssettet.
TRAIN_END = 2018
VAL_START = 2019
VAL_END = 2020
TEST_START = 2021


@dataclass
class TimeSplit:
    """Tre indeks-arrays til X/y for train, val og test."""
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_time_split(years: pd.Series) -> TimeSplit:
    """Del datasettet i train/val/test basert på år."""
    # Tar en Series med år-verdier og returnerer indeksene i X som
    # tilhører hvert sett. Raiser hvis et av settene blir tomt.
    # Konverter til numpy array av int-år for raske sammenligninger
    arr = years.to_numpy()

    # Bygg booleske masker for hvert sett
    train_mask = arr <= TRAIN_END
    val_mask = (arr >= VAL_START) & (arr <= VAL_END)
    test_mask = arr >= TEST_START

    # Konverter til indekser inn i X/y
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]

    # Defensiv sjekk: tomme splits indikerer datakvalitets-problem
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"En av splittene er tom — train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}. Sjekk at "
            f"datasettet dekker årene 2002-2024."
        )

    return TimeSplit(train=train_idx, val=val_idx, test=test_idx)


def time_series_cv_indices(
    years: pd.Series,
    n_splits: int = 3,
    min_train_years: int = 8,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lag tidsserie-folds for hyperparameter-søk. Hver fold trener på flere
    år og validerer på de neste."""
    # Eksempel for n_splits=3:
    #   fold 0: train 2002-2010, val 2011-2012
    #   fold 1: train 2002-2012, val 2013-2014
    #   fold 2: train 2002-2014, val 2015-2016
    arr = years.to_numpy()
    unique_years = sorted(np.unique(arr))
    # Begrens til perioden vi trener på (lukker val/test ut)
    train_period = [y for y in unique_years if y <= TRAIN_END]
    if len(train_period) < min_train_years + 2:
        raise ValueError(
            f"For få år i trenings-perioden ({len(train_period)}) for "
            f"{n_splits} folds. Reduser n_splits eller min_train_years."
        )

    # Beregn val-vindu-størrelse slik at vi får akkurat n_splits folds
    val_size = max(1, (len(train_period) - min_train_years) // n_splits)

    folds = []
    for i in range(n_splits):
        train_end = train_period[min_train_years + i * val_size - 1]
        val_start = train_period[min_train_years + i * val_size]
        val_end = train_period[min(min_train_years + (i + 1) * val_size - 1,
                                   len(train_period) - 1)]
        # Bygg indekser
        train_idx = np.where(arr <= train_end)[0]
        val_idx = np.where((arr >= val_start) & (arr <= val_end))[0]
        if len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))

    return folds
