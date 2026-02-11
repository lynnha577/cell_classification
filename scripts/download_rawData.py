from plot_data import plot_data
from get_features import get_dataframes, get_labels
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
from scipy import signal
import pandas as pd
import os

cell_id = 323865917

labels = get_labels()
epys_features = get_dataframes()
ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
data_set = ctc.get_ephys_data(323865917)
sweeps = ctc.get_ephys_sweeps(323865917)
plot_data(323865917, 35, 0, 10)

from autoencoder_raw_data import get_raw_data
for idx in epys_features.index:
    cell_data_i, cell_data_v, metadata = get_raw_data(idx, data_folder=f"{os.pardir}/cell_classification/data/")

    print(cell_data_i)
    print(cell_data_v)
    print(metadata)