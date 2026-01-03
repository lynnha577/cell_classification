from plot_data import plot_data
from get_features import get_dataframes, get_labels
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
from scipy import signal
import pandas as pd
import os

def get_raw_data(cell_id):
    #get data for cell_id
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    data_set = ctc.get_ephys_data(cell_id)
    sweeps = ctc.get_ephys_sweeps(cell_id)

    #find maxT and data_count to prelocate data
    maxT = 0
    count = 0
    sampling_rates = []
    for i in range(0, len(sweeps)):
        
        if sweeps[i]['stimulus_units'] == 'Volts':
            count += 1
            continue
        sweep_data = data_set.get_sweep(i)
        index_range = sweep_data["index_range"]

        sampling_rate = sweep_data["sampling_rate"] # in Hz
        down_sampling_rate = sampling_rate/100
        sampling_rates.append(sampling_rate)
        t = (index_range[1]+1-index_range[0])/ sampling_rate
        if(t>maxT):
            maxT = t + 1/sampling_rate
    data_count = len(sweeps) - count
    print("these are the sampling rates for data in the cell: ")
    print(np.unique(sampling_rates))

    #store i, v for the sweep into arrays and store metadata
    cell_data_i = np.empty((data_count, int(maxT*down_sampling_rate)+1))
    cell_data_v = np.empty((data_count, int(maxT*down_sampling_rate)+1))
    count = 0
    metadata = []
    for n in range(0, len(sweeps)):
        if sweeps[n]['stimulus_units'] == 'Volts':
            count += 1
            continue
        sweep_data = data_set.get_sweep(n)
        index_range = sweep_data["index_range"]
        i = sweep_data["stimulus"][index_range[0]:index_range[1]+1] # in A
        v = sweep_data["response"][index_range[0]:index_range[1]+1] # in V
        i *= 1e12 # to pA
        i = signal.decimate(i, 10)
        i = signal.decimate(i, 10)

        v *= 1e3 # to mV
        v = signal.decimate(v, 10)
        v = signal.decimate(v, 10)

        sampling_rate = sweep_data["sampling_rate"] # in Hz
        t = (index_range[1]+1-index_range[0])/ sampling_rate

        cell_data_i[n-count, :len(i)] = i
        cell_data_v[n-count, :len(v)] = v
        
        metadata.append([n-count, n, t, down_sampling_rate])
    
    #save data as file
    print(os.getcwd())
    data_folder = f"{os.pardir}/data/"
    np.save(data_folder+ f"{cell_id}_current.npy", cell_data_i)
    np.save(data_folder+ f"{cell_id}_voltage.npy", cell_data_i)

    meta_dataframe = pd.DataFrame(metadata, columns = ['index', 'sweep number', 'time', 'sampling_rate'])
    meta_dataframe.to_csv(data_folder+f"{cell_id}_data.csv")

    return cell_data_i, cell_data_v, meta_dataframe
