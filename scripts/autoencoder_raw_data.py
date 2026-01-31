from plot_data import plot_data
from get_features import get_dataframes, get_labels
from allensdk.core.cell_types_cache import CellTypesCache
import numpy as np
from scipy import signal
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt


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
        down_sampling_rate = sampling_rate
        if(sampling_rate == 200000):
            down_sampling_rate /= 400
        elif(sampling_rate == 50000):
            down_sampling_rate /= 100

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
        v *= 1e3 # to mV

        if(sampling_rate == 200000):
            i = signal.decimate(i, 10)
            i = signal.decimate(i, 10)
            i = signal.decimate(i, 4)

            v = signal.decimate(v, 10)
            v = signal.decimate(v, 10)
            v = signal.decimate(v, 4)
        elif(sampling_rate == 50000):
            i = signal.decimate(i, 10)
            i = signal.decimate(i, 10)

            v = signal.decimate(v, 10)
            v = signal.decimate(v, 10)

        lowcut = 0.01 # Lower cutoff frequency in Hz
        highcut = 150.0 # Upper cutoff frequency in Hz
        order = 4 # Filter order
        print(down_sampling_rate)
        # 'bandpass' type, 'sos' output recommended for stability
        sos = signal.butter(order, [lowcut, highcut], btype='bandpass', fs = down_sampling_rate, output='sos')

        # 3. Apply the filter to the signal
        filtered_x = signal.sosfiltfilt(sos, v) # Use filtfilt for zero phase shift

        sampling_rate = sweep_data["sampling_rate"] # in Hz
        t = (index_range[1]+1-index_range[0])/ sampling_rate
        if(n==35):
            times = np.arange(v.shape[0])/down_sampling_rate
            plt.plot(times, v, color = "blue")
            plt.plot(times, filtered_x, color = 'red')
            plt.show()
        cell_data_i[n-count, :len(i)] = i
        cell_data_v[n-count, :len(v)] = v
        
        metadata.append([n-count, n, t, down_sampling_rate])
    
    #save data as file
    print(os.getcwd())
    data_folder = f"{os.pardir}/data/"
    np.save(data_folder+ f"{cell_id}_current.npy", cell_data_i)
    np.save(data_folder+ f"{cell_id}_voltage_filtered.npy", cell_data_i)

    meta_dataframe = pd.DataFrame(metadata, columns = ['index', 'sweep number', 'time', 'sampling_rate'])
    meta_dataframe.to_csv(data_folder+f"{cell_id}_data.csv")

    return cell_data_i, cell_data_v, meta_dataframe

def sample_batch_training_data(cell_id, input_size, num_samples, output_size):
    """
    Docstring for sample_batch_training_data: Creates a batch of inputs and outputs from
    the cell's voltage based on the number of inputs and outputs.
    
    :param cell_id: the id number of the cell used
    :param input_size: the amount of inputs 
    :param num_samples: the amount of samples to use
    :param output_size: the number of numbers to be predicted
    """
    voltage_data = np.load(f"{os.pardir}/data/{cell_id}_voltage.npy")
    #set the size of input to 20 and the number of samples to 100

    # create an empty list for input and output
    inputs = np.empty((input_size, num_samples))
    outputs = np.empty((output_size, num_samples))

    #get a random sweep number and random start index
    metadata = pd.read_csv(f'../data/{cell_id}_data.csv')

    rng = np.random.default_rng()
    random_sweep_number = rng.integers(low = 0, high = voltage_data.shape[0], size = num_samples)

    sampling_rate = list(metadata.loc[random_sweep_number, 'sampling_rate'])
    time = list(metadata.loc[random_sweep_number, 'time'])

    mean_voltage =np.expand_dims(np.mean(voltage_data, axis=1), axis=1)
    print(np.shape(mean_voltage))
    std = np.std(voltage_data, axis = 1)
    std = np.expand_dims(std, axis =1)
    voltage_data -= mean_voltage
    voltage_data/=std
    print(np.std(voltage_data, axis = 1))
    print(np.mean(voltage_data, axis = 1))

    # from current data, get 20 input numbers from the random starting index at a random sweep and also get the next number.
    for i in range(num_samples):
        random_sample_idx = rng.integers(low = 0, high = int(sampling_rate[i]*time[i])-input_size-output_size, size = 1)[0]
        inputs[:, i] = voltage_data[random_sweep_number[i], random_sample_idx:random_sample_idx+input_size]
        outputs[:, i] = voltage_data[random_sweep_number[i], random_sample_idx+input_size:random_sample_idx+input_size+output_size]
    


    input_tensor = torch.from_numpy(inputs).float()
    output_tensor = torch.from_numpy(outputs).float()

    

    return input_tensor, output_tensor
