import numpy as np
import matplotlib.pyplot as plt
from allensdk.core.cell_types_cache import CellTypesCache

def plot_data(cell_specimen_id, sweep_number, tstart, tend):
    """
    plotting the applied current to the cell over time and the response voltage of the cell over time.

    Args: 
        cell_specimen_id (integer): specimen id of the cell
        sweep_number (integer): the part we are looking at
        tstart (integer): starting time in seconds
        tend (integer): ending time in seconds

    Returns:
        axes: a class that contains subplots
    """
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    data_set = ctc.get_ephys_data(cell_specimen_id)
    sweep_data = data_set.get_sweep(sweep_number)

    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1]+1] # in A
    v = sweep_data["response"][0:index_range[1]+1] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV

    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)

    plt.style.use('fivethirtyeight')
    fig, axes = plt.subplots(2, 1, sharex=True)
    if(t[-1] < tstart):
        print("tstart is greater than the end. Plotting from zero instead")
        tstart = 0

    tstart_samples = int (tstart*sampling_rate)
    tend_samples = int (tend*sampling_rate)

    tshort = t[tstart_samples:tend_samples]
    vshort = v[tstart_samples:tend_samples]
    ishort = i[tstart_samples:tend_samples]


    axes[0].plot(tshort, vshort, color='blue', linewidth = 0.5)
    axes[1].plot(tshort, ishort, color='red', linewidth = 0.5)
    #axes[0].plot(t, v, color='blue', linewidth = 0.5)
    #axes[1].plot(t, i, color='red', linewidth = 0.5)
    axes[0].set_ylabel("mV")
    axes[1].set_ylabel("pA")
    axes[1].set_xlabel("seconds")
    plt.show()
    return axes

def main():
    cell_specimen_id = 501799874
    sweep_number = 35
    tstart = 10
    tend = 28
    plot_data(cell_specimen_id, sweep_number, tstart, tend)

if __name__ == "__main__":
    main()
