
import numpy as np
import matplotlib.pyplot as plt
from allensdk.core.cell_types_cache import CellTypesCache
from allensdk.core.swc import Marker
import pprint
from pathlib import Path
import os

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
    t = np.arange(0, len(v))/ sampling_rate


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

    results_folder = Path(f"{os.pardir}/results/rawDataVisualization")
    plt.savefig(results_folder / f"{cell_specimen_id}_{sweep_number}_ephysData.svg")

    plt.show()
    return axes

def plot_morphology(cell_id):
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    morphology = ctc.get_reconstruction(cell_id) 
    markers = ctc.get_reconstruction_markers(cell_id) 

    # the compartment list has all of the nodes in the file

    fig, axes = plt.subplots(1, 2, sharey=True, sharex=True)
    axes[0].set_aspect('equal', 'box')
    axes[1].set_aspect('equal', 'box')

    # Make a line drawing of x-y and y-z views
    for n in morphology.compartment_list:
        for c in morphology.children_of(n):
            axes[0].plot([n['x'], c['x']], [n['y'], c['y']], color='black')
            axes[1].plot([n['z'], c['z']], [n['y'], c['y']], color='black')

    # cut dendrite markers
    dm = [ m for m in markers if m['name'] == Marker.CUT_DENDRITE ]

    axes[0].scatter([m['x'] for m in dm], [m['y'] for m in dm], color='#3333ff')
    axes[1].scatter([m['z'] for m in dm], [m['y'] for m in dm], color='#3333ff')

    # no reconstruction markers
    nm = [ m for m in markers if m['name'] == Marker.NO_RECONSTRUCTION ]

    axes[0].scatter([m['x'] for m in nm], [m['y'] for m in nm], color='#333333')
    axes[1].scatter([m['z'] for m in nm], [m['y'] for m in nm], color='#333333')

    axes[0].set_ylabel('y')
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('z')
    results_folder = Path(f"{os.pardir}/results/rawDataVisualization")
    plt.savefig(results_folder / f"{cell_id}_morphPlot.svg")
    plt.show()

def plot_morphology_3D(cell_id):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from allensdk.core.swc import Marker
    from allensdk.core.cell_types_cache import CellTypesCache
    plt.style.use('fivethirtyeight')

    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    morphology = ctc.get_reconstruction(cell_id) 
    markers = ctc.get_reconstruction_markers(cell_id) 
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw the morphology (neuronal tree)
    for n in morphology.compartment_list:
        for c in morphology.children_of(n):
            ax.plot([n['x'], c['x']], 
                    [n['y'], c['y']], 
                    [n['z'], c['z']], color='black', lw=0.5)

    # 2. Plot Cut Dendrite markers
    dm = [m for m in markers if m['name'] == Marker.CUT_DENDRITE]
    ax.scatter([m['x'] for m in dm], 
            [m['y'] for m in dm], 
            [m['z'] for m in dm], color='#3333ff', s=20, label='Cut Dendrite')

    # 3. Plot No Reconstruction markers
    nm = [m for m in markers if m['name'] == Marker.NO_RECONSTRUCTION]
    ax.scatter([m['x'] for m in nm], 
            [m['y'] for m in nm], 
            [m['z'] for m in nm], color="#000000", s=20, label='No Recon')
    # Formatting

    plt.rcParams['grid.linewidth'] = 4.0  # Increase to 2.0 or higher
    plt.rcParams['grid.color'] = "#464242" # Ensure color is set if needed
    ax.tick_params(axis='x', labelsize=12) 
    ax.tick_params(axis='y', labelsize=12) 
    ax.tick_params(axis='z', labelsize=12)

    ax.set_xlabel('X ($\mu m$)', fontsize = 20)
    ax.set_ylabel('Y ($\mu m$)', fontsize = 20)
    ax.set_zlabel('Z ($\mu m$)', fontsize = 20)
    ax.set_title('3D Neuron Morphology', fontsize = 30)
    plt.legend()
    ax.view_init(elev=30, azim=25)
    results_folder = Path(f"{os.pardir}/results/rawDataVisualization")
    plt.savefig(results_folder / f"{cell_id}_morphPlot3D.svg")
    plt.show()

def main():
    cell_specimen_id = 501799874
    sweep_number = 35
    tstart = 10
    tend = 28
    plot_data(cell_specimen_id, sweep_number, tstart, tend)

if __name__ == "__main__":
    main()
