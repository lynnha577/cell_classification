from allensdk.core.cell_types_cache import CellTypesCache
import pandas as pd

def get_dataframes(recon_bool=True):
    """
    gets cells that requires recontruction and matches specimen id and returns the cells ephys features

    Args: 
        recon_bool (bool), optional: determines whether cell should require reconstruction

    Returns:
        Dataframe: dataframe of cells and their features
    """
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    # cells with reconstructions
    cells_w_recon = ctc.get_cells(require_reconstruction = recon_bool)

    # download all electrophysiology features for all cells
    features_ephys = ctc.get_ephys_features()
    ef_df = pd.DataFrame(features_ephys)

    all_ids = [item["id"] for item in cells_w_recon]
    ef_with_recon_df = ef_df[ef_df["specimen_id"].isin(all_ids)]
    return ef_with_recon_df

