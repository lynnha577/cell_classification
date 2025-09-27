from allensdk.core.cell_types_cache import CellTypesCache
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    #ef_with_recon_df = ef_df[ef_df["specimen_id"].isin(all_ids)]
    ef_with_recon_df = ef_df[ef_df["specimen_id"].isin(all_ids)]
    ef_with_recon_df.set_index("specimen_id", inplace=True)
    return ef_with_recon_df
    

def get_labels():
    """
    makes a new dataframe with the Y cell features that is filtered 

    Returns:
        Dataframe: of the cells ids and their corresponding labels
    """
    ctc = CellTypesCache(manifest_file='cell_types/manifest.json')
    cells = ctc.get_cells(require_reconstruction = True)

    label_features = ["id", "species", "structure_layer_name", "structure_area_abbrev", "dendrite_type", "donor_id"]
    all_ids = [[item[feature] for feature in label_features] for item in cells]
    dataframe_labels = pd.DataFrame(all_ids, columns = label_features)
    filtered_dataframe_labels = dataframe_labels[dataframe_labels["dendrite_type"].isin(["spiny", "aspiny"])]
    filtered_dataframe_labels.set_index("id", inplace=True)

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le1.fit(filtered_dataframe_labels['dendrite_type'])
    # print(le1.classes_)
    filtered_dataframe_labels['dendrite_type_number'] = le1.transform(filtered_dataframe_labels['dendrite_type'])
    # print(filtered_dataframe_labels[["dendrite_type", "dendrite_type_number"]])

    le2.fit(filtered_dataframe_labels['structure_layer_name'])
    # print(le2.classes_)
    filtered_dataframe_labels['structure_layer_name_number'] = le2.transform(filtered_dataframe_labels['structure_layer_name'])
    # print(filtered_dataframe_labels[["structure_layer_name", "structure_layer_name_number"]])

    le3.fit(filtered_dataframe_labels['species'])
    # print(le3.classes_)
    filtered_dataframe_labels['species_number'] = le3.transform(filtered_dataframe_labels['species'])
    # print(filtered_dataframe_labels[["species_number", "species_number"]])

    filtered_dataframe_labels['processed_structure_layer_name'] = ''

    for ind in filtered_dataframe_labels.index.values:
        name = filtered_dataframe_labels.loc[ind, "structure_layer_name"]
        filtered_dataframe_labels.loc[ind, "processed_structure_layer_name"] = "6"
        if name in ["6a","6b"]:
            filtered_dataframe_labels.loc[ind, "processed_structure_layer_name"] = "6"
        elif name in ["2", "3"]:
            filtered_dataframe_labels.loc[ind, "processed_structure_layer_name"] = "2/3"
        else:
            filtered_dataframe_labels.loc[ind, "processed_structure_layer_name"] = name
    return filtered_dataframe_labels