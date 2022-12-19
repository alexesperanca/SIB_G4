import pandas as pd
import numpy as np
import itertools
from propy import PyPro

def swap_ph_tm(train, update_train:pd.DataFrame)-> pd.DataFrame:
    """_summary_
    Swap ph and tm values in train subset if there ref in update
    Args:
        train (_type_): Train dataset
        update_train (pd.DataFrame): Updated train

    Returns:
        pd.DataFrame: new train dataset with swapped ph and tm
    """

    # Locate all null rows in features,
    get_all_nanfeat = update_train.isnull().all("columns")
    # Locate all indices in update_train
    # Drop all indices which update train iss null
    drop_all_indices = update_train[get_all_nanfeat].index
    train = train.drop(index=drop_all_indices)

    swap_ph_tm = update_train[~get_all_nanfeat].index
    train.loc[swap_ph_tm, ["pH", "tm"]] = update_train.loc[swap_ph_tm, ["pH", "tm"]]

    return train

def CalculateDipeptideComposition(train_array:pd.DataFrame)-> np.ndarray:
    """CalculatesDipeptideComposition for len(tran_array["protein_sequence"])
    Args:
        train_array (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """
    final=[]
    for i in train_array:
        protein= "".join(i)
        result = PyPro.CalculateDipeptideComposition(protein)
        result= list(result.values())
        final.append(result)
        
    final= np.array(final)
    return final
