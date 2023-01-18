import pandas as pd
import numpy as np
import itertools
from propy import *
from Bio.SeqUtils import ProtParam

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
        final.append(result)
        
    return final

def _do_analysis(protein):
    """Generates an analysis with Biopython ProtParam
    Args:
        protein (_type_): _description_
    """
    Biop_analysis = ProtParam.ProteinAnalysis(protein)
    return Biop_analysis


def Calculate_molecular_weight(train_array:pd.DataFrame)-> np.ndarray:
    """_summary_

    Args:
        train_array (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    final=[]
    for i in train_array:
        protein= "".join(i)
        Biop_analysis= _do_analysis(protein)
        molecular_weight = Biop_analysis.molecular_weight()
        final.append(molecular_weight)
        
    final= np.array(final)
    return final

def Calculate_isoelectric_point(train_array:pd.DataFrame)-> np.ndarray:
    """_summary_

    Args:
        train_array (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    final=[]
    for i in train_array:
        protein= "".join(i)
        Biop_analysis= _do_analysis(protein)
        isoelectric_point= Biop_analysis.isoelectric_point()
        final.append(isoelectric_point)
        
    final= np.array(final)
    return final

def Calculate_aromaticity(train_array:pd.DataFrame)-> np.ndarray:
    """_summary_

    Args:
        train_array (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    final=[]
    for i in train_array:
        protein= "".join(i)
        Biop_analysis= _do_analysis(protein)
        aromaticity= Biop_analysis.aromaticity()
        final.append(aromaticity)
        
    final= np.array(final)
    return final

def Calculate_instability_index(train_array:pd.DataFrame)-> np.ndarray:
    """_summary_

    Args:
        train_array (pd.DataFrame): _description_

    Returns:
        np.ndarray: _description_
    """

    final=[]
    for i in train_array:
        protein= "".join(i)
        Biop_analysis= _do_analysis(protein)
        instability_index= Biop_analysis.instability_index()
        final.append(instability_index)
        
    final= np.array(final)
    return final
