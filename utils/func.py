import pandas as pd
<<<<<<< HEAD
from collections import Counter
from itertools import zip_longest
from Bio import pairwise2
from Bio.pairwise2 import format_alignment


def swap_ph_tm(train: pd.DataFrame, update_train: pd.DataFrame) -> pd.DataFrame:
    """Swap ph and tm values in train subset if ref in update

=======
import numpy as np
import itertools
from propy import PyPro

def swap_ph_tm(train, update_train:pd.DataFrame)-> pd.DataFrame:
    """_summary_
    Swap ph and tm values in train subset if there ref in update
>>>>>>> 64eb4726a49b29c4c4a34b6b715800ecadb15bcc
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

<<<<<<< HEAD

def obtain_sequences_values(sequences: list) -> list:
    """Obtain Blosum64 values for each sequence with the consensus.

    Args:
        sequences (list): Sequences from dataset.

    Returns:
        list: Blosum64 scores between each sequence and the consensus.
    """
    zipped_aa = zip_longest(*sequences)
    common_aa_pos = {}
    for pos, elements in enumerate(zipped_aa):
        counts = Counter(elements)
        most_commons = counts.most_common(2)
        max_aa = most_commons[0][0]
        if not max_aa:
            max_aa = most_commons[1][0]
        common_aa_pos[pos] = max_aa
    consensus = "".join(aa for aa in common_aa_pos.values())
    for seq in sequences:
        print(pairwise2.align.globalxx(seq, consensus, score_only=True))
=======
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
>>>>>>> 64eb4726a49b29c4c4a34b6b715800ecadb15bcc
