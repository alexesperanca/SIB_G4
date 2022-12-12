import pandas as pd

def swap_ph_tm(train:pd.DataFrame, update_train:pd.DataFrame)-> pd.DataFrame:
    """_summary_
    Swap ph and tm values in train subset if ref in update
    Args:
        train (pd.DataFrame): Train dataset
        update_train (pd.DataFrame): Updated train

    Returns:
        pd.DataFrame: new train dataset with swaped ph and tm
    """

    #Locate all null rows in features, 
    get_all_nanfeat= update_train.isnull().all("columns")
    #Locate all indices in update_train 
    #Drop all indices which update train iss null
    drop_all_indices= update_train[get_all_nanfeat].index
    train= train.drop(index=drop_all_indices)

    swap_ph_tm = update_train[~get_all_nanfeat].index
    train.loc[swap_ph_tm,["pH","tm"]] = update_train.loc[swap_ph_tm,["pH","tm"]]

    return train
