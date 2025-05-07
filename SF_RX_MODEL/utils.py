import pandas as pd
import numpy as np
import torch

def load_data(X_path, y_path):
    """
    Load datasets from CSV files.

    Args:
        X_path (str): Path to the 'X' dataset.
        y_path (str): Path to the 'y' dataset.

    Returns:
        pd.DataFrame: DataFrame for 'X'.
        pd.DataFrame: DataFrame for 'y'.
    """
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    return X, y

def split_data(X, y, val_fold, lv):
    """
    Split validation data based on a specified fold.

    Args:
        X (pd.DataFrame): DataFrame for drug feature.
        y (pd.DataFrame): DataFrame for targets and scaffolds.
        val_fold (int): Scaffold fold ID for validation.

    Returns:
        pd.DataFrame: Validation 'X'.
        pd.DataFrame: Validation 'y'.
        pd.DataFrame: Training 'X'.
        pd.DataFrame: Training 'y'.
    """
    if lv == 1:
        val_y = y[y.lv1 == val_fold]
        tr_y = y[y.lv1 != val_fold]
    if lv == 2:
        val_y = y[y.FoldID1 == val_fold]
        tr_y = y[y.FoldID1 != val_fold]
    if lv == 3: 
        val_y = y[(y.FoldID1 == val_fold) ^ (y.FoldID2 == val_fold)]
        tr_y = y[~((y.FoldID1 == val_fold) ^ (y.FoldID2 == val_fold)) & ~((y.FoldID1 == val_fold) & (y.FoldID2 == val_fold))]
    if lv == 4: 
        val_y = y[(y.FoldID1 == val_fold) & (y.FoldID2 == val_fold)]
        tr_y = y[~((y.FoldID1 == val_fold) & (y.FoldID2 == val_fold)) & ~((y.FoldID1 == val_fold) ^ (y.FoldID2 == val_fold))]
         
    val_X = X.loc[val_y.index]
    tr_X = X.loc[tr_y.index]
    
    return val_X, val_y, tr_X, tr_y

def adj_lv2_split_validation_data(X, y, val_fold):
    """
    Split validation data based on a specified fold. (Adjusted model)

    Args:
        X (pd.DataFrame): DataFrame for drug feature.
        y (pd.DataFrame): DataFrame for targets and scaffolds.
        val_fold (int): Scaffold fold ID for validation.

    Returns:
        pd.DataFrame: Validation 'X'.
        pd.DataFrame: Validation 'y'.
        np.array: Rows removed from validation set.
    """
    val_y = y[y.FoldID1 == val_fold]
    val_X = X.loc[val_y.index]

    # Filter non-diagonal validation data with direction 0
    non_diag_val_y = val_y[val_y.FoldID2 != val_fold]
    non_diag_val_y_dir0 = non_diag_val_y[non_diag_val_y.direction == 0]

    # Randomly remove half of non-diagonal rows
    np.random.seed(42)
    rows_to_remove = np.random.choice(non_diag_val_y_dir0.index, size=len(non_diag_val_y_dir0) // 2, replace=False)
    val_y = val_y.drop(rows_to_remove)
    val_X = val_X.drop(rows_to_remove)

    return val_X, val_y, rows_to_remove

def adj_lv2_split_training_data(X, y, val_fold, rows_to_remove):
    """
    Split training data and filter based on validation rows. (Adjusted model)

    Args:
        X (pd.DataFrame): DataFrame for drug feature.
        y (pd.DataFrame): DataFrame for targets and scaffolds.
        val_fold (int): Scaffold fold ID for validation.
        rows_to_remove (np.array): Rows removed from validation.

    Returns:
        pd.DataFrame: Training 'X'.
        pd.DataFrame: Training 'y'.
    """
    y['unique_IDs'] = (y.index // 2) + 1
    
    tr_y = y[y.FoldID1 != val_fold]
    tr_X = X.loc[tr_y.index]
    
    # Filter training rows related to validation fold
    tr_to_filt = tr_y[(tr_y.FoldID1 != val_fold) & (tr_y.FoldID2 == val_fold)]
    rows_to_remove2 = tr_to_filt[~(tr_to_filt.unique_IDs.isin(y.loc[rows_to_remove].unique_IDs))].index

    tr_y = tr_y.drop(rows_to_remove2)
    tr_X = tr_X.drop(rows_to_remove2)

    return tr_X, tr_y

def gini_impurity(column, cat_levels):
    """
    Calculates the Gini impurity for a given column of categorical data.

    Parameters:
    column (array-like): A column of categorical data represented as integers.
    cat_levels (int): The number of possible categories (levels) in the data.

    Returns:
    float: The Gini impurity of the column.
    """
    possible_values = np.arange(cat_levels)
    counts = np.bincount(column, minlength=len(possible_values))
    probabilities = counts / len(column)
    
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def load_preds(rst_list, target):
    """
    Processes model predictions, calculates Gini impurity for each column,
    and returns the results.

    Parameters:
    rst_list (list of torch.Tensor): A list of prediction tensors from the model,
                                      where each tensor corresponds to an inference iteration.
    target (str): The target output category ('sev', 'desc', or 'dir') used to determine
                  the number of categorical levels.

    Returns:
    pandas.Series: Gini impurity values for each column of predictions.
    """
    output_levs = {'sev': 3, 'desc': 30, 'dir': 2}
    infs_pred = rst_list
    pred_rst = torch.stack(infs_pred, dim=0).cpu().numpy()
    pred_rst = pd.DataFrame(pred_rst)

    gini_results = pred_rst.apply(lambda col: gini_impurity(col, output_levs[target]))

    return gini_results