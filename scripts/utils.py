import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tsl.data import SpatioTemporalDataset
import torch



def clean_and_prepare_dataset(raw_df: pd.DataFrame, columns_to_std: list[str], columns_to_onehot: list[str], columns_to_fill: dict):
    #fill the missing values
    for col, fill_fn in columns_to_fill.items():
        if fill_fn == 'mean':
            raw_df[col] = raw_df[col].fillna(raw_df[col].mean())
        elif fill_fn == 'median':
            raw_df[col] = raw_df[col].fillna(raw_df[col].median())
        elif fill_fn == 'zero':
            raw_df[col] = raw_df[col].fillna(0)
    try:
        assert raw_df.isna().sum().sum() == 0
    except:
        col_with_nans = raw_df.columns[raw_df.isna().sum() > 0].tolist()
        raise AssertionError(f"NaNs are still present in columns {col_with_nans}")
    
    #scale and one hot

    scaler = StandardScaler()
    numerical_scaled = pd.DataFrame(scaler.fit_transform(raw_df[columns_to_std]), columns=columns_to_std, index=raw_df.index)
    categorical_encoded = pd.get_dummies(raw_df[columns_to_onehot], columns=columns_to_onehot, drop_first=False, dtype=int)
    remaining_columns = [col for col in raw_df.columns if col not in columns_to_std + columns_to_onehot]

    #concat
    processed_dataset = pd.concat([numerical_scaled, categorical_encoded, raw_df[remaining_columns]], axis=1)
    return processed_dataset

def fill_with_row_median(target):
    filled_target = target.copy()

    for i in range(len(target)):
        row = target.iloc[i]
        row_median = np.nanmedian(row)
        filled_target.iloc[i] = row.fillna(row_median)

    return filled_target

def create_st_dataset(nodes_features, timeseries, edge_index=None, edge_weight=None, exog_df=None, args=None):
    target = timeseries
    timestamps = timeseries.index

    # Create mask for valid targets (non-NaNs)
    mask = ~target.isna().to_numpy()
    mask = torch.tensor(mask, dtype=torch.bool)

    # Fill missing target values
    filled_target = fill_with_row_median(target)
    if filled_target.isna().any().any():
        filled_target.fillna(np.nanmedian(filled_target))


    # Create dataset
    assert (exog_df['datetime'][:-24].values == timestamps[24:]).all()

    # Create dataset
    dataset = SpatioTemporalDataset(
        target=filled_target.fillna(0),
        index=timestamps,
        covariates={
            "s": nodes_features.fillna(0),
            "u": exog_df.drop(columns=['datetime']).fillna(0).to_numpy(),
        },

        mask=mask,
        connectivity=(edge_index, edge_weight),
        horizon=args.get("horizon", 6),
        window=args.get("window", 36),
        stride=args.get("stride", 1)
    )

    return dataset, timestamps

def remove_outliers_zscore(data: torch.Tensor, threshold: float = 5.0):
    """
    Removes outliers along the time axis (dim=0) using z-score.
    data: Tensor of shape (time, node, feature)
    threshold: Z-score threshold beyond which values are considered outliers
    Returns: Masked tensor with outliers replaced by NaN
    """
    # Compute mean and std along the time axis
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Z-score
    z = (data - mean) / (std + 1e-8)

    # Mask: Keep only values within threshold
    mask = z.abs() < threshold

    # Replace outliers with NaN
    cleaned = data.copy()
    cleaned[~mask] = np.nan
    return cleaned

def compute_col_corr(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Create a mask for the upper triangle to avoid duplicate pairs
    mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)

    # Apply the mask to get unique pairs
    corr_upper = corr_matrix.where(mask)

    # Find pairs with correlation > 0.5 (absolute value)
    strong_correlations = []
    for i, row in enumerate(corr_upper.index):
        for j, col in enumerate(corr_upper.columns):
            if corr_upper.iloc[i, j] > 0.5 or corr_upper.iloc[i, j] < -0.5:
                strong_correlations.append((row, col, corr_upper.iloc[i, j]))

    # Sort by absolute correlation value (highest first)
    strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

    # Print the results
    print("Column pairs with correlation coefficient > 0.5:")
    for row, col, corr in strong_correlations:
        print(f"{row} & {col}: {corr:.3f}")

    return

def smape_fn(y_hat, y):
    denominator = (torch.abs(y) + torch.abs(y_hat)).clamp(min=1e-6)
    smape = 200.0 * torch.abs(y - y_hat) / denominator
    return smape