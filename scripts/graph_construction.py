import numpy as np
import torch
from typing import Callable, Union, Tuple, List, Optional

def create_balanced_weight_distribution(
    distances: np.ndarray,
    method: str = 'minmax_sqrt',
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a balanced weight distribution from distances.

    Args:
        distances: Array of distance values
        method: Weighting method:
            - 'minmax': Simple min-max scaling to [0.1, 1.0]
            - 'minmax_sqrt': Apply sqrt to compress range differences
            - 'rank': Use rank-based weights
            - 'sigmoid': Apply sigmoid transformation
            - 'power': Apply power transformation with alpha
        alpha: Parameter for methods that need it (power, sigmoid)

    Returns:
        Balanced weight array
    """
    # Handle empty input
    if len(distances) == 0:
        return np.array([], dtype=np.float32)

    # Make a copy to avoid modifying the original
    weights = np.array(distances, dtype=np.float32)

    # Check for valid values
    if np.isnan(weights).any() or np.isinf(weights).any():
        print("Warning: NaN or Inf values in distances")
        weights = np.nan_to_num(weights, nan=np.nanmean(weights), posinf=np.nanmax(weights), neginf=0)

    # Different weighting methods
    if method == 'minmax':
        # Simple min-max scaling to [0.1, 1.0]
        w_min, w_max = np.min(weights), np.max(weights)
        if w_max > w_min:
            weights = 0.1 + 0.9 * (weights - w_min) / (w_max - w_min)
        else:
            weights = np.ones_like(weights) * 0.5

    elif method == 'minmax_sqrt':
        # First normalize, then apply sqrt to compress the range
        w_min, w_max = np.min(weights), np.max(weights)
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
            weights = np.sqrt(weights)  # Apply sqrt to compress
            weights = 0.1 + 0.9 * weights  # Scale to [0.1, 1.0]
        else:
            weights = np.ones_like(weights) * 0.5

    elif method == 'rank':
        # Use ranks (fully uniform distribution )
        ranks = np.argsort(np.argsort(weights))
        weights = 0.1 + 0.9 * ranks / max(1, len(ranks) - 1)

    elif method == 'sigmoid':
        # Center the data
        weights = weights - np.mean(weights)
        # Apply sigmoid with alpha as steepness
        weights = 1.0 / (1.0 + np.exp(-alpha * weights))
        # Scale to [0, 1.0]
        weights = 0.1 + 0.9 * weights

    elif method == 'power':
        # Min-max normalize first
        w_min, w_max = np.min(weights), np.max(weights)
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
            # Apply power transformation
            weights = weights ** alpha
            # Scale to [0.1, 1.0]
            weights = 0.1 + 0.9 * weights
        else:
            weights = np.ones_like(weights) * 0.5

    return weights


def create_weighted_knn_graph(
    feature_data: Union[np.ndarray, torch.Tensor],
    geo_data: Union[np.ndarray, torch.Tensor],
    k: int,
    feature_weight: float = 0.5,
    geo_weight: float = 0.5,
    feature_distance_fn: Callable[[np.ndarray, np.ndarray], float] = None,
    geo_distance_fn: Callable[[np.ndarray, np.ndarray], float] = None,
    min_edges_per_node: int = 1,
    weight_method: str = 'minmax_sqrt'
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Create a weighted k-nearest neighbors (KNN) graph based on a combination of
    feature distances and geographical distances.

    Args:
        feature_data (np.ndarray or torch.Tensor): Shape (N, D_f), feature data like NDVI, elevation.
        geo_data (np.ndarray or torch.Tensor): Shape (N, 2), geographical coordinates (lat, lon).
        k (int): Number of closest neighbors to connect each node to.
        feature_weight (float): Weight for feature-based distance (0-1).
        geo_weight (float): Weight for geographical distance (0-1).
        feature_distance_fn (callable): Optional function for feature distance. Default: Euclidean.
        geo_distance_fn (callable): Optional function for geo distance. Default: Haversine.
        min_edges_per_node (int): Minimum number of edges per node.
        weight_method (str): Method for computing edge weights:
            - 'minmax': Simple min-max scaling
            - 'minmax_sqrt': Apply sqrt to compress range differences (default)
            - 'rank': Use rank-based weights (most balanced)
            - 'sigmoid': Apply sigmoid transformation
            - 'power': Apply power transformation

    Returns:
        edge_index (torch.LongTensor): Shape (2, E), where E is the number of edges.
        edge_weights (torch.FloatTensor): Shape (E,), normalized edge weights.
    """
    # Convert to numpy if needed
    if isinstance(feature_data, torch.Tensor):
        feature_data = feature_data.cpu().numpy()
    if isinstance(geo_data, torch.Tensor):
        geo_data = geo_data.cpu().numpy()

    N = feature_data.shape[0]
    assert geo_data.shape[0] == N, "Feature and geo data must have same number of nodes"

    # Compute feature distances
    if feature_distance_fn is None:
        # Vectorized Euclidean distance
        feature_diffs = feature_data[:, None, :] - feature_data[None, :, :]  
        feature_dist_mat = np.sqrt(np.maximum(0, (feature_diffs**2).sum(axis=-1)))  # maximum to ensure non-negative
    else:
        # Custom distance function
        feature_dist_mat = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                feature_dist_mat[i, j] = feature_distance_fn(feature_data[i], feature_data[j])

    # Compute geographical distances
    if geo_distance_fn is None:
        # Default to Euclidean for simplicity
        geo_diffs = geo_data[:, None, :] - geo_data[None, :, :]  # (N,N,2)
        geo_dist_mat = np.sqrt(np.maximum(0, (geo_diffs**2).sum(axis=-1)))  # maximum to ensure non-negative
    else:
        # Custom geo distance function
        geo_dist_mat = np.zeros((N, N), dtype=float)
        for i in range(N):
            for j in range(N):
                try:
                    geo_dist_mat[i, j] = geo_distance_fn(geo_data[i], geo_data[j])
                except Exception as e:
                    print(f"Error computing geo distance between nodes {i} and {j}: {e}")
                    # Fallback to a large distance value
                    geo_dist_mat[i, j] = np.finfo(np.float32).max / 10

    # Check values
    if np.isnan(feature_dist_mat).any():
        print(f"Warning: NaN values found in feature distances, replacing with large values")
        feature_dist_mat = np.nan_to_num(feature_dist_mat, nan=np.nanmax(feature_dist_mat) * 10)

    if np.isnan(geo_dist_mat).any():
        print(f"Warning: NaN values found in geo distances, replacing with large values")
        geo_dist_mat = np.nan_to_num(geo_dist_mat, nan=np.nanmax(geo_dist_mat) * 10)

    # Normalize distance matrices to [0,1] range
    feature_max = np.max(feature_dist_mat)
    if feature_max > 0:
        feature_dist_mat = feature_dist_mat / feature_max

    geo_max = np.max(geo_dist_mat)
    if geo_max > 0:
        geo_dist_mat = geo_dist_mat / geo_max

    # Combined weighted distance
    combined_dist_mat = feature_weight * feature_dist_mat + geo_weight * geo_dist_mat

    # check values
    if np.isnan(combined_dist_mat).any():
        print("Warning: NaN values found in combined distance matrix")
        combined_dist_mat = np.nan_to_num(combined_dist_mat, nan=1.0)

    # Create edges based on k-nearest neighbors
    edge_list = []
    edge_weights_list = []

    for i in range(N):
        distances = combined_dist_mat[i].copy()
        distances[i] = np.inf  # Exclude self

        # Check values
        if np.all(np.isinf(distances)):
            print(f"Warning: Node {i} has all infinite distances to other nodes")
            distances = combined_dist_mat[i].copy()
            distances[i] = np.max(distances) * 2

        # Get k nearest neighbors
        nn_idx = np.argsort(distances)[:k]

        # Ensure minimum edges per node
        if len(nn_idx) < min_edges_per_node:
            nn_idx = np.argsort(distances)[:min_edges_per_node]

        # Create edges and store weights
        for j in nn_idx:
            edge_list.append((i, j))

            dist = combined_dist_mat[i, j]
            if np.isnan(dist) or np.isinf(dist) or dist <= 0:
                weight = 0.001  # Small default weight for problematic distances
            else:
                # Use inverse distance as weight with a small epsilon to avoid division by zero
                weight = 1.0 / (dist + 1e-8)

            edge_weights_list.append(weight)

    edge_weights = np.array(edge_weights_list, dtype=np.float32)

    # Check values
    if np.isnan(edge_weights).any() or np.isinf(edge_weights).any():
        print("Warning: Found NaN or Inf values in edge weights")
        edge_weights = np.nan_to_num(edge_weights, nan=0.001, posinf=100.0, neginf=0.001)

    # Apply the selected weight repartition function
    edge_weights = create_balanced_weight_distribution(edge_weights, method=weight_method)

    # Print distribution information
    quantiles = np.quantile(edge_weights, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"Weight distribution quantiles: {quantiles}")

    try:
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_weights = torch.FloatTensor(edge_weights)
    except Exception as e:
        print(f"Error converting to tensors: {e}")
        # Fallback with explicit type conversion
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return edge_index, edge_weights


def edge_index_to_weighted_adj_matrix(
    edge_index: torch.Tensor,
    edge_weights: torch.Tensor,
    num_nodes: int,
    symmetric: bool = True
) -> torch.Tensor:
    """
    Convert edge_index and weights to a weighted adjacency matrix.

    Args:
        edge_index (torch.Tensor): Shape (2, E), containing source and target nodes.
        edge_weights (torch.Tensor): Shape (E,), containing the weight of each edge.
        num_nodes (int): Total number of nodes.
        symmetric (bool): Whether to make the adjacency matrix symmetric (undirected graph).

    Returns:
        torch.Tensor: Weighted adjacency matrix of shape (N, N).
    """
    # Ensure inputs are valid
    if edge_index.shape[1] != edge_weights.shape[0]:
        raise ValueError(f"Edge index has {edge_index.shape[1]} edges but weights has {edge_weights.shape[0]} values")

    # Check for index out of bounds
    max_idx = edge_index.max().item()
    if max_idx >= num_nodes:
        raise ValueError(f"Edge index contains node {max_idx} but num_nodes is {num_nodes}")

    # Create adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)

    src, dst = edge_index
    adj_matrix[src, dst] = edge_weights

    if symmetric:
        # For symmetric matrix, we need to be careful with duplicate edges
        # If an edge exists in both directions, take the max weight
        adj_matrix = torch.maximum(adj_matrix, adj_matrix.t())

    return adj_matrix



def haversine_distance(coord1, coord2, coordinates_unit = 'degrees'):
    """
    Calculate the Haversine distance between two points in kilometers,
    or Euclidean distance if coordinates are in meters.

    Args:
        coord1 (np.ndarray): [latitude, longitude] or [x, y] of first point
        coord2 (np.ndarray): [latitude, longitude] or [x, y] of second point
        coordinates_unit (str): 'degrees', 'radians', or 'meters'

    Returns:
        float: Distance in kilometers
    """
    # checks for NaNs
    if np.isnan(coord1).any() or np.isnan(coord2).any():
        return float('inf')  # Return inf for NaN coordinates

    if coordinates_unit == 'meters':
        return np.linalg.norm(coord1 - coord2) / 1000.0

    try:
        # Earth radius in kilometers
        R = 6371.0

        # Convert coordinates to radians if necessary
        if coordinates_unit == 'degrees':
            lat1, lon1 = np.radians(coord1)
            lat2, lon2 = np.radians(coord2)
        elif coordinates_unit == 'radians':
            lat1, lon1 = coord1
            lat2, lon2 = coord2
        else:
            raise ValueError("Invalid coordinates_unit. Must be 'degrees', 'radians', or 'meters'.")

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(np.maximum(0, a)), np.sqrt(np.maximum(0, 1-a)))
        distance = R * c

        return distance
    except Exception as e:
        print(f"Error in haversine calculation: {e}")
        return float('inf')



def create_graph_from_dataframe(df, feature_cols, geo_cols, id_col, k=10, feature_weight=0.6, geo_weight=0.4, weight_method='rank'):
    """
    Create a graph from a dataframe with feature and geographical data.

    Args:
        df (pandas.DataFrame): Input dataframe
        feature_cols (List[str]): Column names for feature data
        geo_cols (List[str]): Column names for geographical coordinates [lat, lon]
        id_col (str): Column name for node IDs
        k (int): Number of neighbors
        feature_weight (float): Weight for feature distance
        geo_weight (float): Weight for geographical distance
        weight_method (str): Method for computing edge weights:
            - 'minmax': Simple min-max scaling
            - 'minmax_sqrt': Apply sqrt to compress range differences
            - 'rank': Use rank-based weights (most balanced)
            - 'sigmoid': Apply sigmoid transformation
            - 'power': Apply power transformation

    Returns:
        Tuple containing edge_index, edge_weights, and node mapping
    """
    # Validate columns names
    for col in feature_cols + geo_cols + [id_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")

    # Check for NaN values
    if df[feature_cols + geo_cols].isna().any().any():
        print("Warning: NaN values found in input columns. Filling with column means.")
        df = df.copy()
        df[feature_cols + geo_cols] = df[feature_cols + geo_cols].fillna(df[feature_cols + geo_cols].mean())

    # Extract node features and geo coordinates
    feature_data = df[feature_cols].to_numpy() #environmental/urban/spatial features
    geo_data = df[geo_cols].to_numpy() #geographical coordinates

    # Create node ID mapping in case node id does not correspond to the index
    node_ids = df[id_col].unique()
    node_mapping = {id_val: idx for idx, id_val in enumerate(node_ids)}

    # Get node indices from original IDs
    # node_indices = np.array([node_mapping[id_val] for id_val in df[id_col]])

    # Print stats
    print(f"Creating graph with {len(node_ids)} nodes")
    print(f"Feature data shape: {feature_data.shape}")
    print(f"Geo data shape: {geo_data.shape}")

    # Create graph with error handling
    try:
        edge_index, edge_weights = create_weighted_knn_graph(
            feature_data=feature_data,
            geo_data=geo_data,
            k=k,
            feature_weight=feature_weight,
            geo_weight=geo_weight,
            geo_distance_fn=haversine_distance,
            min_edges_per_node=1,
            weight_method=weight_method
        )

        # Print edge weight stats for information
        if edge_weights.numel() > 0:
            print(f"Edge weights - Min: {edge_weights.min().item():.6f}, "
                  f"Max: {edge_weights.max().item():.6f}, "
                  f"Mean: {edge_weights.mean().item():.6f}")
            print(f"NaN values in weights: {torch.isnan(edge_weights).sum().item()}")
            print(f"Inf values in weights: {torch.isinf(edge_weights).sum().item()}")
        else:
            print("Warning: No edges created!")

        #Create the weighted adjacency matrix
        weighted_adj = edge_index_to_weighted_adj_matrix(
            edge_index=edge_index,
            edge_weights=edge_weights,
            num_nodes=len(node_ids),
            symmetric=False #not directed
        )

        return edge_index, edge_weights, weighted_adj, node_mapping

    except Exception as e:
        print(f"Error creating graph: {e}")
        # Return minimal valid output for error cases
        return (torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(0, dtype=torch.float),
                torch.eye(len(node_ids), dtype=torch.float),
                node_mapping)

#Exemple usage
"""edge_index, edge_weights, weighted_adj, node_mapping = create_graph_from_dataframe(
    df=node_features_dataset,
    feature_cols=feature_colss
    geo_cols=['latitude', 'longitude'],
    id_col='col_id',
    k=10,
    feature_weight=0.6,
    geo_weight=0.4,
    weight_method='sigmoid'
)"""