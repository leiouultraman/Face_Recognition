import numpy as np
from typing import Union

def find_cosine_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        distances = 1 - dot_product / (source_norm * test_norm)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        cosine_similarities = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = 1 - cosine_similarities
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances

def find_angular_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:

    # calculate cosine similarity first
    # then convert to angular distance
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    if source_representation.ndim == 1 and test_representation.ndim == 1:
        # single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        similarity = dot_product / (source_norm * test_norm)
        distances = np.arccos(similarity) / np.pi
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        # list of embeddings (batch)
        source_normed = l2_normalize(source_representation, axis=1)  # (N, D)
        test_normed = l2_normalize(test_representation, axis=1)  # (M, D)
        similarity = np.dot(test_normed, source_normed.T)  # (M, N)
        distances = np.arccos(similarity) / np.pi
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances

def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> Union[np.float64, np.ndarray]:
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    # Single embedding case (1D arrays)
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        distances = np.linalg.norm(source_representation - test_representation)
    # Batch embeddings case (2D arrays)
    elif source_representation.ndim == 2 and test_representation.ndim == 2:
        diff = (
            source_representation[None, :, :] - test_representation[:, None, :]
        )  # (N, D) - (M, D)  = (M, N, D)
        distances = np.linalg.norm(diff, axis=2)  # (M, N)
    else:
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    return distances


def l2_normalize(
    x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 1e-10
) -> np.ndarray:
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + epsilon)

def find_distance(
    alpha_embedding: Union[np.ndarray, list],
    beta_embedding: Union[np.ndarray, list],
    distance_metric: str,
) -> Union[np.float64, np.ndarray]:
    # Convert inputs to numpy arrays if necessary
    alpha_embedding = np.asarray(alpha_embedding)
    beta_embedding = np.asarray(beta_embedding)

    # Ensure that both embeddings are either 1D or 2D
    if alpha_embedding.ndim != beta_embedding.ndim or alpha_embedding.ndim not in (1, 2):
        raise ValueError(
            f"Both embeddings must be either 1D or 2D, but received "
            f"alpha shape: {alpha_embedding.shape}, beta shape: {beta_embedding.shape}"
        )

    if distance_metric == "cosine":
        distance = find_cosine_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "angular":
        distance = find_angular_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean":
        distance = find_euclidean_distance(alpha_embedding, beta_embedding)
    elif distance_metric == "euclidean_l2":
        axis = None if alpha_embedding.ndim == 1 else 1
        normalized_alpha = l2_normalize(alpha_embedding, axis=axis)
        normalized_beta = l2_normalize(beta_embedding, axis=axis)
        distance = find_euclidean_distance(normalized_alpha, normalized_beta)
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    return np.round(distance, 6)