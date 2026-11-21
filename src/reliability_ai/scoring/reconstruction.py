import numpy as np

def calculate_squared_error(
    actual: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Calculate pointwise squared reconstruction error.

    Parameters
    ----------
    actual:
        Actual input sequences with shape:

        (
            number_of_sequences,
            sequence_length,
            number_of_features,
        )

    reconstructed:
        Reconstructed sequences with the same shape as ``actual``.

    Returns
    -------
    np.ndarray
        Pointwise squared reconstruction error with same shape as inputs.
    """

    if actual.shape != reconstructed.shape:
        raise ValueError(
            "actual and reconstructed must have identical "
            "shapes. "
            f"Received {actual.shape} and "
            f"{reconstructed.shape}."
        )

    return np.square(
        actual - reconstructed
    )

def calculate_sequence_mse(
    actual: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Calculate one mean squared reconstruction error for each sequence.
    """

    squared_error = calculate_squared_error(
        actual=actual,
        reconstructed=reconstructed,
    )

    return squared_error.mean(
        axis=(1, 2)
    )

def calculate_feature_mse(
    actual: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    """
    Calculate mean reconstruction error for every feature within each sequence.

    Returns
    -------
    np.ndarray
        Shape:

        (
            number_of_sequences,
            number_of_features,
        )
    """

    squared_error = calculate_squared_error(
        actual=actual,
        reconstructed=reconstructed,
    )

    return squared_error.mean(
        axis=1
    )

def get_top_contributors(
    feature_errors: np.ndarray,
    feature_names: list[str],
) -> list[str]:
    """
    Identify feature with the largest reconstruction error for each sequence.
    """

    if feature_errors.ndim != 2:
        raise ValueError(
            "feature_errors must have shape (number_of_sequences, number_of_features)."
        )

    if feature_errors.shape[1] != len(
        feature_names
    ):
        raise ValueError(
            "The number of feature names must match the feature dimension of feature_errors."
        )

    top_indices = np.argmax(
        feature_errors,
        axis=1,
    )

    return [
        feature_names[index]
        for index in top_indices
    ]