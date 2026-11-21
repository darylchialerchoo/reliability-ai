import numpy as np

def validate_scores(
    scores: np.ndarray,
) -> None:
    """
    Validate a one-dimensional array of anomaly scores.

    Parameters
    ----------
    scores:
        One anomaly score per sequence.

    Raises
    ------
    TypeError
        If scores is not a NumPy array.

    ValueError
        If scores is not one-dimensional, is empty, or contains NaN or infinite values.
    """

    if not isinstance(
        scores,
        np.ndarray,
    ):
        raise TypeError(
            "scores must be a NumPy array."
        )

    if scores.ndim != 1:
        raise ValueError(
            "scores must be one-dimensional with shape (number_of_sequences,). "
            f"Received shape {scores.shape}."
        )

    if scores.size == 0:
        raise ValueError(
            "scores must contain at least one value."
        )

    if not np.isfinite(
        scores
    ).all():
        raise ValueError(
            "scores contains NaN or infinite values."
        )

def validate_percentile(
    percentile: float,
) -> None:
    """
    Validate an anomaly-threshold percentile.

    Parameters
    ----------
    percentile:
        Percentile expressed on 0 to 100 scale.

    Raises
    ------
    TypeError
        If percentile is not numeric.

    ValueError
        If percentile is outside the interval [0, 100].
    """

    if not isinstance(
        percentile,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "percentile must be numeric."
        )

    if not np.isfinite(
        percentile
    ):
        raise ValueError(
            "percentile must be finite."
        )

    if not 0.0 <= float(
        percentile
    ) <= 100.0:
        raise ValueError(
            "percentile must be between 0 and 100 inclusive."
        )


def calculate_percentile_threshold(
    reference_scores: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """
    Calculate an anomaly threshold from a reference score distribution.

    The reference scores should normally come from normal operating conditions. 

    Parameters
    ----------
    reference_scores:
        One-dimensional reconstruction-error scores from reference dataset.

    percentile:
        Percentile used for the anomaly threshold.

        The default is 99.0, corresponding to P99.

    Returns
    -------
    float
        Scalar reconstruction-error threshold.
    """

    validate_scores(
        scores=reference_scores
    )

    validate_percentile(
        percentile=percentile
    )

    threshold = np.percentile(
        reference_scores,
        percentile,
    )

    return float(
        threshold
    )


def apply_threshold(
    scores: np.ndarray,
    threshold: float,
    inclusive: bool = False,
) -> np.ndarray:
    """
    Classify reconstruction scores using anomaly threshold.

    Parameters
    ----------
    scores:
        One-dimensional anomaly-score array.

    threshold:
        Scalar anomaly threshold.

    inclusive:
        Determines how scores exactly equal to threshold are treated.

        If False:

            anomaly = score > threshold

        If True:

            anomaly = score >= threshold

        Default is False.

    Returns
    -------
    np.ndarray
        Boolean anomaly mask with shape:

        (number_of_sequences,)

    Raises
    ------
    TypeError
        If threshold is not numeric or inclusive is not a Boolean.

    ValueError
        If threshold is not finite.
    """

    validate_scores(
        scores=scores
    )

    if not isinstance(
        threshold,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "threshold must be numeric."
        )

    if not np.isfinite(
        threshold
    ):
        raise ValueError(
            "threshold must be finite."
        )

    if not isinstance(
        inclusive,
        (bool, np.bool_),
    ):
        raise TypeError(
            "inclusive must be a Boolean."
        )

    if inclusive:
        return scores >= float(
            threshold
        )

    return scores > float(
        threshold
    )

def calculate_threshold_and_anomalies(
    reference_scores: np.ndarray,
    scores: np.ndarray,
    percentile: float = 99.0,
    inclusive: bool = False,
) -> tuple[float, np.ndarray]:
    """
    Calculate a percentile threshold and apply it to scores.

    Function combines:

        calculate_percentile_threshold()

    and:

        apply_threshold()

    Parameters
    ----------
    reference_scores:
        Score distribution used to calculate the threshold.

    scores:
        Scores to classify.

    percentile:
        Percentile threshold on 0 to 100 scale.

    inclusive:
        Whether scores equal to threshold should be classified as anomalies.

    Returns
    -------
    tuple[float, np.ndarray]
        Threshold and Boolean anomaly mask.
    """

    threshold = calculate_percentile_threshold(
        reference_scores=reference_scores,
        percentile=percentile,
    )

    anomalies = apply_threshold(
        scores=scores,
        threshold=threshold,
        inclusive=inclusive,
    )

    return (
        threshold,
        anomalies,
    )