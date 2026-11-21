import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

from tensorflow.keras.callbacks import History

from reliability_ai import (
    DetectionConfig,
    TCNAnomalyDetector,
    TCNDetectorConfig,
    TCNModelConfig,
)
from reliability_ai.training import (
    TCNTrainingConfig,
)


# ============================================================
# Test configuration
# ============================================================

TIME_COLUMN = "datetime"

FEATURE_NAMES = (
    "pressure",
    "temperature",
    "flow",
)

SEQUENCE_LENGTH = 4
RANDOM_SEED = 42


# ============================================================
# Test-data helpers
# ============================================================

def make_hourly_dataframe(
    start: str,
    periods: int,
    seed: int,
) -> pd.DataFrame:
    """
    Create deterministic synthetic hourly equipment data.

    The signals contain smooth periodic behaviour and small
    amounts of random measurement noise.
    """

    random_generator = np.random.default_rng(
        seed
    )

    time_index = np.arange(
        periods,
        dtype=np.float64,
    )

    return pd.DataFrame(
        {
            TIME_COLUMN: pd.date_range(
                start=start,
                periods=periods,
                freq="h",
            ),
            "pressure": (
                50.0
                + 3.0
                * np.sin(
                    time_index / 6.0
                )
                + random_generator.normal(
                    loc=0.0,
                    scale=0.1,
                    size=periods,
                )
            ),
            "temperature": (
                80.0
                + 2.0
                * np.cos(
                    time_index / 8.0
                )
                + random_generator.normal(
                    loc=0.0,
                    scale=0.1,
                    size=periods,
                )
            ),
            "flow": (
                100.0
                + 5.0
                * np.sin(
                    time_index / 10.0
                )
                + random_generator.normal(
                    loc=0.0,
                    scale=0.2,
                    size=periods,
                )
            ),
        }
    )


# ============================================================
# Module-scoped fixtures
# ============================================================

@pytest.fixture(
    scope="module"
)
def detector_config() -> TCNDetectorConfig:
    """
    Create a lightweight detector configuration suitable
    for fast unit testing.
    """

    return TCNDetectorConfig(
        time_column=TIME_COLUMN,
        feature_names=FEATURE_NAMES,
        model=TCNModelConfig(
            sequence_length=(
                SEQUENCE_LENGTH
            ),
            dilations=(
                1,
            ),
            kernel_size=2,
            number_of_filters=4,
            bottleneck_filters=2,
            pool_size=1,
            dropout_rate=0.0,
            learning_rate=1e-3,
        ),
        detection=DetectionConfig(
            threshold_percentile=95.0,
            inclusive_threshold=False,
            threshold_reference="training",
        ),
    )


@pytest.fixture(
    scope="module"
)
def training_config() -> TCNTrainingConfig:
    """
    Use one epoch because these tests verify orchestration,
    not model quality or convergence.
    """

    return TCNTrainingConfig(
        epochs=1,
        batch_size=8,
        patience=0,
        restore_best_weights=True,
        shuffle=True,
        verbose=0,
        random_seed=RANDOM_SEED,
    )


@pytest.fixture(
    scope="module"
)
def training_dataframe() -> pd.DataFrame:
    """
    Synthetic training data.
    """

    return make_hourly_dataframe(
        start="2025-01-01 00:00:00",
        periods=40,
        seed=RANDOM_SEED,
    )


@pytest.fixture(
    scope="module"
)
def validation_dataframe() -> pd.DataFrame:
    """
    Synthetic validation data.
    """

    return make_hourly_dataframe(
        start="2025-02-01 00:00:00",
        periods=20,
        seed=RANDOM_SEED + 1,
    )


@pytest.fixture(
    scope="module"
)
def prediction_dataframe() -> pd.DataFrame:
    """
    Synthetic prediction data.
    """

    return make_hourly_dataframe(
        start="2025-03-01 00:00:00",
        periods=16,
        seed=RANDOM_SEED + 2,
    )


@pytest.fixture(
    scope="module"
)
def fitted_detector(
    detector_config: TCNDetectorConfig,
    training_config: TCNTrainingConfig,
    training_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> TCNAnomalyDetector:
    """
    Build and fit one detector for reuse by the prediction
    tests in this module.
    """

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    detector = TCNAnomalyDetector(
        config=detector_config,
        training_config=(
            training_config
        ),
    )

    detector.fit(
        training_dataframe=(
            training_dataframe
        ),
        validation_dataframe=(
            validation_dataframe
        ),
    )

    return detector


@pytest.fixture(
    scope="module"
)
def prediction_results(
    fitted_detector: TCNAnomalyDetector,
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate prediction results once for reuse by the
    result-schema tests.
    """

    return fitted_detector.predict(
        dataframe=prediction_dataframe
    )


# ============================================================
# Construction tests
# ============================================================

def test_detector_can_be_constructed(
    detector_config: TCNDetectorConfig,
    training_config: TCNTrainingConfig,
) -> None:
    """
    A valid configuration should create an unfitted
    detector successfully.
    """

    detector = TCNAnomalyDetector(
        config=detector_config,
        training_config=(
            training_config
        ),
    )

    assert isinstance(
        detector,
        TCNAnomalyDetector,
    )

    assert detector.is_fitted is False
    assert detector.scaler is None
    assert detector.model is None
    assert detector.threshold is None

    assert detector.feature_names == list(
        FEATURE_NAMES
    )

    assert detector.time_column == (
        TIME_COLUMN
    )


def test_prediction_before_fitting_raises_error(
    detector_config: TCNDetectorConfig,
    training_config: TCNTrainingConfig,
    prediction_dataframe: pd.DataFrame,
) -> None:
    """
    Prediction before fitting should fail with a clear
    high-level error.
    """

    detector = TCNAnomalyDetector(
        config=detector_config,
        training_config=(
            training_config
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="must be fitted",
    ):
        detector.predict(
            dataframe=prediction_dataframe
        )


# ============================================================
# Fitting tests
# ============================================================

def test_fitting_creates_required_artifacts(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    Fitting should create the scaler, model, threshold,
    and training history required for prediction.
    """

    assert fitted_detector.is_fitted is True

    assert fitted_detector.scaler is not None
    assert fitted_detector.model is not None
    assert fitted_detector.threshold is not None

    assert (
        fitted_detector.training_history
        is not None
    )


def test_fit_returns_keras_history(
    detector_config: TCNDetectorConfig,
    training_config: TCNTrainingConfig,
    training_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
) -> None:
    """
    fit() should return the underlying Keras History.

    This test uses a separate detector because the return
    value cannot be recovered from the fitted fixture itself.
    """

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    detector = TCNAnomalyDetector(
        config=detector_config,
        training_config=(
            training_config
        ),
    )

    history = detector.fit(
        training_dataframe=(
            training_dataframe
        ),
        validation_dataframe=(
            validation_dataframe
        ),
    )

    assert isinstance(
        history,
        History,
    )

    assert "loss" in history.history
    assert "val_loss" in history.history


def test_fitted_threshold_is_finite(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    The fitted percentile threshold must be a finite
    Python float.
    """

    assert isinstance(
        fitted_detector.threshold,
        float,
    )

    assert np.isfinite(
        fitted_detector.threshold
    )


def test_scaler_was_fitted_on_configured_features(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    The scaler must learn one mean and scale parameter per
    configured feature.
    """

    assert fitted_detector.scaler is not None

    assert fitted_detector.scaler.mean_.shape == (
        len(FEATURE_NAMES),
    )

    assert fitted_detector.scaler.scale_.shape == (
        len(FEATURE_NAMES),
    )


# ============================================================
# Prediction-schema tests
# ============================================================

def test_prediction_returns_dataframe(
    prediction_results: pd.DataFrame,
) -> None:
    """
    predict() should return a pandas DataFrame.
    """

    assert isinstance(
        prediction_results,
        pd.DataFrame,
    )


def test_prediction_has_expected_number_of_rows(
    prediction_results: pd.DataFrame,
    prediction_dataframe: pd.DataFrame,
) -> None:
    """
    For fully contiguous data:

        number of sequences
        =
        number of rows
        - sequence length
        + 1
    """

    expected_rows = (
        len(prediction_dataframe)
        - SEQUENCE_LENGTH
        + 1
    )

    assert len(
        prediction_results
    ) == expected_rows


def test_prediction_contains_summary_columns(
    prediction_results: pd.DataFrame,
) -> None:
    """
    The high-level result must contain the central anomaly
    detection outputs.
    """

    expected_columns = {
        "datetime",
        "anomaly_score",
        "threshold",
        "is_anomaly",
        "top_contributor",
    }

    assert expected_columns.issubset(
        prediction_results.columns
    )


def test_prediction_contains_feature_error_columns(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Each feature should have a reconstruction-error column.
    """

    for feature_name in FEATURE_NAMES:
        expected_column = (
            f"{feature_name}__error"
        )

        assert (
            expected_column
            in prediction_results.columns
        )


def test_prediction_contains_reconstruction_columns(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Each feature should have paired original and
    reconstructed engineering-unit columns.
    """

    for feature_name in FEATURE_NAMES:
        original_column = (
            f"original__{feature_name}"
        )

        reconstructed_column = (
            f"reconstructed__{feature_name}"
        )

        assert (
            original_column
            in prediction_results.columns
        )

        assert (
            reconstructed_column
            in prediction_results.columns
        )


def test_prediction_anomaly_flags_are_boolean(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Anomaly decisions must use Boolean values.
    """

    assert (
        prediction_results[
            "is_anomaly"
        ].dtype
        == np.bool_
    )


def test_prediction_scores_are_finite(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Anomaly and feature-level scores must be finite.
    """

    assert np.isfinite(
        prediction_results[
            "anomaly_score"
        ].to_numpy()
    ).all()

    for feature_name in FEATURE_NAMES:
        assert np.isfinite(
            prediction_results[
                f"{feature_name}__error"
            ].to_numpy()
        ).all()


def test_prediction_threshold_is_constant(
    fitted_detector: TCNAnomalyDetector,
    prediction_results: pd.DataFrame,
) -> None:
    """
    Every prediction row should use the detector's fitted
    threshold.
    """

    assert fitted_detector.threshold is not None

    expected_thresholds = np.full(
        shape=len(
            prediction_results
        ),
        fill_value=(
            fitted_detector.threshold
        ),
    )

    np.testing.assert_allclose(
        prediction_results[
            "threshold"
        ].to_numpy(),
        expected_thresholds,
    )


def test_top_contributors_are_valid_feature_names(
    prediction_results: pd.DataFrame,
) -> None:
    """
    The reported top contributor must come from the
    configured feature list.
    """

    contributors = set(
        prediction_results[
            "top_contributor"
        ].tolist()
    )

    assert contributors.issubset(
        set(FEATURE_NAMES)
    )


def test_prediction_timestamps_match_sequence_endpoints(
    prediction_results: pd.DataFrame,
    prediction_dataframe: pd.DataFrame,
) -> None:
    """
    Results should begin at the final timestamp of the
    first valid sequence.
    """

    expected_first_timestamp = (
        prediction_dataframe.loc[
            SEQUENCE_LENGTH - 1,
            TIME_COLUMN,
        ]
    )

    expected_last_timestamp = (
        prediction_dataframe.loc[
            len(prediction_dataframe) - 1,
            TIME_COLUMN,
        ]
    )

    assert (
        prediction_results.loc[
            0,
            "datetime",
        ]
        == expected_first_timestamp
    )

    assert (
        prediction_results.loc[
            len(prediction_results) - 1,
            "datetime",
        ]
        == expected_last_timestamp
    )


def test_result_timestamps_are_sorted(
    prediction_results: pd.DataFrame,
) -> None:
    """
    Detector results should be sorted chronologically.
    """

    timestamps = prediction_results[
        "datetime"
    ]

    assert timestamps.is_monotonic_increasing


# ============================================================
# Dataframe validation tests
# ============================================================

def test_missing_required_feature_raises_error(
    fitted_detector: TCNAnomalyDetector,
    prediction_dataframe: pd.DataFrame,
) -> None:
    """
    A missing configured feature should fail before
    preprocessing or inference.
    """

    invalid_dataframe = (
        prediction_dataframe.drop(
            columns=[
                "flow",
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        fitted_detector.predict(
            dataframe=invalid_dataframe
        )


def test_missing_time_column_raises_error(
    fitted_detector: TCNAnomalyDetector,
    prediction_dataframe: pd.DataFrame,
) -> None:
    """
    The configured timestamp column is mandatory.
    """

    invalid_dataframe = (
        prediction_dataframe.drop(
            columns=[
                TIME_COLUMN,
            ]
        )
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        fitted_detector.predict(
            dataframe=invalid_dataframe
        )


def test_empty_dataframe_raises_error(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    Empty input should fail with a clear message.
    """

    empty_dataframe = pd.DataFrame(
        columns=[
            TIME_COLUMN,
            *FEATURE_NAMES,
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        fitted_detector.predict(
            dataframe=empty_dataframe
        )


# ============================================================
# Timestamp-continuity test
# ============================================================

def test_prediction_does_not_cross_timestamp_gap(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    The detector must preserve the contiguous-sequence
    rule and never create a sequence across a missing hour.

    Twelve original rows with sequence length four would
    normally produce nine sequences.

    Removing the sixth timestamp produces two segments:

        first segment:
            5 rows -> 2 sequences

        second segment:
            6 rows -> 3 sequences

        total:
            5 valid sequences
    """

    dataframe = make_hourly_dataframe(
        start="2025-04-01 00:00:00",
        periods=12,
        seed=RANDOM_SEED + 3,
    )

    missing_timestamp = dataframe.loc[
        5,
        TIME_COLUMN,
    ]

    dataframe = dataframe.drop(
        index=5
    ).reset_index(
        drop=True
    )

    results = fitted_detector.predict(
        dataframe=dataframe
    )

    assert len(
        results
    ) == 5

    crossing_start = (
        missing_timestamp
        + pd.Timedelta(
            hours=1
        )
    )

    # The first valid sequence after the gap ends three
    # hours after the first post-gap timestamp.
    expected_first_post_gap_endpoint = (
        crossing_start
        + pd.Timedelta(
            hours=SEQUENCE_LENGTH - 1
        )
    )

    result_times = set(
        results[
            "datetime"
        ].tolist()
    )

    assert (
        expected_first_post_gap_endpoint
        in result_times
    )

    # No sequence endpoint should imply a four-row window
    # crossing the deleted hour.
    invalid_endpoints = {
        missing_timestamp
        + pd.Timedelta(
            hours=offset
        )
        for offset in range(
            0,
            SEQUENCE_LENGTH,
        )
    }

    assert result_times.isdisjoint(
        invalid_endpoints
    )


# ============================================================
# Fitted-summary test
# ============================================================

def test_fitted_summary_contains_expected_metadata(
    fitted_detector: TCNAnomalyDetector,
) -> None:
    """
    The fitted summary should expose concise detector
    metadata without serializing the fitted objects.
    """

    summary = (
        fitted_detector
        .get_fitted_summary()
    )

    assert summary[
        "is_fitted"
    ] is True

    assert summary[
        "number_of_features"
    ] == len(
        FEATURE_NAMES
    )

    assert summary[
        "feature_names"
    ] == list(
        FEATURE_NAMES
    )

    assert summary[
        "time_column"
    ] == TIME_COLUMN

    assert summary[
        "sequence_length"
    ] == SEQUENCE_LENGTH

    assert summary[
        "threshold_reference"
    ] == "training"

    assert summary[
        "has_scaler"
    ] is True

    assert summary[
        "has_model"
    ] is True

    assert summary[
        "has_training_history"
    ] is True