from typing import Any

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model

from reliability_ai.config import (
    TCNDetectorConfig,
)
from reliability_ai.detection import (
    apply_threshold,
    calculate_percentile_threshold,
)
from reliability_ai.models import (
    build_tcn_autoencoder,
)
from reliability_ai.preprocessing import (
    make_sequences_from_valid_df,
)
from reliability_ai.results import (
    build_reconstruction_dataframe,
)
from reliability_ai.scoring import (
    calculate_feature_mse,
    calculate_sequence_mse,
    get_top_contributors,
)
from reliability_ai.training import (
    TCNTrainingConfig,
    train_tcn_autoencoder,
)

class TCNAnomalyDetector:
    """
    High-level TensorFlow/Keras TCN autoencoder anomaly detector.

    Detector coordinates:

    - training-data scaling;
    - contiguous sequence generation;
    - TCN autoencoder construction;
    - TCN autoencoder training;
    - reconstruction scoring;
    - anomaly-threshold calculation;
    - feature-contribution calculation;
    - engineering-unit reconstruction results.

    Parameters
    ----------
    config:
        Detector configuration.

    training_config:
        TCN training configuration. If omitted, validated proof-of-concept defaults are used.

    Notes
    -----
    Detector must be fitted before calling ``predict()``.

    Fitted detector owns:

    - a StandardScaler;
    - a trained Keras model;
    - a fitted anomaly threshold;
    - the ordered feature list;
    - model and detection configuration.
    """

    def __init__(
        self,
        config: TCNDetectorConfig,
        training_config: (
            TCNTrainingConfig | None
        ) = None,
    ) -> None:
        config.validate()

        if training_config is None:
            training_config = (
                TCNTrainingConfig()
            )

        training_config.validate()

        self.config = config

        self.training_config = (
            training_config
        )

        self.scaler: StandardScaler | None = (
            None
        )

        self.model: Model | None = None

        self.threshold: float | None = None

        self.training_history: History | None = (
            None
        )

        self.is_fitted: bool = False

    @property
    def feature_names(self) -> list[str]:
        """
        Ordered feature names used by detector.
        """

        return list(
            self.config.feature_names
        )

    @property
    def time_column(self) -> str:
        """
        Input timestamp-column name.
        """

        return self.config.time_column

    def _validate_dataframe(
        self,
        dataframe: pd.DataFrame,
        dataframe_name: str,
    ) -> None:
        """
        Validate that an input DataFrame contains required timestamp and feature columns.
        """

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            raise TypeError(
                f"{dataframe_name} must be a pandas DataFrame."
            )

        required_columns = {
            self.time_column,
            *self.feature_names,
        }

        missing_columns = sorted(
            required_columns
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                f"{dataframe_name} is missing required "
                f"columns: {missing_columns}"
            )

        if dataframe.empty:
            raise ValueError(
                f"{dataframe_name} must contain at least one row."
            )

    def _prepare_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare timestamps and feature values consistently before fitting the scaler.

        Core timestamp and missing-value behavior in sequence-generation module:

        - convert timestamps;
        - floor timestamps to the hour;
        - remove invalid timestamps;
        - sort chronologically;
        - retain first duplicate timestamp;
        - convert feature values to numeric;
        - forward-fill and backward-fill missing features.
        """

        work = dataframe[
            [
                self.time_column,
                *self.feature_names,
            ]
        ].copy()

        work[
            self.time_column
        ] = pd.to_datetime(
            work[
                self.time_column
            ],
            errors="coerce",
        ).dt.floor(
            "h"
        )

        work = (
            work
            .dropna(
                subset=[
                    self.time_column,
                ]
            )
            .sort_values(
                self.time_column
            )
            .drop_duplicates(
                subset=[
                    self.time_column,
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

        if work.empty:
            raise ValueError(
                "No valid rows remain after timestamp cleaning."
            )

        feature_data = work[
            self.feature_names
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if feature_data.isna().any().any():
            feature_data = (
                feature_data
                .ffill()
                .bfill()
            )

        if feature_data.isna().any().any():
            missing_features = (
                feature_data
                .columns[
                    feature_data
                    .isna()
                    .any()
                ]
                .tolist()
            )

            raise ValueError(
                "Feature values remain missing after forward-fill and backward-fill for: "
                f"{missing_features}"
            )

        if not np.isfinite(
            feature_data.to_numpy(
                dtype=np.float64
            )
        ).all():
            raise ValueError(
                "Feature values contain infinite values."
            )

        work[
            self.feature_names
        ] = feature_data

        return work

    def _fit_scaler(
        self,
        training_dataframe: pd.DataFrame,
    ) -> StandardScaler:
        """
        Fit StandardScaler using prepared training feature values only.
        """

        prepared_training_data = (
            self._prepare_dataframe(
                training_dataframe
            )
        )

        scaler = StandardScaler()

        scaler.fit(
            prepared_training_data[
                self.feature_names
            ]
        )

        return scaler

    def _make_sequences(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[
        np.ndarray,
        list[pd.Timestamp],
    ]:
        """
        Generate scaled contiguous sequences using detector's fitted scaler.
        """

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted."
            )

        sequences, end_times = (
            make_sequences_from_valid_df(
                df_valid=dataframe,
                time_col=self.time_column,
                feature_cols=(
                    self.feature_names
                ),
                scaler=self.scaler,
                sequence_length=(
                    self.config
                    .model
                    .sequence_length
                ),
            )
        )

        if sequences.shape[0] == 0:
            raise ValueError(
                "No valid contiguous sequences were generated. " 
                "Check sequence length, timestamp continuity, and available data."
            )

        return (
            sequences,
            end_times,
        )

    def _build_model(self) -> Model:
        """
        Build compiled TCN autoencoder from detector model configuration.
        """

        model_config = (
            self.config.model
        )

        return build_tcn_autoencoder(
            sequence_length=(
                model_config.sequence_length
            ),
            number_of_features=len(
                self.feature_names
            ),
            dilations=(
                model_config.dilations
            ),
            kernel_size=(
                model_config.kernel_size
            ),
            number_of_filters=(
                model_config.number_of_filters
            ),
            bottleneck_filters=(
                model_config.bottleneck_filters
            ),
            pool_size=(
                model_config.pool_size
            ),
            dropout_rate=(
                model_config.dropout_rate
            ),
            learning_rate=(
                model_config.learning_rate
            ),
        )

    @staticmethod
    def _reconstruct(
        model: Model,
        sequences: np.ndarray,
    ) -> np.ndarray:
        """
        Reconstruct scaled sequences using trained autoencoder.
        """

        reconstructed = model.predict(
            sequences,
            verbose=0,
        )

        reconstructed = np.asarray(
            reconstructed,
            dtype=np.float32,
        )

        if reconstructed.shape != sequences.shape:
            raise RuntimeError(
                "Reconstructed tensor shape does not match input sequence shape. "
                f"Received {reconstructed.shape} and {sequences.shape}."
            )

        if not np.isfinite(
            reconstructed
        ).all():
            raise RuntimeError(
                "The model reconstruction contains NaN or infinite values."
            )

        return reconstructed

    def fit(
        self,
        training_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
    ) -> History:
        """
        Fit complete TCN anomaly detector.

        Workflow
        --------
        1. Validate training and validation DataFrames.
        2. Fit StandardScaler using training features only.
        3. Generate training and validation sequences.
        4. Build compiled TCN autoencoder.
        5. Train TCN autoencoder.
        6. Reconstruct configured threshold-reference population.
        7. Calculate percentile anomaly threshold.

        Parameters
        ----------
        training_dataframe:
            Data used to fit scaler and train the model.

        validation_dataframe:
            Held-out data used to calculate validation loss and trigger early stopping.

        Returns
        -------
        History
            Keras training history.
        """

        self._validate_dataframe(
            dataframe=training_dataframe,
            dataframe_name=(
                "training_dataframe"
            ),
        )

        self._validate_dataframe(
            dataframe=validation_dataframe,
            dataframe_name=(
                "validation_dataframe"
            ),
        )

        self.scaler = self._fit_scaler(
            training_dataframe=(
                training_dataframe
            )
        )

        (
            training_sequences,
            _,
        ) = self._make_sequences(
            dataframe=training_dataframe
        )

        (
            validation_sequences,
            _,
        ) = self._make_sequences(
            dataframe=validation_dataframe
        )

        self.model = self._build_model()

        self.training_history = (
            train_tcn_autoencoder(
                model=self.model,
                training_sequences=(
                    training_sequences
                ),
                validation_sequences=(
                    validation_sequences
                ),
                config=self.training_config,
            )
        )

        reference_name = (
            self.config
            .detection
            .threshold_reference
        )

        if reference_name == "training":
            reference_sequences = (
                training_sequences
            )
        else:
            reference_sequences = (
                validation_sequences
            )

        reference_reconstruction = (
            self._reconstruct(
                model=self.model,
                sequences=(
                    reference_sequences
                ),
            )
        )

        reference_scores = (
            calculate_sequence_mse(
                actual=reference_sequences,
                reconstructed=(
                    reference_reconstruction
                ),
            )
        )

        self.threshold = (
            calculate_percentile_threshold(
                reference_scores=(
                    reference_scores
                ),
                percentile=(
                    self.config
                    .detection
                    .threshold_percentile
                ),
            )
        )

        self.is_fitted = True

        return self.training_history

    def _validate_fitted_state(self) -> None:
        """
        Validate that detector has all fitted artifacts required for prediction.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The detector must be fitted before calling predict()."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Fitted detector has no scaler."
            )

        if self.model is None:
            raise RuntimeError(
                "Fitted detector has no model."
            )

        if self.threshold is None:
            raise RuntimeError(
                "Fitted detector has no anomaly threshold."
            )

    def predict(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate anomaly-detection results for a DataFrame.

        The returned DataFrame contains:

        - sequence-end timestamp;
        - anomaly score;
        - fitted threshold;
        - Boolean anomaly classification;
        - top contributing feature;
        - feature-level reconstruction errors;
        - actual final-timestep values in engineering units;
        - reconstructed final-timestep values in engineering units.

        Parameters
        ----------
        dataframe:
            Input time-series data containing configured timestamp and feature columns.

        Returns
        -------
        pd.DataFrame
            Timestamp-aligned anomaly and reconstruction results.
        """

        self._validate_fitted_state()

        self._validate_dataframe(
            dataframe=dataframe,
            dataframe_name="dataframe",
        )

        (
            sequences,
            end_times,
        ) = self._make_sequences(
            dataframe=dataframe
        )

        if self.model is None:
            raise RuntimeError(
                "Fitted detector has no model."
            )

        if self.scaler is None:
            raise RuntimeError(
                "Fitted detector has no scaler."
            )

        if self.threshold is None:
            raise RuntimeError(
                "Fitted detector has no threshold."
            )

        reconstructed = self._reconstruct(
            model=self.model,
            sequences=sequences,
        )

        sequence_scores = (
            calculate_sequence_mse(
                actual=sequences,
                reconstructed=reconstructed,
            )
        )

        feature_errors = (
            calculate_feature_mse(
                actual=sequences,
                reconstructed=reconstructed,
            )
        )

        top_contributors = (
            get_top_contributors(
                feature_errors=feature_errors,
                feature_names=(
                    self.feature_names
                ),
            )
        )

        anomaly_flags = apply_threshold(
            scores=sequence_scores,
            threshold=self.threshold,
            inclusive=(
                self.config
                .detection
                .inclusive_threshold
            ),
        )

        reconstruction_results = (
            build_reconstruction_dataframe(
                actual=sequences,
                reconstructed=reconstructed,
                end_times=end_times,
                feature_names=(
                    self.feature_names
                ),
                scaler=self.scaler,
            )
        )

        summary_results = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    end_times
                ),
                "anomaly_score": (
                    sequence_scores
                ),
                "threshold": self.threshold,
                "is_anomaly": anomaly_flags,
                "top_contributor": (
                    top_contributors
                ),
            }
        )

        for feature_index, feature_name in enumerate(
            self.feature_names
        ):
            summary_results[
                f"{feature_name}__error"
            ] = feature_errors[
                :,
                feature_index,
            ]

        result = summary_results.merge(
            reconstruction_results,
            on="datetime",
            how="inner",
            validate="one_to_one",
        )

        result = result.sort_values(
            "datetime"
        ).reset_index(
            drop=True
        )

        return result

    def fit_predict(
        self,
        training_dataframe: pd.DataFrame,
        validation_dataframe: pd.DataFrame,
        prediction_dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit detector and immediately generate predictions.

        Method equivalent to:

            detector.fit(
                training_dataframe,
                validation_dataframe,
            )

            results = detector.predict(
                prediction_dataframe
            )
        """

        self.fit(
            training_dataframe=(
                training_dataframe
            ),
            validation_dataframe=(
                validation_dataframe
            ),
        )

        return self.predict(
            dataframe=prediction_dataframe
        )

    def get_fitted_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return a concise summary of detector state.

        Returns
        -------
        dict
            Detector state and fitted metadata.
        """

        return {
            "is_fitted": self.is_fitted,
            "number_of_features": len(
                self.feature_names
            ),
            "feature_names": (
                self.feature_names
            ),
            "time_column": (
                self.time_column
            ),
            "sequence_length": (
                self.config
                .model
                .sequence_length
            ),
            "threshold_percentile": (
                self.config
                .detection
                .threshold_percentile
            ),
            "threshold_reference": (
                self.config
                .detection
                .threshold_reference
            ),
            "threshold": self.threshold,
            "has_scaler": (
                self.scaler is not None
            ),
            "has_model": (
                self.model is not None
            ),
            "has_training_history": (
                self.training_history
                is not None
            ),
        }