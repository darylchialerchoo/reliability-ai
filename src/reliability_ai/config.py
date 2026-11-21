from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Literal

import yaml

ThresholdReference = Literal[
    "training",
    "validation",
]

@dataclass(frozen=True)
class TCNModelConfig:
    """
    Configuration for TensorFlow/Keras TCN autoencoder.

    Parameters
    ----------
    sequence_length:
        Number of consecutive time steps in each model input sequence.

    dilations:
        Dilation rates used by encoder and decoder TCN residual blocks.

    kernel_size:
        Kernel size used by the causal dilated Conv1D layers.

    number_of_filters:
        Number of filters used by the TCN residual blocks.

    bottleneck_filters:
        Number of filters used in the encoder bottleneck.

    pool_size:
        Temporal average-pooling and upsampling factor.

        A value of 1 preserves the original sequence length without temporal compression.

    dropout_rate:
        Dropout rate used inside each TCN residual block.

    learning_rate:
        Adam optimizer learning rate.
    """

    sequence_length: int = 48

    dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
    )

    kernel_size: int = 2
    number_of_filters: int = 16

    bottleneck_filters: int = 8
    pool_size: int = 1

    dropout_rate: float = 0.2
    learning_rate: float = 1e-3

    def validate(self) -> None:
        """
        Validate the TCN model configuration.
        """

        if self.sequence_length < 1:
            raise ValueError(
                "sequence_length must be at least 1."
            )

        if not self.dilations:
            raise ValueError(
                "dilations must contain at least one value."
            )

        if any(
            dilation < 1
            for dilation in self.dilations
        ):
            raise ValueError(
                "all dilation values must be at least 1."
            )

        if self.kernel_size < 1:
            raise ValueError(
                "kernel_size must be at least 1."
            )

        if self.number_of_filters < 1:
            raise ValueError(
                "number_of_filters must be at least 1."
            )

        if self.bottleneck_filters < 1:
            raise ValueError(
                "bottleneck_filters must be at least 1."
            )

        if self.pool_size < 1:
            raise ValueError(
                "pool_size must be at least 1."
            )

        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be in the range [0, 1)."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be greater than 0."
            )

@dataclass(frozen=True)
class DetectionConfig:
    """
    Configuration for converting reconstruction scores into anomaly decisions.

    Parameters
    ----------
    threshold_percentile:
        Percentile used to calculate the anomaly threshold.

    inclusive_threshold:
        If False:

            anomaly = score > threshold

        If True:

            anomaly = score >= threshold

    threshold_reference:
        Dataset used to calculate the threshold.

        ``training``:
            Calculate threshold from training-sequence reconstruction scores.

        ``validation``:
            Calculate the threshold from validation-sequence reconstruction scores.
    """

    threshold_percentile: float = 99.0
    inclusive_threshold: bool = False

    threshold_reference: ThresholdReference = (
        "training"
    )

    def validate(self) -> None:
        """
        Validate anomaly-detection configuration.
        """

        if not 0.0 <= self.threshold_percentile <= 100.0:
            raise ValueError(
                "threshold_percentile must be between 0 and 100 inclusive."
            )

        if not isinstance(
            self.inclusive_threshold,
            bool,
        ):
            raise TypeError(
                "inclusive_threshold must be a Boolean."
            )

        if self.threshold_reference not in {
            "training",
            "validation",
        }:
            raise ValueError(
                "threshold_reference must be 'training' or 'validation'."
            )


@dataclass(frozen=True)
class TCNDetectorConfig:
    """
    Complete configuration for a TCN anomaly detector.

    Parameters
    ----------
    time_column:
        Name of the timestamp column in input DataFrames.

    feature_names:
        Ordered feature names used by the scaler and model.

        Feature ordering is part of the trained-model and must not change between fitting and prediction.

    model:
        TCN architecture configuration.

    detection:
        Anomaly-threshold configuration.
    """

    time_column: str

    feature_names: tuple[str, ...]

    model: TCNModelConfig = field(
        default_factory=TCNModelConfig
    )

    detection: DetectionConfig = field(
        default_factory=DetectionConfig
    )

    def validate(self) -> None:
        """
        Validate the complete detector configuration.
        """

        if not isinstance(
            self.time_column,
            str,
        ):
            raise TypeError(
                "time_column must be a string."
            )

        if not self.time_column.strip():
            raise ValueError(
                "time_column must not be empty."
            )

        if not self.feature_names:
            raise ValueError(
                "feature_names must contain at least one feature."
            )

        if any(
            not isinstance(
                feature_name,
                str,
            )
            for feature_name in self.feature_names
        ):
            raise TypeError(
                "all feature_names must be strings."
            )

        if any(
            not feature_name.strip()
            for feature_name in self.feature_names
        ):
            raise ValueError(
                "feature_names must not contain empty values."
            )

        if len(
            set(self.feature_names)
        ) != len(
            self.feature_names
        ):
            raise ValueError(
                "feature_names must not contain duplicates."
            )

        if self.time_column in self.feature_names:
            raise ValueError(
                "time_column must not also appear in feature_names."
            )

        self.model.validate()
        self.detection.validate()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration into serializable dictionary.
        """

        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "TCNDetectorConfig":
        """
        Create detector configuration from dictionary.
        """

        if not isinstance(
            data,
            dict,
        ):
            raise TypeError(
                "data must be a dictionary."
            )

        model_data = data.get(
            "model",
            {},
        )

        detection_data = data.get(
            "detection",
            {},
        )

        feature_names = tuple(
            data["feature_names"]
        )

        model_data = dict(
            model_data
        )

        if "dilations" in model_data:
            model_data["dilations"] = tuple(
                model_data["dilations"]
            )

        config = cls(
            time_column=data["time_column"],
            feature_names=feature_names,
            model=TCNModelConfig(
                **model_data
            ),
            detection=DetectionConfig(
                **detection_data
            ),
        )

        config.validate()

        return config

    def to_yaml(
        self,
        path: str | Path,
    ) -> None:
        """
        Save detector configuration to YAML.
        """

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = self.to_dict()

        data["feature_names"] = list(
            self.feature_names
        )

        data["model"]["dilations"] = list(
            self.model.dilations
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            yaml.safe_dump(
                data,
                file,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> "TCNDetectorConfig":
        """
        Load detector configuration from YAML.
        """

        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = yaml.safe_load(
                file
            )

        if data is None:
            raise ValueError(
                "Configuration file is empty."
            )

        return cls.from_dict(
            data
        )