from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model

@dataclass(frozen=True)
class TCNTrainingConfig:
    """
    Configuration for training a TensorFlow/Keras TCN autoencoder.

    Defaults reproduce the validated HPC proof-of-concept training settings.

    Parameters
    ----------
    epochs:
        Maximum number of training epochs.

    batch_size:
        Number of sequences processed in each training batch.

    patience:
        Number of consecutive epochs without validation-loss improvement before early stopping.

    restore_best_weights:
        Whether to restore model weights from the epoch with the lowest validation loss.

    shuffle:
        Whether to shuffle training sequences before each epoch.

    verbose:
        Keras training verbosity:

        0:
            No epoch output.

        1:
            Progress bar.

        2:
            One output line per epoch.

    random_seed:
        NumPy and TensorFlow random seed applied immediately before model training.
    """

    epochs: int = 500
    batch_size: int = 128
    patience: int = 5

    restore_best_weights: bool = True
    shuffle: bool = True

    verbose: int = 1
    random_seed: int = 42

    def validate(self) -> None:
        """
        Validate training configuration.

        Raises
        ------
        ValueError
            If any configuration value is invalid.
        """

        if self.epochs < 1:
            raise ValueError(
                "epochs must be at least 1."
            )

        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1."
            )

        if self.patience < 0:
            raise ValueError(
                "patience must be at least 0."
            )

        if self.verbose not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                "verbose must be 0, 1, or 2."
            )

        if self.random_seed < 0:
            raise ValueError(
                "random_seed must be at least 0."
            )

def validate_training_sequences(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> None:
    """
    Validate training and validation sequence tensors.

    Both arrays must use the library sequence contract:

        (
            number_of_sequences,
            sequence_length,
            number_of_features,
        )

    Training and validation data may contain different numbers of sequences, 
    but their sequence lengths and feature counts must match.

    Parameters
    ----------
    training_sequences:
        Training sequence tensor.

    validation_sequences:
        Validation sequence tensor.

    Raises
    ------
    TypeError
        If either input is not a NumPy array.

    ValueError
        If either input has an invalid shape, contains no sequences, 
        has incompatible dimensions, or contains NaN or infinite values.
    """

    if not isinstance(
        training_sequences,
        np.ndarray,
    ):
        raise TypeError(
            "training_sequences must be a NumPy array."
        )

    if not isinstance(
        validation_sequences,
        np.ndarray,
    ):
        raise TypeError(
            "validation_sequences must be a NumPy array."
        )

    if training_sequences.ndim != 3:
        raise ValueError(
            "training_sequences must have shape (number_of_sequences, sequence_length, number_of_features). "
            f"Received shape {training_sequences.shape}."
        )

    if validation_sequences.ndim != 3:
        raise ValueError(
            "validation_sequences must have shape (number_of_sequences, sequence_length, number_of_features). "
            f"Received shape {validation_sequences.shape}."
        )

    if training_sequences.shape[0] == 0:
        raise ValueError(
            "training_sequences must contain at least one sequence."
        )

    if validation_sequences.shape[0] == 0:
        raise ValueError(
            "validation_sequences must contain at least one sequence."
        )

    if training_sequences.shape[1] == 0:
        raise ValueError(
            "training sequence length must be at least 1."
        )

    if validation_sequences.shape[1] == 0:
        raise ValueError(
            "validation sequence length must be at least 1."
        )

    if training_sequences.shape[2] == 0:
        raise ValueError(
            "training feature count must be at least 1."
        )

    if validation_sequences.shape[2] == 0:
        raise ValueError(
            "validation feature count must be at least 1."
        )

    if (
        training_sequences.shape[1:]
        != validation_sequences.shape[1:]
    ):
        raise ValueError(
            "Training and validation sequences must have the same sequence length and feature count. "
            f"Received {training_sequences.shape[1:]} and {validation_sequences.shape[1:]}."
        )

    if not np.isfinite(
        training_sequences
    ).all():
        raise ValueError(
            "training_sequences contains NaN or infinite values."
        )

    if not np.isfinite(
        validation_sequences
    ).all():
        raise ValueError(
            "validation_sequences contains NaN or infinite values."
        )

def validate_model_input_shape(
    model: Model,
    training_sequences: np.ndarray,
) -> None:
    """
    Validate that a sequence tensor matches the Keras model input shape.

    Parameters
    ----------
    model:
        TensorFlow/Keras autoencoder.

    training_sequences:
        Training sequences using shape:

        (
            number_of_sequences,
            sequence_length,
            number_of_features,
        )

    Raises
    ------
    ValueError
        If model has multiple inputs or its expected sequence length or feature count does not match training tensor.
    """

    expected_input_shape = model.input_shape

    if not isinstance(
        expected_input_shape,
        tuple,
    ):
        raise ValueError(
            "The training module currently supports single-input Keras models only."
        )

    if len(expected_input_shape) != 3:
        raise ValueError(
            "The model input must have shape (batch, sequence_length, number_of_features). "
            f"Received {expected_input_shape}."
        )

    expected_sequence_length = (
        expected_input_shape[1]
    )

    expected_number_of_features = (
        expected_input_shape[2]
    )

    actual_sequence_length = (
        training_sequences.shape[1]
    )

    actual_number_of_features = (
        training_sequences.shape[2]
    )

    if (
        expected_sequence_length is not None
        and expected_sequence_length
        != actual_sequence_length
    ):
        raise ValueError(
            "Training sequence length does not match the model input shape. "
            f"Model expects {expected_sequence_length}; "
            f"received {actual_sequence_length}."
        )

    if (
        expected_number_of_features is not None
        and expected_number_of_features
        != actual_number_of_features
    ):
        raise ValueError(
            "Training feature count does not match the model input shape. "
            f"Model expects {expected_number_of_features}; "
            f"received {actual_number_of_features}."
        )

def validate_compiled_model(
    model: Model,
) -> None:
    """
    Validate that a Keras model has been compiled.

    Keras Functional models may not expose an ``optimizer`` attribute before compilation. 
    ``getattr`` used instead of accessing ``model.optimizer`` directly.

    Parameters
    ----------
    model:
        TensorFlow/Keras model to validate.

    Raises
    ------
    ValueError
        If the model has not been compiled with both optimizer and loss.
    """

    optimizer = getattr(
        model,
        "optimizer",
        None,
    )

    loss = getattr(
        model,
        "loss",
        None,
    )

    if optimizer is None or loss is None:
        raise ValueError(
            "model must be compiled with an optimizer and loss before training."
        )

def build_early_stopping_callback(
    config: TCNTrainingConfig,
) -> EarlyStopping:
    """
    Build validation-loss early-stopping callback.

    Callback monitors validation loss and optionally restores model weights from epoch with lowest validation loss.

    Parameters
    ----------
    config:
        Training configuration.

    Returns
    -------
    EarlyStopping
        Configured Keras early-stopping callback.
    """

    config.validate()

    return EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=config.patience,
        restore_best_weights=(
            config.restore_best_weights
        ),
        verbose=config.verbose,
    )

def train_tcn_autoencoder(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    config: TCNTrainingConfig | None = None,
) -> History:
    """
    Train a TensorFlow/Keras TCN autoencoder.

    Autoencoder trained to reconstruct its own input:

        input  = training_sequences
        target = training_sequences

    Validation follows the same reconstruction objective:

        validation input  = validation_sequences
        validation target = validation_sequences

    Parameters
    ----------
    model:
        Compiled TensorFlow/Keras autoencoder.

    training_sequences:
        Scaled contiguous training sequences with shape:

        (
            number_of_training_sequences,
            sequence_length,
            number_of_features,
        )

    validation_sequences:
        Scaled contiguous validation sequences with shape:

        (
            number_of_validation_sequences,
            sequence_length,
            number_of_features,
        )

    config:
        Training configuration. If omitted, validated HPC proof-of-concept defaults are used.

    Returns
    -------
    History
        Keras training history containing at least ``loss`` and ``val_loss``.
    """

    if not isinstance(
        model,
        Model,
    ):
        raise TypeError(
            "model must be a TensorFlow/Keras Model."
        )

    validate_compiled_model(
        model=model
    )

    if config is None:
        config = TCNTrainingConfig()

    config.validate()

    validate_training_sequences(
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
    )

    validate_model_input_shape(
        model=model,
        training_sequences=training_sequences,
    )

    np.random.seed(
        config.random_seed
    )

    tf.random.set_seed(
        config.random_seed
    )

    early_stopping = (
        build_early_stopping_callback(
            config=config
        )
    )

    history = model.fit(
        training_sequences,
        training_sequences,
        validation_data=(
            validation_sequences,
            validation_sequences,
        ),
        epochs=config.epochs,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        callbacks=[
            early_stopping,
        ],
        verbose=config.verbose,
    )

    return history