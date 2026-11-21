import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

from reliability_ai.models import (
    build_tcn_autoencoder,
)
from reliability_ai.training import (
    TCNTrainingConfig,
    build_early_stopping_callback,
    train_tcn_autoencoder,
    validate_compiled_model,
    validate_model_input_shape,
    validate_training_sequences,
)


# ============================================================
# Test configuration
# ============================================================

SEQUENCE_LENGTH = 8
NUMBER_OF_FEATURES = 3

NUMBER_OF_TRAINING_SEQUENCES = 24
NUMBER_OF_VALIDATION_SEQUENCES = 8

RANDOM_SEED = 42


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def training_sequences() -> np.ndarray:
    """
    Create deterministic synthetic training sequences.
    """

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    return random_generator.normal(
        size=(
            NUMBER_OF_TRAINING_SEQUENCES,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        )
    ).astype(
        np.float32
    )


@pytest.fixture
def validation_sequences() -> np.ndarray:
    """
    Create deterministic synthetic validation sequences.
    """

    random_generator = np.random.default_rng(
        RANDOM_SEED + 1
    )

    return random_generator.normal(
        size=(
            NUMBER_OF_VALIDATION_SEQUENCES,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        )
    ).astype(
        np.float32
    )


@pytest.fixture
def model() -> Model:
    """
    Build a small TCN autoencoder for fast unit testing.
    """

    tf.keras.backend.clear_session()

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    return build_tcn_autoencoder(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=(
            NUMBER_OF_FEATURES
        ),
        dilations=(
            1,
            2,
        ),
        kernel_size=2,
        number_of_filters=4,
        bottleneck_filters=2,
        pool_size=1,
        dropout_rate=0.0,
        learning_rate=1e-3,
    )


@pytest.fixture
def training_config() -> TCNTrainingConfig:
    """
    Create a short training configuration for tests.
    """

    return TCNTrainingConfig(
        epochs=2,
        batch_size=8,
        patience=1,
        restore_best_weights=True,
        shuffle=True,
        verbose=0,
        random_seed=RANDOM_SEED,
    )


def build_uncompiled_model() -> Model:
    """
    Build an uncompiled Keras Functional model.
    """

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        )
    )

    outputs = tf.keras.layers.Dense(
        NUMBER_OF_FEATURES
    )(inputs)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
    )


# ============================================================
# Configuration tests
# ============================================================

def test_default_configuration_matches_validated_poc() -> None:
    """
    Default training parameters should preserve the
    validated HPC proof-of-concept settings.
    """

    config = TCNTrainingConfig()

    assert config.epochs == 500
    assert config.batch_size == 128
    assert config.patience == 5

    assert config.restore_best_weights is True
    assert config.shuffle is True

    assert config.verbose == 1
    assert config.random_seed == 42


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
        "expected_message",
    ),
    [
        (
            "epochs",
            0,
            "epochs must be at least 1",
        ),
        (
            "batch_size",
            0,
            "batch_size must be at least 1",
        ),
        (
            "patience",
            -1,
            "patience must be at least 0",
        ),
        (
            "verbose",
            3,
            "verbose must be 0, 1, or 2",
        ),
        (
            "random_seed",
            -1,
            "random_seed must be at least 0",
        ),
    ],
)
def test_invalid_configuration_raises_error(
    field_name: str,
    invalid_value: int,
    expected_message: str,
) -> None:
    """
    Invalid training settings should fail before model
    training begins.
    """

    configuration_values = {
        "epochs": 2,
        "batch_size": 8,
        "patience": 1,
        "restore_best_weights": True,
        "shuffle": True,
        "verbose": 0,
        "random_seed": RANDOM_SEED,
    }

    configuration_values[
        field_name
    ] = invalid_value

    config = TCNTrainingConfig(
        **configuration_values
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        config.validate()


# ============================================================
# Early-stopping tests
# ============================================================

def test_builds_expected_early_stopping_callback(
    training_config: TCNTrainingConfig,
) -> None:
    """
    The trainer should monitor validation loss and restore
    the best model weights.
    """

    callback = (
        build_early_stopping_callback(
            config=training_config
        )
    )

    assert isinstance(
        callback,
        EarlyStopping,
    )

    assert callback.monitor == "val_loss"
    assert callback.mode == "min"

    assert (
        callback.patience
        == training_config.patience
    )

    assert (
        callback.restore_best_weights
        is True
    )


def test_early_stopping_uses_configured_restore_setting() -> None:
    """
    The restore-best-weights option should be configurable.
    """

    config = TCNTrainingConfig(
        epochs=2,
        batch_size=8,
        patience=1,
        restore_best_weights=False,
        shuffle=True,
        verbose=0,
        random_seed=RANDOM_SEED,
    )

    callback = (
        build_early_stopping_callback(
            config=config
        )
    )

    assert (
        callback.restore_best_weights
        is False
    )


# ============================================================
# Sequence-validation tests
# ============================================================

def test_valid_training_sequences_pass_validation(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> None:
    """
    Compatible finite three-dimensional arrays should pass.
    """

    validate_training_sequences(
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
    )


def test_training_sequences_must_be_numpy_array(
    validation_sequences: np.ndarray,
) -> None:
    """
    Non-NumPy training input should raise a clear error.
    """

    training_sequences = [
        [
            [1.0],
        ],
    ]

    with pytest.raises(
        TypeError,
        match="training_sequences must be a NumPy array",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_validation_sequences_must_be_numpy_array(
    training_sequences: np.ndarray,
) -> None:
    """
    Non-NumPy validation input should raise a clear error.
    """

    validation_sequences = [
        [
            [1.0],
        ],
    ]

    with pytest.raises(
        TypeError,
        match="validation_sequences must be a NumPy array",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_training_sequences_must_be_three_dimensional(
    validation_sequences: np.ndarray,
) -> None:
    """
    Training data must follow the (N, T, F) contract.
    """

    training_sequences = np.zeros(
        (
            10,
            NUMBER_OF_FEATURES,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="training_sequences must have shape",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_validation_sequences_must_be_three_dimensional(
    training_sequences: np.ndarray,
) -> None:
    """
    Validation data must follow the (N, T, F) contract.
    """

    validation_sequences = np.zeros(
        (
            10,
            NUMBER_OF_FEATURES,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="validation_sequences must have shape",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_empty_training_sequences_raise_error(
    validation_sequences: np.ndarray,
) -> None:
    """
    A model cannot train without training sequences.
    """

    training_sequences = np.empty(
        (
            0,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least one sequence",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_empty_validation_sequences_raise_error(
    training_sequences: np.ndarray,
) -> None:
    """
    Validation-loss early stopping requires validation data.
    """

    validation_sequences = np.empty(
        (
            0,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="at least one sequence",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_sequence_length_mismatch_raises_error(
    training_sequences: np.ndarray,
) -> None:
    """
    Training and validation sequence lengths must match.
    """

    validation_sequences = np.zeros(
        (
            NUMBER_OF_VALIDATION_SEQUENCES,
            SEQUENCE_LENGTH + 1,
            NUMBER_OF_FEATURES,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="same sequence length and feature count",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_feature_count_mismatch_raises_error(
    training_sequences: np.ndarray,
) -> None:
    """
    Training and validation feature counts must match.
    """

    validation_sequences = np.zeros(
        (
            NUMBER_OF_VALIDATION_SEQUENCES,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES + 1,
        ),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="same sequence length and feature count",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_nan_in_training_sequences_raises_error(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> None:
    """
    NaN values should not reach TensorFlow training.
    """

    training_sequences = (
        training_sequences.copy()
    )

    training_sequences[
        0,
        0,
        0,
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


def test_infinite_validation_value_raises_error(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
) -> None:
    """
    Infinite validation values should fail before fitting.
    """

    validation_sequences = (
        validation_sequences.copy()
    )

    validation_sequences[
        0,
        0,
        0,
    ] = np.inf

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_training_sequences(
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
        )


# ============================================================
# Model-compilation tests
# ============================================================

def test_compiled_model_passes_validation(
    model: Model,
) -> None:
    """
    A compiled model should pass compilation validation.
    """

    validate_compiled_model(
        model=model
    )


def test_uncompiled_model_raises_clear_error() -> None:
    """
    An uncompiled Functional model may not have an
    optimizer attribute. The validator must still raise
    a clear ValueError.
    """

    uncompiled_model = (
        build_uncompiled_model()
    )

    with pytest.raises(
        ValueError,
        match="compiled with an optimizer and loss",
    ):
        validate_compiled_model(
            model=uncompiled_model
        )


# ============================================================
# Model-shape tests
# ============================================================

def test_model_input_shape_matches_training_sequences(
    model: Model,
    training_sequences: np.ndarray,
) -> None:
    """
    Compatible model and sequence shapes should pass.
    """

    validate_model_input_shape(
        model=model,
        training_sequences=training_sequences,
    )


def test_model_sequence_length_mismatch_raises_error(
    training_sequences: np.ndarray,
) -> None:
    """
    Model sequence length must match the generated sequences.
    """

    incompatible_model = (
        build_tcn_autoencoder(
            sequence_length=(
                SEQUENCE_LENGTH + 1
            ),
            number_of_features=(
                NUMBER_OF_FEATURES
            ),
            dilations=(1,),
            number_of_filters=4,
            bottleneck_filters=2,
            dropout_rate=0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="sequence length does not match",
    ):
        validate_model_input_shape(
            model=incompatible_model,
            training_sequences=training_sequences,
        )


def test_model_feature_count_mismatch_raises_error(
    training_sequences: np.ndarray,
) -> None:
    """
    Model feature count must match the generated sequences.
    """

    incompatible_model = (
        build_tcn_autoencoder(
            sequence_length=SEQUENCE_LENGTH,
            number_of_features=(
                NUMBER_OF_FEATURES + 1
            ),
            dilations=(1,),
            number_of_filters=4,
            bottleneck_filters=2,
            dropout_rate=0.0,
        )
    )

    with pytest.raises(
        ValueError,
        match="feature count does not match",
    ):
        validate_model_input_shape(
            model=incompatible_model,
            training_sequences=training_sequences,
        )


# ============================================================
# Training tests
# ============================================================

def test_training_returns_keras_history(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    Training should return the standard Keras History object.
    """

    history = train_tcn_autoencoder(
        model=model,
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
        config=training_config,
    )

    assert isinstance(
        history,
        tf.keras.callbacks.History,
    )


def test_training_history_contains_loss_and_validation_loss(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    Autoencoder training should report reconstruction loss
    for both training and validation data.
    """

    history = train_tcn_autoencoder(
        model=model,
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
        config=training_config,
    )

    assert "loss" in history.history
    assert "val_loss" in history.history

    assert len(
        history.history["loss"]
    ) >= 1

    assert len(
        history.history["val_loss"]
    ) >= 1


def test_training_losses_are_finite(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    Training and validation losses must remain finite.
    """

    history = train_tcn_autoencoder(
        model=model,
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
        config=training_config,
    )

    training_losses = np.asarray(
        history.history["loss"]
    )

    validation_losses = np.asarray(
        history.history["val_loss"]
    )

    assert np.isfinite(
        training_losses
    ).all()

    assert np.isfinite(
        validation_losses
    ).all()


def test_training_does_not_exceed_configured_epochs(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    Early stopping may reduce the number of epochs, but
    training must never exceed the configured maximum.
    """

    history = train_tcn_autoencoder(
        model=model,
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
        config=training_config,
    )

    assert len(
        history.history["loss"]
    ) <= training_config.epochs


def test_training_updates_at_least_one_model_weight(
    model: Model,
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    At least one trainable model weight should change
    after optimization.
    """

    weights_before = [
        weight.copy()
        for weight in model.get_weights()
    ]

    train_tcn_autoencoder(
        model=model,
        training_sequences=training_sequences,
        validation_sequences=(
            validation_sequences
        ),
        config=training_config,
    )

    weights_after = model.get_weights()

    weight_changed = any(
        not np.allclose(
            before,
            after,
        )
        for before, after in zip(
            weights_before,
            weights_after,
        )
    )

    assert weight_changed


def test_model_must_be_compiled(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    The trainer should reject an uncompiled Keras model
    with a clear ValueError.
    """

    uncompiled_model = (
        build_uncompiled_model()
    )

    with pytest.raises(
        ValueError,
        match="compiled with an optimizer and loss",
    ):
        train_tcn_autoencoder(
            model=uncompiled_model,
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
            config=training_config,
        )


def test_model_must_be_keras_model(
    training_sequences: np.ndarray,
    validation_sequences: np.ndarray,
    training_config: TCNTrainingConfig,
) -> None:
    """
    A non-Keras object should be rejected clearly.
    """

    with pytest.raises(
        TypeError,
        match="TensorFlow/Keras Model",
    ):
        train_tcn_autoencoder(
            model="not-a-model",
            training_sequences=training_sequences,
            validation_sequences=(
                validation_sequences
            ),
            config=training_config,
        )