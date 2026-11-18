import numpy as np
import pytest
import tensorflow as tf

from reliability_ai.models import build_tcn_autoencoder


# ============================================================
# Test configuration
# ============================================================

SEQUENCE_LENGTH = 48
NUMBER_OF_FEATURES = 6

DILATIONS = (1, 2, 4, 8)
KERNEL_SIZE = 2
NUMBER_OF_FILTERS = 16
BOTTLENECK_FILTERS = 8
POOL_SIZE = 1
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def model() -> tf.keras.Model:
    """
    Construct a fresh TCN autoencoder for each test.
    """

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    return build_tcn_autoencoder(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=NUMBER_OF_FEATURES,
        dilations=DILATIONS,
        kernel_size=KERNEL_SIZE,
        number_of_filters=NUMBER_OF_FILTERS,
        bottleneck_filters=BOTTLENECK_FILTERS,
        pool_size=POOL_SIZE,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )


# ============================================================
# Basic model tests
# ============================================================

def test_build_returns_keras_model(
    model: tf.keras.Model,
) -> None:
    """
    The model factory must return a TensorFlow/Keras Model.
    """

    assert isinstance(
        model,
        tf.keras.Model,
    )


def test_model_name(
    model: tf.keras.Model,
) -> None:
    """
    The expected model name should be preserved.
    """

    assert model.name == "dgs_tcn_autoencoder"


# ============================================================
# Input and output shape tests
# ============================================================

def test_input_and_output_shapes(
    model: tf.keras.Model,
) -> None:
    """
    The autoencoder must reconstruct the same tensor shape
    that it receives.
    """

    assert model.input_shape == (
        None,
        SEQUENCE_LENGTH,
        NUMBER_OF_FEATURES,
    )

    assert model.output_shape == (
        None,
        SEQUENCE_LENGTH,
        NUMBER_OF_FEATURES,
    )


def test_forward_pass_shape(
    model: tf.keras.Model,
) -> None:
    """
    A real batch must pass through the model without
    changing its shape.
    """

    batch_size = 4

    inputs = tf.random.normal(
        shape=(
            batch_size,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        ),
        seed=42,
    )

    reconstructed = model(
        inputs,
        training=False,
    )

    assert reconstructed.shape == inputs.shape


def test_forward_pass_contains_finite_values(
    model: tf.keras.Model,
) -> None:
    """
    Model inference must not produce NaN or infinite values.
    """

    inputs = tf.random.normal(
        shape=(
            4,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        ),
        seed=42,
    )

    reconstructed = model(
        inputs,
        training=False,
    )

    reconstructed_array = reconstructed.numpy()

    assert np.isfinite(
        reconstructed_array
    ).all()


# ============================================================
# Architecture tests
# ============================================================

def test_model_has_expected_encoder_layers(
    model: tf.keras.Model,
) -> None:
    """
    Check representative encoder layers from the
    validated architecture.
    """

    expected_layer_names = {
        "encoder_input",
        "encoder_bottleneck_conv1d_k1",
        "encoder_temporal_average_pooling",
    }

    actual_layer_names = {
        layer.name
        for layer in model.layers
    }

    assert expected_layer_names.issubset(
        actual_layer_names
    )


def test_model_has_expected_decoder_layers(
    model: tf.keras.Model,
) -> None:
    """
    Check representative decoder layers from the
    validated architecture.
    """

    expected_layer_names = {
        "decoder_sample_and_hold_upsampling",
        "decoder_crop_to_original_sequence_length",
        "decoder_output_conv1d_k1",
    }

    actual_layer_names = {
        layer.name
        for layer in model.layers
    }

    assert expected_layer_names.issubset(
        actual_layer_names
    )


@pytest.mark.parametrize(
    "dilation",
    DILATIONS,
)
def test_encoder_contains_each_dilation(
    model: tf.keras.Model,
    dilation: int,
) -> None:
    """
    The encoder must contain one residual block for every
    configured dilation rate.
    """

    expected_layer_name = (
        "encoder_tcn_"
        f"residual_block_d{dilation}_"
        "residual_add"
    )

    layer = model.get_layer(
        expected_layer_name
    )

    assert isinstance(
        layer,
        tf.keras.layers.Add,
    )


@pytest.mark.parametrize(
    "dilation",
    DILATIONS,
)
def test_decoder_contains_each_dilation(
    model: tf.keras.Model,
    dilation: int,
) -> None:
    """
    The decoder must contain one residual block for every
    configured dilation rate.
    """

    expected_layer_name = (
        "decoder_tcn_"
        f"residual_block_d{dilation}_"
        "residual_add"
    )

    layer = model.get_layer(
        expected_layer_name
    )

    assert isinstance(
        layer,
        tf.keras.layers.Add,
    )


# ============================================================
# Convolution tests
# ============================================================

@pytest.mark.parametrize(
    "dilation",
    DILATIONS,
)
@pytest.mark.parametrize(
    "convolution_number",
    (1, 2),
)
def test_encoder_dilated_convolutions(
    model: tf.keras.Model,
    dilation: int,
    convolution_number: int,
) -> None:
    """
    Encoder TCN convolutions must be standard Conv1D layers
    with the expected causal padding and dilation rate.
    """

    layer_name = (
        "encoder_tcn_"
        f"residual_block_d{dilation}_"
        f"dilated_conv_{convolution_number}"
    )

    layer = model.get_layer(
        layer_name
    )

    assert isinstance(
        layer,
        tf.keras.layers.Conv1D,
    )

    assert layer.padding == "causal"

    assert layer.dilation_rate == (
        dilation,
    )

    assert layer.kernel_size == (
        KERNEL_SIZE,
    )

    assert (
        layer.filters
        == NUMBER_OF_FILTERS
    )


@pytest.mark.parametrize(
    "dilation",
    DILATIONS,
)
@pytest.mark.parametrize(
    "convolution_number",
    (1, 2),
)
def test_decoder_dilated_convolutions(
    model: tf.keras.Model,
    dilation: int,
    convolution_number: int,
) -> None:
    """
    Decoder TCN convolutions must be standard Conv1D layers
    with the expected causal padding and dilation rate.
    """

    layer_name = (
        "decoder_tcn_"
        f"residual_block_d{dilation}_"
        f"dilated_conv_{convolution_number}"
    )

    layer = model.get_layer(
        layer_name
    )

    assert isinstance(
        layer,
        tf.keras.layers.Conv1D,
    )

    assert layer.padding == "causal"

    assert layer.dilation_rate == (
        dilation,
    )

    assert layer.kernel_size == (
        KERNEL_SIZE,
    )

    assert (
        layer.filters
        == NUMBER_OF_FILTERS
    )


def test_number_of_dilated_convolutions(
    model: tf.keras.Model,
) -> None:
    """
    There should be two dilated Conv1D layers per residual
    block in both the encoder and decoder.

    For four dilation rates:

        4 dilations
        x 2 convolutions
        x 2 stacks
        = 16 dilated convolutions
    """

    dilated_convolution_names = []

    for stack_name in (
        "encoder_tcn",
        "decoder_tcn",
    ):
        for dilation in DILATIONS:
            for convolution_number in (
                1,
                2,
            ):
                dilated_convolution_names.append(
                    (
                        f"{stack_name}_"
                        f"residual_block_d{dilation}_"
                        f"dilated_conv_{convolution_number}"
                    )
                )

    assert len(
        dilated_convolution_names
    ) == 16

    for layer_name in dilated_convolution_names:
        layer = model.get_layer(
            layer_name
        )

        assert isinstance(
            layer,
            tf.keras.layers.Conv1D,
        )


# ============================================================
# Bottleneck and output tests
# ============================================================

def test_bottleneck_filter_count(
    model: tf.keras.Model,
) -> None:
    """
    The encoder bottleneck must use the requested number
    of filters.
    """

    bottleneck_layer = model.get_layer(
        "encoder_bottleneck_conv1d_k1"
    )

    assert isinstance(
        bottleneck_layer,
        tf.keras.layers.Conv1D,
    )

    assert (
        bottleneck_layer.filters
        == BOTTLENECK_FILTERS
    )

    assert bottleneck_layer.kernel_size == (
        1,
    )


def test_pool_size(
    model: tf.keras.Model,
) -> None:
    """
    The encoder pooling layer must use the configured
    temporal pool size.
    """

    pooling_layer = model.get_layer(
        "encoder_temporal_average_pooling"
    )

    assert isinstance(
        pooling_layer,
        tf.keras.layers.AveragePooling1D,
    )

    assert pooling_layer.pool_size == (
        POOL_SIZE,
    )


def test_upsampling_size(
    model: tf.keras.Model,
) -> None:
    """
    The decoder upsampling layer must mirror the configured
    encoder pool size.
    """

    upsampling_layer = model.get_layer(
        "decoder_sample_and_hold_upsampling"
    )

    assert isinstance(
        upsampling_layer,
        tf.keras.layers.UpSampling1D,
    )

    assert upsampling_layer.size == (
        POOL_SIZE
    )


def test_output_filter_count(
    model: tf.keras.Model,
) -> None:
    """
    The final Conv1D layer must generate one output for
    every input feature.
    """

    output_layer = model.get_layer(
        "decoder_output_conv1d_k1"
    )

    assert isinstance(
        output_layer,
        tf.keras.layers.Conv1D,
    )

    assert (
        output_layer.filters
        == NUMBER_OF_FEATURES
    )

    assert output_layer.kernel_size == (
        1,
    )


# ============================================================
# Compilation tests
# ============================================================

def test_model_is_compiled_with_mse(
    model: tf.keras.Model,
) -> None:
    """
    Preserve the mean-squared-error reconstruction objective.
    """

    assert model.loss == "mse"


def test_model_uses_adam_optimizer(
    model: tf.keras.Model,
) -> None:
    """
    The model must use the Adam optimizer.
    """

    assert isinstance(
        model.optimizer,
        tf.keras.optimizers.Adam,
    )


def test_model_uses_requested_learning_rate(
    model: tf.keras.Model,
) -> None:
    """
    The Adam optimizer must use the requested learning rate.
    """

    actual_learning_rate = float(
        tf.keras.backend.get_value(
            model.optimizer.learning_rate
        )
    )

    assert actual_learning_rate == pytest.approx(
        LEARNING_RATE,
    )


# ============================================================
# Training test
# ============================================================

def test_one_training_batch_runs(
    model: tf.keras.Model,
) -> None:
    """
    Verify that the compiled autoencoder can complete
    one optimization step.
    """

    tf.keras.utils.set_random_seed(42)

    inputs = tf.random.normal(
        shape=(
            8,
            SEQUENCE_LENGTH,
            NUMBER_OF_FEATURES,
        ),
        seed=42,
    )

    loss = model.train_on_batch(
        inputs,
        inputs,
    )

    loss_value = float(
        np.asarray(loss)
    )

    assert np.isfinite(
        loss_value
    )

    assert loss_value >= 0.0


# ============================================================
# Reusability tests
# ============================================================

def test_different_feature_count_changes_input_and_output_shape() -> None:
    """
    The autoencoder must support a different number of
    equipment features without modifying library source code.
    """

    alternative_feature_count = 10

    alternative_model = build_tcn_autoencoder(
        sequence_length=SEQUENCE_LENGTH,
        number_of_features=alternative_feature_count,
        dilations=DILATIONS,
        kernel_size=KERNEL_SIZE,
        number_of_filters=NUMBER_OF_FILTERS,
        bottleneck_filters=BOTTLENECK_FILTERS,
        pool_size=POOL_SIZE,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    assert alternative_model.input_shape == (
        None,
        SEQUENCE_LENGTH,
        alternative_feature_count,
    )

    assert alternative_model.output_shape == (
        None,
        SEQUENCE_LENGTH,
        alternative_feature_count,
    )


def test_different_sequence_length_changes_input_and_output_shape() -> None:
    """
    The autoencoder must support a different sequence length
    without modifying library source code.
    """

    alternative_sequence_length = 24

    alternative_model = build_tcn_autoencoder(
        sequence_length=alternative_sequence_length,
        number_of_features=NUMBER_OF_FEATURES,
        dilations=DILATIONS,
        kernel_size=KERNEL_SIZE,
        number_of_filters=NUMBER_OF_FILTERS,
        bottleneck_filters=BOTTLENECK_FILTERS,
        pool_size=POOL_SIZE,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    assert alternative_model.input_shape == (
        None,
        alternative_sequence_length,
        NUMBER_OF_FEATURES,
    )

    assert alternative_model.output_shape == (
        None,
        alternative_sequence_length,
        NUMBER_OF_FEATURES,
    )