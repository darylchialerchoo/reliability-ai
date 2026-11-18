import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Dropout,
    Activation,
    Add,
    AveragePooling1D,
    UpSampling1D,
    Lambda,
)
from tensorflow.keras.models import Model

def causal_conv1d(
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int = 1,
    padding: str = "causal",
    name: str | None = None,
):
    """
    Apply a causal, dilated Conv1D layer.
    """

    return Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=True,
        name=name,
    )(x)

def tcn_residual_block(
    x,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    block_name: str,
):
    """
    Build a TCN residual block containing: 

    1. Causal dilated Conv1D
    2. ReLU activation
    3. Dropout
    4. Causal dilated Conv1D
    5. ReLU activation
    6. Dropout
    7. Residual connection

    A 1x1 Conv1D is applied to the residual path when the number of input channels does not match the number of TCN filters.
    """

    residual = x

    # ========================================================
    # First dilated convolution
    # ========================================================

    x = causal_conv1d(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        name=f"{block_name}_dilated_conv_1",
    )

    x = Activation(
        "relu",
        name=f"{block_name}_relu_1",
    )(x)

    x = Dropout(
        rate=dropout_rate,
        name=f"{block_name}_dropout_1",
    )(x)

    # ========================================================
    # Second dilated convolution
    # ========================================================

    x = causal_conv1d(
        x=x,
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="causal",
        name=f"{block_name}_dilated_conv_2",
    )

    x = Activation(
        "relu",
        name=f"{block_name}_relu_2",
    )(x)

    x = Dropout(
        rate=dropout_rate,
        name=f"{block_name}_dropout_2",
    )(x)

    # ========================================================
    # Residual connection
    # ========================================================

    input_channels = residual.shape[-1]

    if input_channels != filters:
        residual = Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            name=f"{block_name}_residual_1x1_conv",
        )(residual)

    x = Add(
        name=f"{block_name}_residual_add",
    )(
        [
            x,
            residual,
        ]
    )

    return x

def tcn_stack(
    x,
    dilations: tuple[int, ...],
    filters: int,
    kernel_size: int,
    dropout_rate: float,
    stack_name: str,
):
    """
    Build a stack of TCN residual blocks, with a residual block for each dilation rate. 
    """

    for dilation_rate in dilations:
        x = tcn_residual_block(
            x=x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            block_name=(
                f"{stack_name}_residual_block_d{dilation_rate}"
            ),
        )

    return x

def build_tcn_autoencoder(
    sequence_length: int,
    number_of_features: int,
    dilations: tuple[int, ...] = (1, 2, 4, 8),
    kernel_size: int = 2,
    number_of_filters: int = 16,
    bottleneck_filters: int = 8,
    pool_size: int = 1,
    dropout_rate: float = 0.2,
    learning_rate: float = 1e-3,
) -> Model:
    """
    Build and compile a Temporal Convolutional Network Autoencoder.

    Parameters
    ----------
    sequence_length:
        Number of time steps in each input sequence.

    number_of_features:
        Number of input features at each time step.

    dilations:
        Dilation rates used by the TCN residual blocks.

    kernel_size:
        Kernel size of the dilated Conv1D layers.

    number_of_filters:
        Number of filters used in the TCN residual blocks.

    bottleneck_filters:
        Number of filters in the encoder bottleneck.

    pool_size:
        Temporal compression factor used by the encoder and decoder.

    dropout_rate:
        Dropout rate applied inside each TCN residual block.

    learning_rate:
        Learning rate used by the Adam optimizer.

    Architecture
    ------------
    Encoder:

        Input
            -> TCN residual stack
            -> 1x1 Conv bottleneck
            -> temporal average pooling

    Decoder:

        Encoded representation
            -> sample-and-hold upsampling
            -> crop to original sequence length
            -> TCN residual stack
            -> 1x1 Conv
            -> reconstructed features

    Notes
    -----
    The input and output tensor shape is:

        (batch, sequence_length, number_of_features)
    """

    # ========================================================
    # Input
    # ========================================================

    inputs = Input(
        shape=(
            sequence_length,
            number_of_features,
        ),
        name="encoder_input",
    )

    # ========================================================
    # Encoder
    # ========================================================

    x = tcn_stack(
        x=inputs,
        dilations=dilations,
        filters=number_of_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        stack_name="encoder_tcn",
    )

    x = Conv1D(
        filters=bottleneck_filters,
        kernel_size=1,
        padding="same",
        activation="linear",
        name="encoder_bottleneck_conv1d_k1",
    )(x)

    encoded = AveragePooling1D(
        pool_size=pool_size,
        strides=pool_size,
        padding="same",
        name="encoder_temporal_average_pooling",
    )(x)

    # ========================================================
    # Decoder
    # ========================================================

    x = UpSampling1D(
        size=pool_size,
        name="decoder_sample_and_hold_upsampling",
    )(encoded)

    x = Lambda(
        lambda tensor: tensor[:, :sequence_length, :],
        name="decoder_crop_to_original_sequence_length",
    )(x)

    x = tcn_stack(
        x=x,
        dilations=dilations,
        filters=number_of_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        stack_name="decoder_tcn",
    )

    outputs = Conv1D(
        filters=number_of_features,
        kernel_size=1,
        padding="same",
        activation="linear",
        name="decoder_output_conv1d_k1",
    )(x)

    # ========================================================
    # Model
    # ========================================================

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="dgs_tcn_autoencoder",
    )

    # ========================================================
    # Compile
    # ========================================================

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="mse",
    )

    return model