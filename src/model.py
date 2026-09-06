"""The deployed occupancy-grid model: early Dense fusion, separable conv refinement."""

import os

# Set before TensorFlow loads, or they have no effect.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def build_grid_model(
    input_shape=(6, 3, 47, 1), output_shape=(18, 12, 1), early_fusion_channels=2
):
    """Build the occupancy grid model.

    `input_shape` is (radars, antennas, range bins, channels). Only the Flatten
    feeding the early-fusion Dense sees the channel count, so an extra channel
    costs 216,576 parameters and nothing else: 330,851 -> 547,427 (~578 KB
    flash). A third reaches 764,003 (~807 KB), over the limit, which is why two
    channels is the hard ceiling on all 6 radars.
    """
    inputs = tf.keras.Input(shape=input_shape, name="radar_input")

    # Early fusion is a Dense, not a conv, and that is a geometric argument: the
    # length-6 axis indexes radars distributed around the room, so it carries no
    # translation structure for a kernel to exploit. The mapping from six range
    # profiles to one room is arbitrary and global, so it is learned outright.
    x = tf.keras.layers.Flatten()(inputs)

    # Ordering everywhere below is Linear -> BatchNorm -> ReLU. TFLite folds a BN
    # into the preceding op's weights only when the path between them is purely
    # linear, so the nonlinearity must be its own layer AFTER the BN, never an
    # `activation=` on the conv/dense itself -- that gives linear -> relu -> BN,
    # which folds in neither direction and ships the BN as runtime MUL+ADD.
    x = tf.keras.layers.Dense(256, name="early_fusion_hidden")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_early_fusion")(x)
    x = tf.keras.layers.ReLU(name="relu_early_fusion")(x)
    x = tf.keras.layers.Dense(
        output_shape[0] * output_shape[1] * early_fusion_channels,
        activation="relu",
        name="early_fusion_proj",
    )(x)
    x = tf.keras.layers.Reshape(
        (output_shape[0], output_shape[1], early_fusion_channels)
    )(x)

    for i, filters in enumerate((16, 32, 16), start=1):
        x = tf.keras.layers.SeparableConv2D(
            filters, (3, 3), padding="same", name=f"sepconv_{i}"
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
        x = tf.keras.layers.ReLU(name=f"relu_{i}")(x)

    # Linear head, not sigmoid: MSE + sigmoid on ~98%-zero targets is prone to
    # severe vanishing gradients. `submission/code.py` therefore thresholds the
    # raw output and applies no activation of its own -- the two must agree.
    outputs = tf.keras.layers.Conv2D(
        output_shape[2],
        (3, 3),
        padding="same",
        activation="linear",
        name="heatmap_output",
    )(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="occupancy_grid_model")


if __name__ == "__main__":
    build_grid_model().summary()
