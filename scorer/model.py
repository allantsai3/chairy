import tensorflow as tf
import datetime

from tensorflow import keras
from tensorflow.keras import layers


def get_model(width=128, height=128, depth=128):
    """
    Build a 3D convolutional neural network model.
    Referenced from [this paper](https://arxiv.org/abs/2007.13224)
    """

    inputs = keras.Input((width, height, depth, 1), dtype=tf.float32)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)

    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


def train_model(model: keras.Model, dataset: tf.data.Dataset):
    initial_learning_rate = 0.0001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "3d_chair_classification.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=15)

    dataset = dataset.map(to_float, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.shuffle(buffer_size=len(dataset), reshuffle_each_iteration=False)
    test_size = int(len(dataset) * 0.1)
    test_dataset = dataset.take(test_size)
    test_dataset = test_dataset.batch(2)
    test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    train_dataset = dataset.skip(test_size)
    train_dataset = train_dataset.shuffle(len(train_dataset))
    train_dataset = train_dataset.batch(2)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    epochs = 100
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )


def to_float(volume, label):
    return tf.cast(volume, tf.float32), tf.cast(label, tf.float32)
