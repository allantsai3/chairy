import datetime
import numpy as np
import tensorflow as tf
import trimesh
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path


class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self):
        return {
            'num_features': self.num_features,
            'l2reg': self.l2reg,
        }

    @classmethod
    def from_config(cls, config):
        return OrthogonalRegularizer(num_features=config['num_features'], l2reg=config['l2reg'])


def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])


def get_pointnet_model(num_points=2048):
    inputs = keras.Input(shape=(num_points, 3))

    x = tnet(inputs, 3)
    x = conv_bn(x, 32)
    x = conv_bn(x, 32)
    x = tnet(x, 32)
    x = conv_bn(x, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = layers.Dropout(0.3)(x)
    x = dense_bn(x, 128)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
    return model


def parse_dataset(data_dir, num_points=2048, test_size=1000):

    files = list(Path(data_dir).glob('**/*.obj'))
    rnd = np.random.default_rng(12345)
    rnd.shuffle(files)
    points = []
    labels = []

    for file in files:
        labels.append(0 if 'NonChair' in str(file.parent) else 1)
        points.append(trimesh.load(file).sample(num_points))

    train_points = points[:-test_size]
    train_labels = labels[:-test_size]
    test_points = points[-test_size:]
    test_labels = labels[-test_size:]

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels)
    )


def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


if __name__ == '__main__':
    NUM_POINTS = 2048
    BATCH_SIZE = 32

    model = get_pointnet_model(NUM_POINTS)

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        metrics=["acc"],
    )

    # train_points, test_points, train_labels, test_labels = parse_dataset('D:/Documents/CMPT 464/', NUM_POINTS)
    # np.save('train_points.npy', train_points)
    # np.save('test_points.npy', test_points)
    # np.save('train_labels.npy', train_labels)
    # np.save('test_labels.npy', test_labels)
    train_points = np.load('train_points.npy')
    test_points = np.load('test_points.npy')
    train_labels = np.load('train_labels.npy')
    test_labels = np.load('test_labels.npy')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

    train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
    test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

    # Define callbacks.
    log_dir = "logs/fit/pointnet" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "pointnet.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=15)

    epochs = 100
    model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb],
    )
