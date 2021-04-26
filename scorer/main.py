from model import get_model, train_model
from pathlib import Path
from pprint import pprint
import binvox_rw
import argparse
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

def is_chair(path: Path) -> bool:
    return 'nonchair' not in path.parent.name


def read_binvox(path: Path) -> tuple:
    with path.open('rb') as f:
        model = binvox_rw.read_as_3d_array(f)
        return model.data, is_chair(path)


def batch_read_binvox(paths: list) -> tuple:
    if len(paths) == 0:
        raise Exception("No binvox file provided.")

    first, first_label = read_binvox(paths[0])
    if len(paths) == 1:
        return first, first_label

    shape = (len(paths),) + tuple(first.shape)
    volumes = np.empty(shape, dtype=np.bool)
    labels = np.zeros((len(paths), 1), dtype=np.bool)
    volumes[0] = first.data
    labels[0] = is_chair(paths[0])
    for i, path in enumerate(paths[1:]):
        volume, label = read_binvox(path)
        volumes[i + 1] = volume
        labels[i + 1] = label
        # if voxels:
        # else:
        #     print("Warning: Empty voxels at \"{path}\"")
        #     volumes[i + 1] = np.zeros(tuple(first.dims))

    return volumes, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run 3D neural network')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory of chair voxel files')
    parser.add_argument('--load', type=str, default='3d_chair_classification.h5',
                        help='Path to load dataset from')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Run model in train mode')

    args = parser.parse_args()
    chair_paths = []
    nonchair_paths = []

    if not args.dir and not args.train:
        print('Need to provide volume directory for evaluation.')
        exit(-1)

    if args.train and not args.dir:
        dataset = tfds.load('chairy_dataset',
                            split=['train'],
                            as_supervised=True)[0]
    else:
        paths = list(Path(args.dir).glob('**/*.binvox'))
        rnd = np.random.default_rng(12345)
        rnd.shuffle(paths)
        volumes, labels = batch_read_binvox(paths[:3000])
        volumes = np.expand_dims(volumes, axis=4)
        dataset = tf.data.Dataset.from_tensor_slices((volumes, labels))

    model = get_model()
    if args.train:
        model.summary()
        train_model(model, dataset)
    else:
        model.load_weights(args.load)
        result = model.predict(dataset)
        pprint(result)