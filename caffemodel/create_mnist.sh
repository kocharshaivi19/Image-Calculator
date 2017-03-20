#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

# specify the cafferoot here
LMDB=data/hdf5
DATA=data/data
BUILD=data/build

BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $LMDB/mnist_train_${BACKEND}
rm -rf $LMDB/mnist_test_${BACKEND}

$BUILD/convert_mnist_data.bin $DATA/train-images-idx3-ubyte \
  $DATA/train-labels-idx1-ubyte $LMDB/mnist_train_${BACKEND} --backend=${BACKEND}
$BUILD/convert_mnist_data.bin $DATA/t10k-images-idx3-ubyte \
  $DATA/t10k-labels-idx1-ubyte $LMDB/mnist_test_${BACKEND} --backend=${BACKEND}

echo "Done."
