#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/ponu/DATA/Places205_resize/images256
DATA=/media/ponu/DATA/Places205_resize/images256
TOOLS=/home/ponu/caffe/build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/Places_mean.binaryproto

echo "Done."
