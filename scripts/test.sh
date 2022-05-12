#!/bin/bash


CHECKPOINTS_DIR=$1
DATA_DIR=$2

if [ -z "${CHECKPOINTS_DIR}" ] | [ -z "${DATA_DIR}" ]; then
    echo "Usage: test.sh <checkpoints_dir> <data_dir>"
    exit -1
fi

ROOT=$(realpath $(dirname $0)/..)
cd ${ROOT}

TEST_CHECKPOINT_DIR=${CHECKPOINTS_DIR}/$(ls $CHECKPOINTS_DIR -t | head -n 1)

TIMESTAMP=$(date --utc +%FT%T.%3NZ)
OUT_DIRNAME="$(basename ${TEST_CHECKPOINT_DIR})-${TIMESTAMP}"

python test.py \
    --batch_size 16 --gpu 0 \
    --checkpoint ${TEST_CHECKPOINT_DIR}/model_best.pth \
    --boxes_transcripts ${DATA_DIR}/boxes_and_transcripts-test \
    --images_path ${DATA_DIR}/images \
    --output_folder ${DATA_DIR}/test_output/${OUT_DIRNAME}
