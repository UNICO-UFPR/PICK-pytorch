#!/bin/bash

DATA_DIR=$1

if [ -z "${CHECKPOINTS_DIR}" ] | [ -z "${DATA_DIR}" ]; then
    echo "Usage: test.sh <data_dir>"
    exit -1
fi

ROOT=$(realpath $(dirname $0)/..)
cd ${ROOT}

source .venv/bin/activate

tmux new -s PICK-mkt-train_jose -d "python train.py -c data/rg-gfuhr/config.json -d 0 -dist false &> log.log"
