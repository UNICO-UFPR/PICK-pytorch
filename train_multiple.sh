#!/bin/bash

for cfg in $(ls data/train_configs/*.json); do
    .venv/bin/python train.py -c $cfg -d 0 -dist false
done
