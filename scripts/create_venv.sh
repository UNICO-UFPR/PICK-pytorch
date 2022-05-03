#!/bin/bash

ROOT=$(realpath $(dirname $0)/..)
cd ${ROOT}

if [ ! -d .venv ]; then
    python3.6 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
fi
