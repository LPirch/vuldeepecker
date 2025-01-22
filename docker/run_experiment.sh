#!/usr/bin/env bash

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

seed=$1

CUDA_VISIBLE_DEVICES= python vuldeepecker.py \
    data/raw/primevul_train.jsonl \
    data/raw/primevul_valid.jsonl \
    data/raw/primevul_test.jsonl \
    data/results/seed_$seed \
    $seed
