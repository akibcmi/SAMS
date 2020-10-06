#!/usr/bin/env bash

data=$1
type=$2


dataset=$data

if [ $type = cityu ]; then
opencc -i ${data} -o ${data}_s -c hk2s
dataset=${data}_s
fi

if [ $type = as ]; then
opencc -i ${data} -o ${data}_s -c tw2s
dataset=${data}_s
fi

python  preprocess.py --type produces \
    --dataset ${dataset} \
    --to ${data}_repo \
    --engs ${data}_engs \
    --nums ${data}_nums  \
    --lines ${data}_repo2