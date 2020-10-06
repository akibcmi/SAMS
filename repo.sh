#!/usr/bin/env bash

data=$1
repofile=$2
dataset2=$3
type=$4


dataset=$data
replaces=
if [ $type = cityu -o $type = as ]; then
replaces="--replace "${data}
dataset=${data}_s
fi

python  preprocess.py --type repos \
        --dataset ${dataset} \
        --to ${repofile} \
        --lines ${data}_repo2 \
        --nums ${data}_nums  \
        --engs ${data}_engs    \
        --dataset2 $dataset2  ${replace}