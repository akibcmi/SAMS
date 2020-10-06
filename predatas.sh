#!/bin/bash
name=$1

train=${name}_training.utf8
dev=${name}_dev.utf8
if [ $name = cityu ]; then
opencc -i training/${name}_training.utf8 -o training/${name}_training_s.utf8 -c hk2s
opencc -i training/${name}_dev.utf8 -o training/${name}_dev_s.utf8 -c hk2s
train=${name}_training_s.utf8
dev=${name}_dev_s.utf8
fi
if [ $name = as ]; then
opencc -i training/${name}_training.utf8 -o training/${name}_training_s.utf8 -c tw2s
opencc -i training/${name}_dev.utf8 -o training/${name}_dev_s.utf8 -c tw2s
train=${name}_training_s.utf8
dev=${name}_dev_s.utf8
fi
echo ${train}
echo ${dev}
echo 'make training datas'
python  preprocess.py --type produces \
    --dataset training/${train} \
    --to training/${name}_trains_repo \
    --engs training/${name}_trains_engs \
    --nums training/${name}_trains_nums  \
    --lines training/${name}_trains_repo2

echo 'make dev datas'
python preprocess.py --type produces \
    --dataset training/${dev} \
    --to training/${name}_dev_repo \
    --nums training/${name}_dev_nums \
    --engs training/${name}_dev_engs \
    --lines training/${name}_dev_repo2

