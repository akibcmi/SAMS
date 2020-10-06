#/bin/bash
cpu=0
name=
checkpoint=
testfile=
type=

gold=
words=

batch_size=16
showsteps=100

log=${testfile}.log

bash pro.sh $testfile $type

savefile=${testfile}_eval

CUDA_VISIBLE_DEVICES=$cpu   python valid_single.py --logfiles $log \
        --name $name \
        --evalfile  ${testfile}_repo \
        --savefiles $savefile \
        --showsteps $showsteps \
        --gpu \
        --valid_batch_size $batch_size \
        --use_buffers \
        --model  $checkpoint



bash repo.sh $testfile $savefile ${savefile}_repo $type



python calculatePRF1.py --pred ${savefile}_repo --gold ${gold} --word ${words}
