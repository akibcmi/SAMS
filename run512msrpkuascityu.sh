cpu=0
name=pku
namespace=pku1
data=training/as

bash createTrainData.sh $data 2000 $name

bash predatas.sh $name

dropout=
lr=
warmup=

layer=
head=
dim=
ff=



echo 'start training'
mkdir gofiles
mkdir ./gofiles/checkpoints
mkdir ./gofiles/checkpoints/${namespace}
mkdir ./gofiles/logs
mkdir ./gofiles/valid
mkdir ./gofiles/valid/${namespace}_valid



CUDA_VISIBLE_DEVICES=$cpu  python train_single.py  --logfiles gofiles/logs/${namespace}.log \
        --name ${name} \
        --dataset ${namespace} \
        --gpu \
        --trainfile training/${name}_trains_repo \
        --evalfile training/${name}_dev_repo \
        --savefiles gofiles/checkpoints/${namespace}/${namespace}_model \
        --savesteps 5000 \
        --savevalid gofiles/valid/${namespace}_valid \
        --showsteps 100 \
        --dropout $dropout \
        --use_buffers \
        --learning_rate $lr \
        --warmup_steps $warmup \
        --epoch 100 \
        --optim adam \
        --loss crossentropyloss \
        --encoder transformer \
        --head $head \
        --layer $layer \
        --dim $dim \
        --ff $ff \
        --norm_after \
        --decay_method noam \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        -p \
        --warmup_start_lr 1e-07 \
        --train_steps 200000 \
        --valid_steps 5000 \
        --valid_batch_size 16 \
        --seglayers \
        --segwords \
	    --middecode \
        --token token \
        --batch_size  4096 \
        --des \
        --gate
         #> msr_train_txt
