# Self-Attention Model for Segmentation (SAMS)
This is a Python implementation of the CWS described in "Attention is All You Need for Chinese Word Segmentation"

##Contents
[Requirements](#Requirements)
[Training](#Training)
[Citation](#Citation)
[Credits](#Credits)

## Requirements
* Python 3.6
* [Pytorch](http://pytorch.org/) 1.0.0 or later. 1.5.0 is available.
* [Scipy](scipy.org) 1.5.2

### Pre-trained Embeddings
* Gensim

## Training
### Data
Training dataset is from [SIGHAN05](http://sighan.cs.uchicago.edu/bakeoff2005/). There are four datasets:
|Corpus|Encoding|Word Types|Words|Character|Types|Characters|
|---|---|---|---|---|---|---|
|Peking University (PKU)|CP936/Unicode|55,303|1,109,947|4,698|1,826,448|
|Microsoft Research (MSR)|CP936/Unicode|88,119|2,368,391|5,167|4,050,469|
|Academia Sinica (AS)|CP936/Unicode|141,340|5,449,698|6,117|8,368,050|
|City University of Hong Kong (CITYU)|HKSCS Unicode/Big Five|69,085|1,455,629|4,923|2,403,355|

### Training Instructions
We provide run512msrpkuascityu.sh for training. This shell will preprocess training data and train the model. After preprocessing, given data will be splited into two parts for training and development.

To train the model, you should fill the shell with arguments
* name of dataset.
* namespace of your training.
* path to your training data.
* dropout
* learning rate
* warmup step
* number of layer
* number of head
* dimension of model
* dimension of feed-forward layer.

Here we give a list of value for you to train the model.
* dropout=0.1
* learning rate=0.0003
* warmup step=16000
* number of layer=6
* number of head=4
* dimension of model=256
* dimension of feed-forward layer=1024

```
bash predatas.sh $name

dropout=0.1
lr=0.0003
warmup=16000

layer=6
head=4
dim=256
ff=1024
```

After fill the shell, simply run:
```
bash run512msrpkuascityu.sh
```

Log will be saved to
```
gofile/log/your namespace
```

Two checkpoints will be saved to 
```
gofile/checkpoints/your namespace
```
One is the last checkpoint and another is the best checkpoint.

You can also train the model with other shells.

### Arguments for Train
If you want to train the model with other shells, you can use command 
```
python train_single.py
```
with the following arguments:

Argument|Description|Type|Required
---|---|---|---
`--logfiles`|Path to save log|string|Yes
`--name`|Name of dataset|string|Yes
`--dataset`|Name of your training|string|No
`--trainfile`|Path to train data|string|Yes
`--evalfile`|Path to development data|string|Yes
`--savefiles`|Path to save checkpoint|string|Yes
`--savesteps`|Valid and save checkpoint every steps|int|Yes
`--savevalid`|Path to save result of eval|string|Yes
`--train_from`|Path to load checkpoint|string|No
`--showsteps`|Show status of model every steps|int|No
`--gpu`|Use gpu||No
`--use_buffers`|Use buffer for dataset||No
`--dropout`|Dropout for Model|float|Yes
`--des`|Sort data||No
`--buffer_size`|Size of buffer|No
`--token`|Token Type|token or sentence|No
`-e`,`--epoch`|Epoch|int|Yes
`--warmup_start_lr`|Warmup learning rate|float|No
`--warmup_steps`|Steps for warmup|int|No
`--batch_size`,`-b`|Size of batch|int|No
`--loss`|Type of loss|crossentropyloss,nllloss|No
`--learning_rate`|Learning rate|float|No
`--adam_beta1`|Beta1 for Adam|float|No
`--adam_beta2`|Beta2 for Adam|float|No
`--optim`|Optimizer|sgd,adagrad,adam|No
`--cyc`|Train model using steps only and epoch will not be shown||No
`--seglayers`|Using HiRED layer||No
`--segwords`|Using result of HiRED layer to generate a word vector||No
`--middecode`|Using output vector of HiRED layer as feature||No
`--gate`|Using gate to incorporate directional representation||No
`--head`|Heads of multi-head attention|int|Yes
`-l`,`--layer`|Layers of model|int|Yes
`-d`,`--dimension`|Dimension of model|int|Yes
`-f`,`--ff`|Dimension of feed-forward layer|int|Yes
`-p`,`--position_encoding`|Using position-encoding||No
`--norm_after`|Norm after||No
`--reloadlrs`|Reload learning rate||No

You can also use shell `createTrainData.sh` to generate dataset for training and validation.
```
bash createTrainData.sh indata sizeofvalid nameofdataset
```

### Evaluation Instructions
With a saved checkpoint, you can use 
```
bash dovalids.sh
```
to evaluate your model.

Before evaluation, you should fill `dovalids.sh` with arguments
* name, name of your evaluation
* checkpoint, path to checkpoint
* testfile, path to file for testing
* type, pku, msr, as ,cityu
* gold, path to gold
* words, path to vocabulary


This process will create some files and a one files end with `eval_repo` is the result. The precision, recall and f-1 value will be printed. 

You can also use 
```
perl score path_to_words path_to_gold path_to_result
```
or
```
python calculatePRF1.py --pred path_to_result --gold path_to_gold --word path_to_words
```
to get score.

## Sequence Labeling
Coming soon.

