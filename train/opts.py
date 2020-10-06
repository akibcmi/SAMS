# -*- encoding=utf-8 -*-
import argparse

def create_parse():
    parser = argparse.ArgumentParser()
    #trainfiles
    parser.add_argument("--logfiles", type=str, default=""
                        ,help="path to log files")
    parser.add_argument("--name", type=str, required=True
                        ,help="name of dataset")
    parser.add_argument("--dataset", type=str, default=None
                        , help="name of train")
    parser.add_argument("--trainfile", type=str, required=True
                        , help="path to train files")
    parser.add_argument("--evalfile", type=str, required=True
                        ,help="name of eval dataset")
    parser.add_argument("--pretrain_files", nargs="?", default="", type=str
                        ,help="path to pre-training")
    parser.add_argument("--savefiles",type=str,required=True
                        ,help="path to save")
    parser.add_argument("--savesteps",type=int,required=True
                        ,help="steps for save")
    parser.add_argument("--savevalid",type=str,required=True,default=""
                        ,help="steps for valid")
    parser.add_argument("--train_from", type=str, default=""
                        ,help="checkpoint")

    #train
    parser.add_argument("--showsteps",type=int,default=100
                        ,help="step to show status")
    parser.add_argument("--gpu",action="store_true"
                        ,help="using gpu")
    parser.add_argument("--use_buffers", action="store_true"
                        ,help="using buffer")
    parser.add_argument("--dropout", required=True, type=float
                        ,help="dropout")
    parser.add_argument("--des", action="store_true"
                        ,help="sort data")
    parser.add_argument("--token", default="sentence", choices=['token', 'sentence']
                        ,help="batch type")
    parser.add_argument("--buffer_size", default=0, type=int
                        ,help="size of buffer")
    parser.add_argument("--learning_rate", type=float, default=1.0
                        ,help="learning rate")
    parser.add_argument("--warmup_start_lr", type=float, default=0.0
                        ,help="warmup learning rate")
    parser.add_argument("--warmup_steps", type=int
                        ,help="warmup steps")
    parser.add_argument("-b", "--batch_size", default=32, type=int
                        ,help="size of batch")
    parser.add_argument("-e", "--epoch", required=True, type=int
                        ,help="epoch")
    parser.add_argument("--optim", default="sgd", choices=['sgd', 'adagrad', 'adam']
                        ,help="optim")
    parser.add_argument("--loss", default="crossentropyloss", choices=['crossentropyloss', 'nllloss']
                        ,help="loss")
    parser.add_argument("--adam_beta1", type=float, default=0.9
                        ,help="adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.98
                        ,help="adam beta2")
    parser.add_argument("--decay_method", type=str, default="none", choices=['none', 'noam']
                        ,help="decay_method")
    parser.add_argument("--start_decay_steps", type=int, default=None
                        )
    parser.add_argument("--decay_steps", type=int, default=None
                        )
    parser.add_argument("--train_steps", default=0, type=int
                        )
    parser.add_argument("--optims", default='fairseq', type=str
                        ,help="type of optim")
    parser.add_argument("--cyc", action="store_true"
                        ,help="train in cycle without epoch")
    parser.add_argument("--train_counts_type", default="epoch", choices=["epoch", "step"]
                        )
    #train




    #valid
    parser.add_argument("--valid_token",default='sentence',choices=['token',"sentence"]
                        ,help="batch type for valid")
    parser.add_argument("--valid_batch_size",default=32,type=int
                        ,help="batch size for valid")
    parser.add_argument("--valid_steps",default=5000,type=int
                        ,help="")


    #model
    parser.add_argument("--seglayers",action="store_true"
                        ,help="middle layer in model")
    parser.add_argument("--segwords",action="store_true"
                        ,help="inner-word-embedding in model")
    parser.add_argument("--middecode",action="store_true"
                        ,help="label-embedding in model")
    parser.add_argument("--gate",action="store_true"
                        ,help="gate for backward and forward")
    parser.add_argument("--encoder",
                        help="The encoder of transformer", default="transformer", choices=["transformer"])
    parser.add_argument("--head", type=int, help="The heads of transformer encoder", required=True)
    parser.add_argument("-l", "--layer", help="The layers of transformer encoder", required=True, type=int)
    parser.add_argument("-d", "--dim", help="The dims of transformer encoder", required=True, type=int)
    parser.add_argument("-f", "--ff", required=True, type=int
                        ,help="The dims of ff layer")
    parser.add_argument("-p", "--position_encoding", action="store_true"
                        ,help="position encoding")
    parser.add_argument("-w", "--window", default=5, type=int,help="size of window")

    #parser.add_argument("--save")
    parser.add_argument("--norm_after" , action="store_true",help="norm after")
    parser.add_argument("--reloadlrs",action="store_true",help="reset learning rate")
    parser.add_argument("--reloadoptims",action="store_true",help="reset optims")
    parser.add_argument("--multigpu",action="store_true",help="multigpu")
    parser.add_argument("--accumulate",default=1,type=int,help="accum")
    #model

    return parser.parse_args()









