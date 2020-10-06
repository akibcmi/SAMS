#!/usr/bin/env bash

data=$1
cous=$2
name=$3



python gettraindev.py --infile $data \
            --count $cous \
            --name $name