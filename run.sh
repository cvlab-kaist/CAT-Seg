#!/bin/sh

gpus=4
config=$1
output=$2

if [ -z $config ]
then
    echo "No config file found! Run with "sh run.sh [CONFIG_FILE] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh run.sh [CONFIG_FILE] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 2
opts=${@}

python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --resume \
 OUTPUT_DIR $output \
 $opts

sh eval.sh $config $output $opts