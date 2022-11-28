#!/bin/bash

#master_ip=$1
#rank=$2
#bsize=$3
#dataset_location=$4
#log_file=$5
#num_workers=${6}
#s3_prefix=${7}

#echo "$master_ip";
#echo "$rank";
#echo "$bsize";
#echo "$dataset_location";
#echo "$log_file";
#echo "$num_workers";
#echo $s3_prefix;

source activate pytorch

set -x

# sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
# python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 50 --num-procs 4 --data-set ImageNet --log-file-prefix

#sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
#sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB

sudo tc qdisc add dev lo root tbf rate 10gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 256 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --log-file-prefix 2-2-10gbi-bs256

sudo tc qdisc del dev lo root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev lo root tbf rate 1gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 256 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --log-file-prefix 2-2-1gbit-bs256

sudo tc qdisc del dev lo root tbf rate 0.1gbit latency 50ms burst 10MB
