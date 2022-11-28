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

# sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
# python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 50 --num-procs 4 --data-set ImageNet --log-file-prefix

# sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --log-file-prefix 2-2-10gbit

sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --log-file-prefix 2-2-1gbit

sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.3 --log-file-prefix 2-2-10gbit-rand30

sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.5 --log-file-prefix 2-2-10gbit-rand50

sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.8 --log-file-prefix 2-2-10gbit-rand80

sudo tc qdisc del dev ens3 root tbf rate 10gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.3 --log-file-prefix 2-2-1gbit-rand30

sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.5 --log-file-prefix 2-2-1gbit-rand50

sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
sudo tc qdisc add dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
python train_model_4gpus.py --lr 0.003 --bs 32 --mb 4 --epochs 2 --num-procs 2 --data-set ImageNet --compression-type randomk --compression-ratio 0.8 --log-file-prefix 2-2-1gbit-rand80

sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB

#sudo tc qdisc del dev ens3 root tbf rate 1gbit latency 50ms burst 10MB
#sudo tc qdisc add dev ens3 root tbf rate 2gbit latency 50ms burst 10MB
#OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$num_workers --node_rank=$rank --master_addr=$master_ip --master_port=2345 main_bert.py --batch-size $bsize --dataset-location $dataset_location --log-file $log_file --s3-prefix "${s3_prefix}_2gbps" --node_rank $rank --max_seq_length 512
#sudo tc qdisc del dev ens3 root tbf rate 2gbit latency 50ms burst 10MB
#sudo tc qdisc add dev ens3 root tbf rate 4gbit latency 50ms burst 10MB
#OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$num_workers --node_rank=$rank --master_addr=$master_ip --master_port=2345 main_bert.py --batch-size $bsize --dataset-location $dataset_location --log-file $log_file --s3-prefix "${s3_prefix}_4gbps" --node_rank $rank --max_seq_length 512
#sudo tc qdisc del dev ens3 root tbf rate 4gbit latency 50ms burst 10MB
#sudo tc qdisc add dev ens3 root tbf rate 8gbit latency 50ms burst 10MB
#OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$num_workers --node_rank=$rank --master_addr=$master_ip --master_port=2345 main_bert.py --batch-size $bsize --dataset-location $dataset_location --log-file $log_file --s3-prefix "${s3_prefix}_8gbps" --node_rank $rank --max_seq_length 512

# python main.py --arch $arch --master-ip $2 --rank $3 --reducer $4 --batch-size $5 --dataset-location $6 --device cuda:1 --log-file $8 --reducer $9 --reducer-param $reducer_param
