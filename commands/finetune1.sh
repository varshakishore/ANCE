# echo "sleeping"
# sleep 4200
# echo "done sleeping"
gpu_no=4

# model type
model_type="classify"
seq_length=256

# hyper parameters
batch_size=16
learning_rate=1e-4

# input/output directories
base_data_dir="/home/vk352/ANCE/data/NQ320k_data_original_tied_random/"
raw_data_dir="../NQ320k_dataset/"
# raw_data_dir="/scratch/ir/MSMARCO320k"
tensorboard_name="nq320k_pretrained_try4"
# tensorboard_name="temp"
job_name="ann_NQ_test"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
step_num=0
# pretrained_checkpoint_dir="/home/vk352/ANCE/data/NQ320k_data_original_tied_random/ann_NQ_test/checkpoint-${step_num}"
# pretrained_checkpoint_dir="/home/cw862/DPR/MSMARCO320k_output/dpr_biencoder.1"
pretrained_checkpoint_dir="None"

# while true
# do
#     END_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $END_PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done

train_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/finetune.py \
--model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir --data_dir $base_data_dir \
--ann_dir $model_ann_data_dir --max_seq_length $seq_length --per_gpu_train_batch_size=$batch_size \
--learning_rate $learning_rate --output_dir $model_dir --raw_data_dir $raw_data_dir --step_num $step_num --average_doc_embeddings \
--logging_steps 100 --save_steps 1000 --log_dir "~/tensorboard/${DLWS_JOB_ID}logs/${tensorboard_name}" \
"

echo $train_cmd
eval $train_cmd

echo "copy current script to model directory"
# cp $0 $model_dir