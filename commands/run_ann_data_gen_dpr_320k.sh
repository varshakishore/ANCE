# tokenization
wiki_dir="/home/zl327/NQ320K_ANCE/" # path for "passages". These are the generated queries
ans_dir="/home/zl327/NQ320K_ANCE/" # path for test data
question_dir="/home/zl327/NQ320K_ANCE/" # path for training data
data_type=0 #0 is nq, 1 is trivia, 2 is both
out_data_dir="../data/NQ320k_data/" # change this for different data_type

tokenization_cmd="\
python ../data/nq320k_data.py --wiki_dir $wiki_dir --question_dir $question_dir --data_type $data_type --answer_dir $ans_dir \
--out_data_dir $out_data_dir \
"

echo $tokenization_cmd
eval $tokenization_cmd


gpu_no=4

# model type
model_type="dpr"
seq_length=256

# ann parameters
batch_size=16
ann_topk=200
ann_negative_sample=100

# input/output directories
base_data_dir="${out_data_dir}"
job_name="ann_NQ_test"
model_dir="${base_data_dir}${job_name}/"
model_ann_data_dir="${model_dir}ann_data/"
pretrained_checkpoint_dir="../../DPR/dpr/downloads/checkpoint/retriever/multiset/bert-base-encoder.cp"
passage_path="/home/zl327/NQ320K_ANCE/"
test_qa_path="/home/zl327/NQ320K_ANCE/"


data_gen_cmd="\
python -m torch.distributed.launch --nproc_per_node=$gpu_no ../drivers/run_ann_data_gen_dpr.py --training_dir $model_dir \
--init_model_dir $pretrained_checkpoint_dir --model_type $model_type --output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" --data_dir $base_data_dir --max_seq_length $seq_length \
--per_gpu_eval_batch_size $batch_size --topk_training $ann_topk --negative_sample $ann_negative_sample \
--passage_path $passage_path --test_qa_path $test_qa_path --nq320k\
"

echo $data_gen_cmd
eval $data_gen_cmd