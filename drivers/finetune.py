from os.path import join
import sys
sys.path += ['../']
import argparse
import torch 
import numpy as np
import pandas as pd
import torch.distributed as dist
from model.models import MSMarcoConfigDict, ALL_MODELS
from utils.util import (
    StreamingDataset, 
    EmbeddingCache, 
    barrier_array_merge,
    get_checkpoint_no, 
    get_latest_ann_data,
    set_seed,
    is_first_worker,
)
from utils.dpr_utils import (
    load_states_from_checkpoint, 
    get_model_obj, 
    CheckpointState, 
    get_optimizer, 
    all_gather_list
)
from run_ann_data_gen_dpr import StreamInferenceDoc
from torch.utils.data import DataLoader
from data.nq320k_data import GetClassificationDataProcessingFn, load_mapping
import os
import pickle as pkl
import logging
logger = logging.getLogger(__name__)
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import random #TODO check if you need to set seed for this


def getDocumentEmbeddingMatrix(args):
    #dim [num_docs, hidden_size]
    if is_first_worker():
        logger.info(f'Loading weight matrix')
    temp_sz = args.world_size
    args.world_size = 4
    embedding_prefix = "ann_data/passage_"+ str(args.step_num)+ "__emb_p_"
    embedding_id_prefix = "ann_data/passage_"+ str(args.step_num)+ "__embid_p_"
    passage_embedding = barrier_array_merge(args, None, prefix = embedding_prefix, load_cache = True, only_load_in_master = False) 
    passage_embedding2id = barrier_array_merge(args, None, prefix = embedding_id_prefix, load_cache = True, only_load_in_master = False)
    
    args.world_size = temp_sz
    
    # sort passage_embeddings to be in order of 1 to n. Note you still need to do offset2pid
    embedding_matrix = passage_embedding[np.argsort(passage_embedding2id)] 
    
    if is_first_worker():
        logger.info(f'Sorting weights')
    
    ques2doc = pkl.load(open(os.path.join(args.raw_data_dir, "quesid2docid.pkl"), 'rb'))
    pid2offset, offset2pid = load_mapping(args.data_dir, "pid2offset")
    doc_ids = [ques2doc[offset2pid[i]] for i in range(len(passage_embedding))]
    index2docid = {}
    
    
    if args.average_doc_embeddings:
        embedding_matrix_pd = pd.DataFrame(embedding_matrix)
        embedding_matrix_pd.insert(0, "doc_ids", doc_ids)
        # average embeddigns corresponding to the same doc_id
        embedding_matrix_avg = embedding_matrix_pd.groupby('doc_ids').mean()
        index2docid = dict(zip([i for i in range(len(embedding_matrix_avg.index))], embedding_matrix_avg.index.values))
        
        # unique_doc_ids = np.unique(doc_ids)
        # new_embedding_matrix = np.empty((len(unique_doc_ids), embedding_matrix.shape[1]), np.float32)
        # for i, doc_id in enumerate(unique_doc_ids):
        #     index2docid[i] = doc_id
        #     new_embedding_matrix[i] = embedding_matrix[np.where(doc_ids == doc_id)[0]].mean(axis=0)
        docid2index  = {v: k for k, v in index2docid.items()}
        return embedding_matrix_avg.to_numpy(), index2docid, docid2index, ques2doc, offset2pid
    else:
        for i, doc_id in enumerate(doc_ids):
            index2docid[i] = doc_id
        
        return embedding_matrix, index2docid, None, ques2doc, offset2pid

def load_model(args, num_classes):
    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    tokenizer = configObj.tokenizer_class.from_pretrained(
        "bert-base-uncased",
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
    
    # 768 is BERT hidden_size
    model = configObj.model_class(args, torch.ones(num_classes, 768))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    return tokenizer, model

def load_saved_weights(args, model, strict_set=False):
    if args.model_name_or_path != "None":
        saved_state = load_states_from_checkpoint(args.model_name_or_path)
        model_to_load = get_model_obj(model)
        model_to_load.load_state_dict(saved_state.model_dict, strict=strict_set)
    

def validate(args, model):
    model.eval()
    dev_query_collection_path = os.path.join(args.data_dir, f"dev-query")
    dev_query_cache = EmbeddingCache(dev_query_collection_path)
    
    with dev_query_cache:
        dev_data_path = os.path.join(args.data_dir, f"dev-data")
        with open(dev_data_path, 'r') as f:
            dev_data = f.readlines()
        dev_dataset = StreamingDataset(dev_data, GetClassificationDataProcessingFn(args, dev_query_cache))
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.train_batch_size*2)
        
        total_correct_predictions = 0
        total_correct_predictions_10 = 0

        for i, batch in enumerate(dev_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(), "labels": batch[3].long(), "validate":True}
            with torch.no_grad():
                loss, correct_cnt, correct_cnt_10 = model(**inputs)
            
            total_correct_predictions += correct_cnt
            total_correct_predictions_10 += correct_cnt_10
    
    dist.all_reduce(total_correct_predictions, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_correct_predictions_10, op=dist.ReduceOp.SUM)
    dist.barrier()
    
    correct_ratio = float(total_correct_predictions / len(dev_data))
    correct_ratio_10 = float(total_correct_predictions_10 / len(dev_data))
    return correct_ratio, correct_ratio_10

def train(args, model, global_step, optimizer):
    model.train()
    train_query_collection_path = os.path.join(args.data_dir, f"train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path)
    
    with train_query_cache:
        train_data_path = os.path.join(args.data_dir, f"train-data")
        with open(train_data_path, 'r') as f:
            train_data = f.readlines()
        random.shuffle(train_data)
        train_dataset = StreamingDataset(train_data, GetClassificationDataProcessingFn(args, train_query_cache))
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)
        
        total_correct_predictions = 0
        tr_loss = 0

        for i, batch in enumerate(train_dataloader):
            batch = tuple(t.to(args.device) for t in batch)
            dist.barrier()
            inputs = {"query_ids": batch[0].long(), "attention_mask_q": batch[1].long(), "labels": batch[3].long()}
            
            loss, correct_cnt, _ = model(**inputs)

            tr_loss += loss.item()
            
            total_correct_predictions += correct_cnt
            if (i + 1) % args.logging_steps == 0 and is_first_worker():
                logger.info(f'Train step: {i}, loss: {(tr_loss/i)}')

            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1
    
    
    dist.all_reduce(total_correct_predictions, op=dist.ReduceOp.SUM)
    dist.barrier()
    correct_ratio = float(total_correct_predictions / len(train_data))
    
    if is_first_worker():
        logger.info(f'Train accuracy:{correct_ratio}')
        logger.info(f'Loss:{tr_loss/i}')
    return correct_ratio, global_step

def _save_checkpoint(args, model, step: int) -> str:
    epoch = 0
    model_to_save = get_model_obj(model)
    cp = os.path.join(args.output_dir, 'dpr_checkpoint_finetuned-' + str(step))

    meta_params = {}

    state = CheckpointState(model_to_save.state_dict(),
                            None,
                            None,
                            step,
                            epoch, meta_params
                            )
    torch.save(state._asdict(), cp)
    logger.info('Saved checkpoint at %s', cp)
    return cp

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )
    parser.add_argument(
        "--ann_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MSMarcoConfigDict.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--raw_data_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the raw data is present.",
    )

    parser.add_argument(
        "--num_epoch",
        default=10,
        type=int,
        help="Number of epoch to train, if specified will use training data instead of ann",
    )

    
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--max_steps",
        default=300000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--step_num", type=int, help="Step number for loading passage embeddings")
    parser.add_argument("--average_doc_embeddings", action="store_true", help="average the generated document embeddings")
    parser.add_argument("--validate_only", action="store_true", help="only runs topk validation")


    args = parser.parse_args()

    return args


def set_env(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )
    logging.basicConfig(filename=f'{args.data_dir}/out.log', encoding='utf-8', level=logging.DEBUG)

    # Set seed
    set_seed(args)

    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

def main():
    args = get_arguments()
    set_env(args)
    docMatrix, index2docid, docid2index, ques2doc, offset2pid = getDocumentEmbeddingMatrix(args)
    if is_first_worker():
        logger.info("Weight readign complete")
    tokenizer, model = load_model(args, docMatrix.shape[0])
    
    # load weights and set the weights of the classification layer and set mappings required for forward
    load_saved_weights(args, model)
    with torch.no_grad():
        model.classifier.weight.data = torch.from_numpy(docMatrix)
        model.classifier.weight.data = model.classifier.weight.data.cuda()
    model.index2docid = index2docid
    model.docid2index = docid2index
    model.ques2doc = ques2doc
    model.offset2pid = offset2pid

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.1)

    if args.validate_only:
        load_saved_weights(args, model, strict_set=True)

    args.train_batch_size = args.per_gpu_train_batch_size

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )
    
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)
    global_step = 0
    
    correct_ratio, correct_ratio_10 = validate(args, model)

    if is_first_worker():
        tb_writer.add_scalar("Evaluation Accuracy", correct_ratio, global_step)
        logger.info(f'Validation accuracy at step {global_step}:{correct_ratio}')
        logger.info(f'Hits@10 at step {global_step}:{correct_ratio_10}')
        
    if not args.validate_only:
        for i in range(args.num_epoch):
            print(f"Learning Rate: {scheduler.get_last_lr()}")
            correct_ratio_train, global_step = train(args, model, global_step, optimizer)
            scheduler.step()
            correct_ratio, correct_ratio_10 = validate(args, model)

            if is_first_worker():
                logger.info(f'Validation accuracy at step {global_step}, epoch {i}:{correct_ratio}')
                logger.info(f'Validation hits@10 at step {global_step}, epoch {i}:{correct_ratio_10}')
                tb_writer.add_scalar("Evaluation Accuracy", correct_ratio, global_step)
                tb_writer.add_scalar("Evaluation Hits@10", correct_ratio_10, global_step)
                tb_writer.add_scalar("Train Accuracy", correct_ratio_train, global_step)

            _save_checkpoint(args, model, global_step)
    
    if args.local_rank != -1:
        dist.barrier()
        tb_writer.close()
        

if __name__ == "__main__":
    main()