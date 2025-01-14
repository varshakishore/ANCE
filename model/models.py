import sys
sys.path += ['../']
import torch
from torch import nn
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    BertModel,
    BertTokenizer,
    BertConfig
)
import torch.nn.functional as F
from data.process_fn import triple_process_fn, triple2dual_process_fn
from model.SEED_Encoder import SEEDEncoderConfig, SEEDTokenizer, SEEDEncoderForSequenceClassification,SEEDEncoderForMaskedLM
import pickle as pkl
import os

class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained 
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        assert isinstance(emb_all, tuple)
        if self.use_mean:
            return self.masked_mean(emb_all[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class NLL_MultiChunk(EmbeddingMixin):
    def forward(
            self,
            query_ids,
            attention_mask_q,
            input_ids_a=None,
            attention_mask_a=None,
            input_ids_b=None,
            attention_mask_b=None,
            is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        [batchS, full_length] = input_ids_a.size()
        chunk_factor = full_length // self.base_len

        # special handle of attention mask -----
        attention_mask_body = attention_mask_a.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), a_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_a = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        # special handle of attention mask -----
        attention_mask_body = attention_mask_b.reshape(
            batchS, chunk_factor, -1)[:, :, 0]  # [batchS, chunk_factor]
        inverted_bias = ((1 - attention_mask_body) * (-9999)).float()

        a12 = torch.matmul(
            q_embs.unsqueeze(1), b_embs.transpose(
                1, 2))  # [batch, 1, chunk_factor]
        logits_b = (a12[:, 0, :] + inverted_bias).max(dim=-
                                                      1, keepdim=False).values  # [batch]
        # -------------------------------------

        logit_matrix = torch.cat(
            [logits_a.unsqueeze(1), logits_b.unsqueeze(1)], dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(),)


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """

    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


class RobertaDot_CLF_ANN_NLL_MultiChunk(NLL_MultiChunk, RobertaDot_NLL_LN):
    def __init__(self, config):
        RobertaDot_NLL_LN.__init__(self, config)
        self.base_len = 512

    def body_emb(self, input_ids, attention_mask):
        [batchS, full_length] = input_ids.size()
        chunk_factor = full_length // self.base_len

        input_seq = input_ids.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)
        attention_mask_seq = attention_mask.reshape(
            batchS,
            chunk_factor,
            full_length //
            chunk_factor).reshape(
            batchS *
            chunk_factor,
            full_length //
            chunk_factor)

        outputs_k = self.roberta(input_ids=input_seq,
                                 attention_mask=attention_mask_seq)

        compressed_output_k = self.embeddingHead(
            outputs_k[0])  # [batch, len, dim]
        compressed_output_k = self.norm(compressed_output_k[:, 0, :])

        [batch_expand, embeddingS] = compressed_output_k.size()
        complex_emb_k = compressed_output_k.reshape(
            batchS, chunk_factor, embeddingS)

        return complex_emb_k  # size [batchS, chunk_factor, embeddingS]

class SEEDEncoderDot_NLL_LN(NLL, SEEDEncoderForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        SEEDEncoderForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.encoder_embed_dim, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask=None):
        outputs1 = self.seed_encoder.encoder(input_ids)

        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))

        return query1

    def body_emb(self, input_ids, attention_mask=None):
        return self.query_emb(input_ids, attention_mask)

class HFBertEncoder(BertModel):
    def __init__(self, config):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.init_weights()
    @classmethod
    def init_encoder(cls, args, dropout: float = 0.1):
        cfg = BertConfig.from_pretrained("bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained("bert-base-uncased", config=cfg)
    def forward(self, input_ids, attention_mask):
        hidden_states = None
        sequence_output, pooled_output = super().forward(input_ids=input_ids,
                                                         attention_mask=attention_mask)
        pooled_output = sequence_output[:, 0, :]
        return sequence_output, pooled_output, hidden_states
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args):
        super(BiEncoder, self).__init__()
        self.question_model = HFBertEncoder.init_encoder(args)
        self.ctx_model = HFBertEncoder.init_encoder(args)
        
        if args.tie_weights:
            self.tie_weights = True
        else:
            self.tie_weights = False
    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        return pooled_output
    def body_emb(self, input_ids, attention_mask):
        if self.tie_weights: 
            return self.query_emb(input_ids, attention_mask)
        sequence_output, pooled_output, hidden_states = self.ctx_model(input_ids, attention_mask)
        return pooled_output
    def forward(self, query_ids, attention_mask_q, input_ids_a = None, attention_mask_a = None, input_ids_b = None, attention_mask_b = None):
        if input_ids_b is None:
            q_embs = self.query_emb(query_ids, attention_mask_q)
            a_embs = self.body_emb(input_ids_a, attention_mask_a)
            return (q_embs, a_embs)
        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)
        # commenting out the old loss
        # logit_matrix = torch.cat([(q_embs*a_embs).sum(-1).unsqueeze(1), (q_embs*b_embs).sum(-1).unsqueeze(1)], dim=1) #[B, 2]
        # lsm = F.log_softmax(logit_matrix, dim=1)
        # loss = -1.0*lsm[:,0]
        # return (loss.mean(),)

        # import pdb; pdb.set_trace()
        # new infonce loss func
        scores = torch.matmul(q_embs, torch.transpose(torch.cat([a_embs, b_embs]), 0, 1)) #[B, a_embs_dim+b_embs_dim]
        softmax_scores = F.log_softmax(scores, dim=1)
        positive_idx_per_question = [i for i in range(len(q_embs))]
        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )
        return (loss,)

class QueryClassifier(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """
    def __init__(self, args, documentEmbeddings):
        super(QueryClassifier, self).__init__()
        # note here we only have question encoder
        self.question_model = HFBertEncoder.init_encoder(args)
        self.classifier = nn.Linear(self.question_model.config.hidden_size, len(documentEmbeddings), bias=False)
        self.lossfunc = nn.CrossEntropyLoss()
        self.index2docid = None
        self.docid2index = None
        self.ques2doc = None
        self.offset2pid = None

        # self.ques2doc = pkl.load(open("/home/vk352/ANCE/NQ320k_dataset/quesid2docid.pkl", 'rb')) # TODO remove this hardcoding
        # self.pid2offset, self.offset2pid = load_mapping("/home/vk352/ANCE/data/NQ320k_data_original_10k", "pid2offset") #TODO remove hardcoding
        # # set the weights of the classifier here

    def query_emb(self, input_ids, attention_mask):
        sequence_output, pooled_output, hidden_states = self.question_model(input_ids, attention_mask)
        return pooled_output
    def forward(self, query_ids, attention_mask_q, labels, validate=False):
        q_embs = self.query_emb(query_ids, attention_mask_q)

        top10 = 0
        
        if self.docid2index is None:
            logits = self.classifier(q_embs)
            loss = self.lossfunc(logits, labels)
        else:    
            new_labels = torch.tensor([self.docid2index[self.ques2doc[self.offset2pid[label.item()]]] for label in labels]).to(labels.device)
            logits = self.classifier(q_embs)
            loss = self.lossfunc(logits, new_labels)
        
        _, max_idxs = torch.max(logits, 1)
        
        docids = torch.tensor([self.ques2doc[self.offset2pid[label.item()]] for label in labels]).to(max_idxs.device)
        # max_idxs_rp = torch.tensor([self.ques2doc[self.offset2pid[max_idx.item()]] for max_idx in max_idxs]).to(max_idxs.device)
        max_idxs_rp = torch.tensor([self.index2docid[max_idx.item()] for max_idx in max_idxs]).to(max_idxs.device)

        correct_predictions_count = (max_idxs_rp == docids).sum()

        if validate:  
            max_idxs_10 = torch.argsort(logits, 1, descending=True)[:, :10]
            max_idxs_rp10 = torch.tensor([[self.index2docid[max_idx.item()] for max_idx in max_idx_row] for max_idx_row in max_idxs_10]).to(max_idxs.device)
            top10 = (max_idxs_rp10 == docids.unsqueeze(1)).any(1).sum()

        return (loss, correct_predictions_count, top10)
        
def load_mapping(data_dir, out_name):
    out_path = os.path.join(
        data_dir,
        out_name ,
    )
    pid2offset = {}
    offset2pid = {}
    with open(out_path, 'r') as f:
        for line in f.readlines():
            line_arr = line.split('\t')
            pid2offset[int(line_arr[0])] = int(line_arr[1])
            offset2pid[int(line_arr[1])] = int(line_arr[0])
    return pid2offset, offset2pid
        
        

# --------------------------------------------------
ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            RobertaConfig,
        ) if hasattr(conf,'pretrained_config_archive_map')
    ),
    (),
)


default_process_fn = triple_process_fn


class MSMarcoConfig:
    def __init__(self, name, model, process_fn=default_process_fn, use_mean=True, tokenizer_class=RobertaTokenizer, config_class=RobertaConfig):
        self.name = name
        self.process_fn = process_fn
        self.model_class = model
        self.use_mean = use_mean
        self.tokenizer_class = tokenizer_class
        self.config_class = config_class


configs = [
    MSMarcoConfig(name="rdot_nll",
                model=RobertaDot_NLL_LN,
                use_mean=False,
                ),
    MSMarcoConfig(name="rdot_nll_multi_chunk",
                model=RobertaDot_CLF_ANN_NLL_MultiChunk,
                use_mean=False,
                ),
    MSMarcoConfig(name="dpr",
                model=BiEncoder,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
                use_mean=False,
                ),
    MSMarcoConfig(name="seeddot_nll",
                model=SEEDEncoderDot_NLL_LN,
                use_mean=False,
                tokenizer_class=SEEDTokenizer,
                config_class=SEEDEncoderConfig,
                ),
    MSMarcoConfig(name="classify",
                model=QueryClassifier,
                tokenizer_class=BertTokenizer,
                config_class=BertConfig,
                use_mean=False,
                ),
]

MSMarcoConfigDict = {cfg.name: cfg for cfg in configs}
