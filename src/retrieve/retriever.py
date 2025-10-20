from typing import List, Dict, Tuple
import os
import time
import tqdm
import uuid
import numpy as np
import torch
#import faiss
import logging
import pandas as pd
from transformers import AutoTokenizer, AutoModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "beir"))
import beir
logging.getLogger("beir_import_diag").info(f"imported beir module from: {getattr(beir, '__file__', 'built-in or package without __file__')}" )
logging.getLogger("beir_import_diag").info(f"sys.path[0:5]={sys.path[0:5]}")

from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

def get_random_doc_id():
    return f'_{uuid.uuid4()}'

class BM25:
    """Lightweight BM25 retrieval wrapper backed by a BEIR/Elasticsearch BM25Search.

    This class provides a small interface to perform lexical BM25 retrieval for a
    list of queries. It builds on the repository-local BEIR implementation
    (wrapped by `EvaluateRetrieval`) and returns retrieved passage ids and
    passages (texts) in a shape-friendly form for downstream use.

    Usage:
        tokenizer = AutoTokenizer.from_pretrained(...)
        retriever = BM25(tokenizer=tokenizer, index_name="wiki", engine="elasticsearch")
        docids, docs = retriever.retrieve(["query1", "query2"], topk=5)

    Constructor args:
        tokenizer: a HuggingFace tokenizer used only for optional query
            truncation/padding. If `max_query_length` is provided to
            `retrieve`, the tokenizer will be used to truncate queries.
        index_name: Elasticsearch index name where the corpus is (or will be)
            indexed.
        engine: currently only 'elasticsearch' is supported (kept for API
            compatibility).
        **search_engine_kwargs: forwarded to the underlying BM25Search/ES
            constructor if needed.

    retrieve args/returns:
        retrieve(queries: List[str], topk: int = 1, max_query_length: int = None)
            - queries: list of query strings (length = batch size)
            - topk: number of passages to return per query
            - max_query_length: if provided, queries are tokenized and
              truncated/padded to this length before search (tokenizer must be
              provided)
        returns: (docids, docs)
            - docids: numpy array of shape (batch_size, topk) containing the
              retrieved document ids (strings). When insufficient results are
              found, dummy ids like '_<uuid>' are inserted.
            - docs: numpy array of shape (batch_size, topk) containing the
              retrieved document texts (strings). Empty strings are used for
              missing documents.

    Side effects and notes:
        - If the underlying BM25Search is constructed with `initialize=True`,
          the corpus will be indexed (and the index may be recreated). This
          can be slow for large corpora; prefer constructing BM25Search with
          `initialize=False` if the index already exists.
        - This wrapper expects the repository-local `beir` implementation. The
          module import order is adjusted in this file to prefer the local
          `beir` package to avoid mismatches with any installed `beir`.

    """
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        index_name: str = None,
        engine: str = 'elasticsearch',
        **search_engine_kwargs,
    ):
        self.tokenizer = tokenizer
        # load index
        assert engine in {'elasticsearch', 'bing'}
        if engine == 'elasticsearch':
            self.max_ret_topk = 1000
            self.retriever = EvaluateRetrieval(
                BM25Search(index_name=index_name, hostname='http://localhost:9200', initialize=False, number_of_shards=1),
                k_values=[self.max_ret_topk])

    def retrieve(
        self,
        queries: List[str],  # (bs,)
        topk: int = 1,
        max_query_length: int = None,
    ):
        assert topk <= self.max_ret_topk
        device = None
        bs = len(queries)

        # truncate queries
        if max_query_length:
            ori_ps = self.tokenizer.padding_side
            ori_ts = self.tokenizer.truncation_side
            # truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'
            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_query_length,
                add_special_tokens=False,
                return_tensors='pt')['input_ids']
            self.tokenizer.padding_side = ori_ps
            self.tokenizer.truncation_side = ori_ts
            queries = self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        # retrieve
        results: Dict[str, Dict[str, Tuple[float, str]]] = self.retriever.retrieve(
            None, dict(zip(range(len(queries)), queries)), disable_tqdm=True)

        # prepare outputs
        docids: List[str] = []
        docs: List[str] = []
        for qid, query in enumerate(queries):
            _docids: List[str] = []
            _docs: List[str] = []
            if qid in results:
                for did, (score, text) in results[qid].items():
                    _docids.append(did)
                    _docs.append(text)
                    if len(_docids) >= topk:
                        break
            if len(_docids) < topk:  # add dummy docs
                _docids += [get_random_doc_id() for _ in range(topk - len(_docids))]
                _docs += [''] * (topk - len(_docs))
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)  # (bs, topk)
        docs = np.array(docs).reshape(bs, topk)  # (bs, topk)
        return docids, docs


# Add text info in elasticsearch hits, patching the original BM25Search.search method.
def bm25search_search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: int, *args, **kwargs) -> Dict[str, Dict[str, float]]:
    # Index the corpus within elastic-search
    # False, if the corpus has been already indexed
    # corpus: dict of doc_id -> {"title":..., "text":...}
    # queries: dict of query_id -> query_text
    # output: dict mapping query_id -> { doc_id -> (score: float, text: str) }
    if self.initialize:
        self.index(corpus)
        # Sleep for few seconds so that elastic-search indexes the docs properly
        time.sleep(self.sleep_for)

    #retrieve results from BM25
    query_ids = list(queries.keys())
    queries = [queries[qid] for qid in query_ids]

    final_results: Dict[str, Dict[str, Tuple[float, str]]] = {}
    for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que', disable=kwargs.get('disable_tqdm', False)):
        query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
        results = self.es.lexical_multisearch(
            texts=queries[start_idx:start_idx+self.batch_size],
            top_hits=top_k)
        for (query_id, hit) in zip(query_ids_batch, results):
            scores = {}
            for corpus_id, score, text in hit['hits']:
                scores[corpus_id] = (score, text) # including text
                final_results[query_id] = scores

    return final_results

BM25Search.search = bm25search_search


def elasticsearch_lexical_multisearch(self, texts: List[str], top_hits: int, skip: int = 0) -> Dict[str, object]:
    """Multiple Query search in Elasticsearch

    Args:
        texts (List[str]): Multiple query texts
        top_hits (int): top k hits to be retrieved
        skip (int, optional): top hits to be skipped. Defaults to 0.

    Returns:
        Dict[str, object]: Hit results
    """
    request = []

    assert skip + top_hits <= 10000, "Elastic-Search Window too large, Max-Size = 10000"

    for text in texts:
        req_head = {"index" : self.index_name, "search_type": "dfs_query_then_fetch"}
        req_body = {
            "_source": True, # No need to return source objects
            "query": {
                "multi_match": {
                    "query": text, # matching query with both text and title fields
                    "type": "best_fields",
                    "fields": [self.title_key, self.text_key],
                    "tie_breaker": 0.5
                    }
                },
            "size": skip + top_hits, # The same paragraph will occur in results
            }
        request.extend([req_head, req_body])

    res = self.es.msearch(body = request)

    result = []
    for resp in res["responses"]:
        responses = resp["hits"]["hits"][skip:] if 'hits' in resp else []

        hits = []
        for hit in responses:
            hits.append((hit["_id"], hit['_score'], hit['_source']['txt']))

        result.append(self.hit_template(es_res=resp, hits=hits))
    return result

ElasticSearch.lexical_multisearch = elasticsearch_lexical_multisearch


def elasticsearch_hit_template(self, es_res: Dict[str, object], hits: List[Tuple[str, float]]) -> Dict[str, object]:
    """Hit output results template

    Args:
        es_res (Dict[str, object]): Elasticsearch response
        hits (List[Tuple[str, float]]): Hits from Elasticsearch

    Returns:
        Dict[str, object]: Hit results
    """
    result = {
        'meta': {
            'total': es_res['hits']['total']['value'] if 'hits' in es_res else None,
            'took': es_res['took'] if 'took' in es_res else None,
            'num_hits': len(hits)
        },
        'hits': hits,
    }
    return result

ElasticSearch.hit_template = elasticsearch_hit_template


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
tokenizer.pad_token = tokenizer.eos_token
bm25_retriever = BM25(
    tokenizer = tokenizer, 
    index_name = "wiki", 
    engine = "elasticsearch",
)

def bm25_retrieve(question, topk):
    docs_ids, docs = bm25_retriever.retrieve(
        [question], 
        topk=topk, 
        max_query_length=256
    )
    return docs[0].tolist()


class SGPT:
    cannot_encode_id = [6799132, 6799133, 6799134, 6799135, 6799136, 6799137, 6799138, 6799139, 8374206, 8374223, 9411956, 
        9885952, 11795988, 11893344, 12988125, 14919659, 16890347, 16898508]
    # 这些向量是 SGPT 不能 encode 的，设置为全 0 向量，点积为 0，不会影响检索

    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file
    ):
        logger.info(f"Loading SGPT model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path, device_map="auto")
        self.model.eval()
        self.SPECB_QUE_BOS = self.tokenizer.encode("[", add_special_tokens=False)[0]
        self.SPECB_QUE_EOS = self.tokenizer.encode("]", add_special_tokens=False)[0]
        self.SPECB_DOC_BOS = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.SPECB_DOC_EOS = self.tokenizer.encode("}", add_special_tokens=False)[0]

        logger.info(f"Building SGPT indexes")

        self.p_reps = []

        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        dir_point = 0
        pbar = tqdm.tqdm(total=len(dir_names))
        split_parts = 0
        while True:
            split_parts += 1
            flag = False
            for d in dir_names:
                if d.startswith(f'{split_parts}_'):
                    flag = True
                    break
            if flag == False:
                break

        for i in range(split_parts):
            start_point = dir_point
            while dir_point < len(dir_names) and dir_names[dir_point].startswith(f'{i}_'):
                # filename = dir_names[dir_point]
                dir_point += 1
            cnt = dir_point - start_point
            for j in range(cnt):
                filename = f"{i}_{j}.pt"
                pbar.update(1)
                tp = torch.load(os.path.join(encode_file_path, filename))

                def get_norm(matrix):
                    norm = matrix.norm(dim=1)
                    if 0 in norm:
                        norm = torch.where(norm == 0, torch.tensor(1.0), norm)
                    return norm.view(-1, 1)

                sz = tp.shape[0] // 2
                tp1 = tp[:sz, :]
                tp2 = tp[sz:, :]
                self.p_reps.append((tp1.cuda(i), get_norm(tp1).cuda(i)))
                self.p_reps.append((tp2.cuda(i), get_norm(tp2).cuda(i)))
            
        docs_file = passage_file
        df = pd.read_csv(docs_file, delimiter='\t')
        self.docs = list(df['text'])


    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        # Add special brackets & pay attention to them
        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            if is_query:
                seq.insert(0, self.SPECB_QUE_BOS)
                seq.append(self.SPECB_QUE_EOS)
            else:
                seq.insert(0, self.SPECB_DOC_BOS)
                seq.append(self.SPECB_DOC_EOS)
            att.insert(0, 1)
            att.append(1)
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state

        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        q_reps = self.get_weightedmean_embedding(
            self.tokenize_with_specb(queries, is_query=True)
        )
        q_reps.requires_grad_(False)
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        prev_count = 0
        for p_rep, p_rep_norm in self.p_reps:
            sim = p_rep @ q_reps_trans.to(p_rep.device)
            sim = sim / p_rep_norm
            # print(sim.shape)
            topk_values, topk_indices = torch.topk(sim, k=topk, dim=0)
            # print(torch.transpose(topk_values, 0, 1)[0])
            # print(torch.transpose(topk_indices, 0, 1)[0] + prev_count)
            topk_values_list.append(topk_values.to('cpu'))
            topk_indices_list.append(topk_indices.to('cpu') + prev_count)
            prev_count += p_rep.shape[0]

        all_topk_values = torch.cat(topk_values_list, dim=0)
        global_topk_values, global_topk_indices = torch.topk(all_topk_values, k=topk, dim=0)

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                idx = global_topk_indices[j][qid].item()
                fid, rk = idx // topk, idx % topk
                psg = self.docs[topk_indices_list[fid][rk][qid]]
                ret.append(psg)
            psgs.append(ret)
        return psgs