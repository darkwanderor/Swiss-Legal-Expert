#!/usr/bin/env python3
import argparse, csv, heapq, json, os, re, time, faiss
from pathlib import Path
import numpy as np
import scipy.sparse
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

FIELDS=("normal_query","meta_searchterm","keywords","fulltext","citations")
LAW_RE=re.compile(r"\b(?:Art\.|BGE|ATF|BGer|TF|[1-9][A-Z]?_[0-9]+/[0-9]{4}|[0-9][A-Z]_[0-9]+/[0-9]{4}|[0-9A-Z]{1,3}_[0-9]+/[0-9]{4})\b", re.I)
CIT_RE=re.compile(r"\b(?:Art\.\s*[^,;.()]{1,80}|BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.a-z]+)?|(?:\d[A-Z]?|[A-Z])_[0-9]+/[0-9]{4}(?:\s+E\.\s*[\d.a-z]+)?)")

EXTERNAL_CHUNKS_DIRS = [
    '/kaggle/input/datasets/darkwanderor/chunk-1/corpus_index',
    '/kaggle/input/datasets/dinosaurking/chunk-2/corpus_index',
    '/kaggle/input/datasets/dunkelwald/chunk-3/corpus_index',
    '/kaggle/input/datasets/dinosaurking/chunk-4/corpus_index',
    '/kaggle/input/datasets/dunkelwald/chunk-5/corpus_index'
]

# --- KAGGLE CONFIGURATION ---
def find_dataset(file_or_dir_name):
    base = "/kaggle/input"
    if not os.path.exists(base): return f"./{file_or_dir_name}"
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return os.path.join(root, file_or_dir_name)
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

INDEX_DIR = '/kaggle/working/corpus_index'
# The dataset input will be dynamically determined via argparse.
OUT_JSONL = '/kaggle/working/retrieval_candidates.jsonl'

try:
    EMBEDDING_MODEL_PATH = find_dataset('Qwen3-Embedding-8B')
except FileNotFoundError:
    EMBEDDING_MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/8b/1"

try:
    LLM_MODEL_PATH = find_dataset('Qwen2.5-14B-Instruct')
except FileNotFoundError:
    LLM_MODEL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1"
# ----------------------------

def clean(x): return re.sub(r"\s+"," ",str(x or "")).strip()

def generate_true_hyde(queries, llm_model_path):
    print(f"[INFO] Loading Qwen3-4B from {llm_model_path} for True HyDE Generation...")
    sys_msg = "You are a Supreme Court Swiss Judge. Generate a HYPOTHETICAL Swiss legal judgment resolving this factual case. Output ONLY a raw JSON: {\"sentences\": [\"Sentence 1\", \"Sentence 2\"], \"keywords\": \"list of specific factual nouns\"}"
    
    prompts = [
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\nQuery: {q[:1500]}<|im_end|>\n<|im_start|>assistant\n{{"
        for q in queries
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_path, trust_remote_code=True, local_files_only=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    llm = AutoModelForCausalLM.from_pretrained(
        llm_model_path, device_map="auto", torch_dtype=torch.float16,
        attn_implementation="sdpa", trust_remote_code=True, local_files_only=True
    ).eval()
    
    outputs = []
    batch_size = 4
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=4096-250)
        inputs = {k: v.to(llm.device) for k, v in inputs.items()}
        with torch.inference_mode():
            gen_tokens = llm.generate(**inputs, max_new_tokens=250, temperature=0.1, do_sample=False)
        input_len = inputs['input_ids'].shape[1]
        decoded = tokenizer.batch_decode(gen_tokens[:, input_len:], skip_special_tokens=True)
        outputs.extend(decoded)
        
    del llm
    torch.cuda.empty_cache()
    
    meta_texts, keyword_texts = [], []
    for out_text in outputs:
        text = "{" + out_text
        try:
            m = re.search(r'\{.*\}', text.replace('\n', ' '), flags=re.DOTALL)
            parsed = json.loads(m.group(0)) if m else {}
            sents = parsed.get("sentences", [])
            meta_texts.append(" ".join(sents) if isinstance(sents, list) else str(sents))
            keyword_texts.append(str(parsed.get("keywords", "")))
        except:
            meta_texts.append("")
            keyword_texts.append("")
            
    return meta_texts, keyword_texts

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding: return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), sequence_lengths]

@torch.inference_mode()
def load_encoder(model_path):
    tok=AutoTokenizer.from_pretrained(model_path, padding_side="left", local_files_only=True)
    model=AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, local_files_only=True, low_cpu_mem_usage=True, device_map="auto", attn_implementation="sdpa").eval()
    return tok, model

@torch.inference_mode()
def encode(texts, tok, model, max_length=1024, batch_size=16):
    outs=[]
    for i in range(0,len(texts),batch_size):
        batch=[t if clean(t) else ' ' for t in texts[i:i+batch_size]]
        inp=tok(batch,padding=True,truncation=True,max_length=max_length,return_tensors='pt').to('cuda')
        out=model(**inp)
        emb=last_token_pool(out.last_hidden_state, inp['attention_mask'])
        emb=F.normalize(emb,p=2,dim=1)
        outs.append(emb.detach().cpu().float().numpy())
    return np.vstack(outs)

def find_chunk_path(base_dir, chunk_idx):
    p = os.path.join(base_dir, f"dense_chunk_{chunk_idx}.index")
    if os.path.exists(p): return p
    for d in EXTERNAL_CHUNKS_DIRS:
        p = os.path.join(d, f"dense_chunk_{chunk_idx}.index")
        if os.path.exists(p): return p
    return None

def topk_gpu_distributed(base_dir, embs_dict, topk, device='cuda'):
    # embs_dict contains encoded Qs for all 5 fields. Shape: (b, dim)
    b = len(next(iter(embs_dict.values())))
    heaps = {f: [[] for _ in range(b)] for f in FIELDS}
    
    Q_np_dict = {f: embs_dict[f].astype(np.float32) for f in FIELDS}
    
    chunk_idx = 0
    global_offset = 0
    while True:
        chunk_path = find_chunk_path(base_dir, chunk_idx)
        if not chunk_path: break
        
        print(f"[INFO] Searching FAISS Index {chunk_path}...")
        index = faiss.read_index(chunk_path)
        chunk_size = index.ntotal
        
        if device == 'cuda' and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        take = min(topk, chunk_size)
        
        for field in FIELDS:
            D, I = index.search(Q_np_dict[field], take)
            for qi in range(b):
                heap = heaps[field][qi]
                for k_idx in range(take):
                    sc = float(D[qi, k_idx])
                    loc = int(I[qi, k_idx])
                    gi = global_offset + loc
                    if len(heap) < topk: heapq.heappush(heap, (sc, gi))
                    elif sc > heap[0][0]: heapq.heapreplace(heap, (sc, gi))
                    
        del index
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        global_offset += chunk_size
        chunk_idx += 1
        
    return {f: [sorted(h, reverse=True) for h in heaps[f]] for f in FIELDS}

def topk_sparse(queries, vectorizer_path, sparse_mat_path, topk):
    with open(vectorizer_path, 'rb') as f: vectorizer = pickle.load(f)
    Q = vectorizer.transform(queries)
    mat = scipy.sparse.load_npz(sparse_mat_path)
    scores = Q.dot(mat.T).tocsr()
    
    b = Q.shape[0]
    results = []
    for qi in range(b):
        row = scores.getrow(qi)
        indices = row.indices
        data = row.data
        if len(indices) == 0:
            results.append([])
            continue
        take = min(topk, len(indices))
        top_idx = np.argpartition(data, -take)[-take:]
        hits = [(float(data[i]), int(indices[i])) for i in top_idx]
        results.append(sorted(hits, reverse=True))
    return results

def rrf_fuse(field_hits, weights, rrf_k=60, fusion_top=1000):
    outs=[]
    for qi in range(len(next(iter(field_hits.values())))):
        scores={}; details={}
        for field,hits_by_q in field_hits.items():
            w=weights.get(field,1.0)
            for rank,(score,gi) in enumerate(hits_by_q[qi],1):
                scores[gi]=scores.get(gi,0.0)+w/(rrf_k+rank)
                details.setdefault(gi,[]).append({'field':field,'rank':rank,'score':score})
        fused=sorted(scores.items(), key=lambda x:x[1], reverse=True)[:fusion_top]
        outs.append((fused,details))
    return outs

def load_metadata(meta_path, wanted):
    wanted=set(wanted); out={}
    with open(meta_path,'r', encoding='utf-8') as f:
        for line in f:
            rec=json.loads(line); gi=rec['global_idx']
            if gi in wanted:
                out[gi]=rec
                if len(out)==len(wanted): break
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test', choices=['train', 'test', 'val'], help='Dataset to run retrieval on')
    args = parser.parse_args()
    
    try:
        input_csv = find_dataset(f'{args.dataset}.csv')
    except FileNotFoundError:
        input_csv = f'{args.dataset}.csv'
        
    index=Path(INDEX_DIR)
    
    qrows=[]
    with open(input_csv,newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        for r in rd: qrows.append(r)
        
    qids=[r['query_id'] for r in qrows]
    raw=[r['query'] for r in qrows]
    
    print("="*60)
    print("[INFO] Stage 01: Asymmetric Multi-Vector Retrieval")
    print("="*60)
    
    meta, keys = generate_true_hyde(raw, LLM_MODEL_PATH)
    
    query_texts={
        'normal_query':raw,
        'meta_searchterm':meta,
        'keywords':keys,
        'fulltext':raw,
        'citations':[clean(' '.join(m.group(0) for m in CIT_RE.finditer(q))) or q for q in raw]
    }
    
    tok,model=load_encoder(EMBEDDING_MODEL_PATH)
    
    embs={}
    for field in FIELDS:
        print(f"[INFO] Encoding '{field}' queries...")
        embs[field]=encode(query_texts[field],tok,model,1024,16)
        
    del model, tok
    torch.cuda.empty_cache()
    
    # Run the distributed chunk search
    field_hits = topk_gpu_distributed(str(index), embs, 1000)
        
    print(f"[INFO] Searching sparse TF-IDF for queries...")
    sparse_path = str(index/'sparse_corpus.npz')
    vec_path = str(index/'vectorizer.pkl')
    field_hits['sparse'] = topk_sparse(raw, vec_path, sparse_path, 1000)
        
    weights={'normal_query':0.4,'meta_searchterm':1.25,'keywords':0.85,'fulltext':1.35,'citations':0.15,'sparse':0.50}
    fused=rrf_fuse(field_hits,weights,60,1000)
    
    wanted=set()
    for f,details in fused:
        wanted.update(gi for gi,score in f)
        
    print("[INFO] Extracting metadata...")
    metas=load_metadata(index/'metadata.jsonl',wanted)
    
    outp=Path(OUT_JSONL)
    outp.parent.mkdir(parents=True,exist_ok=True)
    with outp.open('w',encoding='utf-8') as fo:
        for qi,(f,details) in enumerate(fused):
            for rank,(gi,fs) in enumerate(f,1):
                rec={
                    'query_id':qids[qi],
                    'rank':rank,
                    'global_idx':gi,
                    'fusion_score':fs,
                    'metadata':metas.get(gi,{'global_idx':gi, 'citation': metas.get(gi,{}).get('citation', '')}),
                    'doc_id':metas.get(gi,{}).get('doc_id')
                }
                rec['metadata']['hits'] = details.get(gi, [])
                fo.write(json.dumps(rec,ensure_ascii=False)+'\n')
                
    print(f"[INFO] Stage 01 Complete. Distributed Asymmetric Retrieval Output saved to {OUT_JSONL}")

if __name__=='__main__': main()
