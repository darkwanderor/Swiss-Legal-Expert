#!/usr/bin/env python3
import os
import json
import re
import math
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

def find_dataset(file_or_dir_name):
    base = "/kaggle/input"
    if not os.path.exists(base): return f"./{file_or_dir_name}"
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return os.path.join(root, file_or_dir_name)
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

try:
    LAWS_CSV = find_dataset('laws_de.csv')
    COURT_CSV = find_dataset('court_considerations.csv')
except FileNotFoundError:
    LAWS_CSV = "laws_de.csv"
    COURT_CSV = "court_considerations.csv"

try:
    LLM_MODEL = find_dataset('Qwen3-Embedding-8B')
except FileNotFoundError:
    LLM_MODEL = "/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/8b/1"

OUTPUT_DIR = '/kaggle/working/corpus_index'
WORDS_PER_CHUNK = 800
OVERLAP_WORDS = 200
DENSE_CHUNK_SIZE = 100_000

# --- DISTRIBUTED CONFIGURATION ---
# Edit these variables manually when running across different Kaggle machines.
START_CHUNK = 0
END_CHUNK = None  # Set to an integer to stop early
BUILD_SPARSE = True  # Only set this to True on ONE of your instances to build the TF-IDF matrix
# ---------------------------------

def normalize_legal_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\b(al\.|cpv\.)\b', 'Abs.', text, flags=re.IGNORECASE)
    text = re.sub(r'\bCC\b', 'ZGB', text)
    text = re.sub(r'\bCO\b', 'OR', text)
    text = re.sub(r'\bCP\b', 'StGB', text)
    text = re.sub(r'\bCPP\b', 'StPO', text)
    text = re.sub(r'\bCPC\b', 'ZPO', text)
    text = re.sub(r'\b(ATF|DTF)\b', 'BGE', text)
    return text

def chunk_text_sliding_window(df):
    chunked_data = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking Text"):
        text = normalize_legal_text(str(row.get('text', '')))
        sentences = re.split(r'(?<=[.!?\n])\s+', text)
        
        current_chunk_words = 0
        current_chunk = []
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            if current_chunk_words + sentence_words <= WORDS_PER_CHUNK:
                current_chunk.append(sentence)
                current_chunk_words += sentence_words
            else:
                if current_chunk:
                    chunked_data.append({"citation": row['citation'], "text": " ".join(current_chunk)})
                overlap_words = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    s_words = len(s.split())
                    if overlap_words + s_words <= OVERLAP_WORDS:
                        overlap_chunk.insert(0, s)
                        overlap_words += s_words
                    else: break
                current_chunk = overlap_chunk + [sentence]
                current_chunk_words = overlap_words + sentence_words
                
        if current_chunk:
            chunked_data.append({"citation": row['citation'], "text": " ".join(current_chunk)})
            
    return pd.DataFrame(chunked_data)

def load_encoder():
    print(f"Loading {LLM_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, padding_side="left", local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, attn_implementation="sdpa", trust_remote_code=True, local_files_only=True).eval()
    if torch.cuda.is_available(): model = model.cuda()
    return tokenizer, model

def last_token_pool(last_hidden_states, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding: return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    device = last_hidden_states.device
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=device), sequence_lengths]

@torch.inference_mode()
def encode_texts(tokenizer, model, texts, batch_size=128, max_length=1024):
    vectors = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = [t if str(t).strip() else " " for t in texts[start:start+batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
        outputs = model(**inputs)
        emb = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        vectors.append(emb.detach().cpu().to(torch.float16).numpy())
    return np.concatenate(vectors, axis=0)

def main():
    print("="*60)
    print("[INFO] Stage 00a: Distributed Corpus Indexer")
    print(f"[INFO] Processing Chunks {START_CHUNK} to {END_CHUNK if END_CHUNK is not None else 'END'}")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("[INFO] Loading datasets...")
    laws_df = pd.read_csv(LAWS_CSV).dropna(subset=['text', 'citation'])
    try:
        court_df = pd.read_csv(COURT_CSV, engine='c', low_memory=False).dropna(subset=['text', 'citation'])
    except:
        court_df = pd.DataFrame()
        
    corpus_df = pd.concat([laws_df, court_df], ignore_index=True)
    
    print("[INFO] Saving unchunked original texts cache...")
    corpus_df[['citation', 'text']].drop_duplicates(subset=['citation']).to_parquet(os.path.join(OUTPUT_DIR, 'original_texts.parquet'), index=False)
    
    corpus_df = chunk_text_sliding_window(corpus_df)
    
    corpus_df['full_injected_text'] = "[" + corpus_df['citation'] + "] " + corpus_df['text']
    texts = corpus_df['full_injected_text'].tolist()
    citations = corpus_df['citation'].tolist()
    
    num_rows = len(texts)
    print(f"[INFO] Total Chunks to Index: {num_rows:,}")
    
    if BUILD_SPARSE:
        print("[INFO] Building Global Sparse TF-IDF Index...")
        vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, max_features=2_000_000)
        sparse_matrix = vectorizer.fit_transform(texts)
        scipy.sparse.save_npz(os.path.join(OUTPUT_DIR, 'sparse_corpus.npz'), sparse_matrix)
        import pickle
        with open(os.path.join(OUTPUT_DIR, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
            
        print("Saving Metadata and Texts...")
        with open(os.path.join(OUTPUT_DIR, 'metadata.jsonl'), 'w', encoding='utf-8') as f:
            for i, cit in enumerate(citations):
                f.write(json.dumps({"global_idx": i, "doc_id": cit, "citation": cit}, ensure_ascii=False) + '\n')
                
        # Save texts for Script 03
        corpus_df[['citation', 'full_injected_text']].to_parquet(os.path.join(OUTPUT_DIR, 'corpus_texts.parquet'), index=False)
                
        with open(os.path.join(OUTPUT_DIR, 'manifest.json'), 'w') as f:
            json.dump({"rows": num_rows}, f)
            
        print("[INFO] Sparse Matrix, Metadata, and Texts Complete.")
        
    # -------------------------------------------------------------
    # DISTRIBUTED DENSE CHUNKING
    # -------------------------------------------------------------
    tokenizer, model = load_encoder()
    
    total_chunks = math.ceil(num_rows / DENSE_CHUNK_SIZE)
    start_c = START_CHUNK
    end_c = END_CHUNK if END_CHUNK is not None else total_chunks
    
    for c_idx in range(start_c, end_c):
        print(f"\n[INFO] Processing Dense Chunk {c_idx}/{total_chunks-1}")
        start_idx = c_idx * DENSE_CHUNK_SIZE
        end_idx = min(num_rows, (c_idx + 1) * DENSE_CHUNK_SIZE)
        
        chunk_texts = texts[start_idx:end_idx]
        batch_embs = encode_texts(tokenizer, model, chunk_texts, batch_size=128, max_length=1024)
        
        index_path = os.path.join(OUTPUT_DIR, f'dense_chunk_{c_idx}.index')
        faiss_idx = faiss.IndexFlatIP(batch_embs.shape[1])
        faiss_idx.add(batch_embs.astype(np.float32))
        faiss.write_index(faiss_idx, index_path)
        print(f"[INFO] Saved FAISS Index {index_path} (Shape: {batch_embs.shape})")
        
    print(f"\n[INFO] Indexing Complete for chunks {start_c} to {end_c-1}. Datasets saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
