#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

# =======================================================================
# KAGGLE INPUT DATASETS (AUTO-DETECTED)
# =======================================================================
def find_dataset(file_or_dir_name):
    base = "/kaggle/input"
    if not os.path.exists(base): return Path(f"./{file_or_dir_name}")
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return Path(root) / file_or_dir_name
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

try:
    TEST_CSV = find_dataset('test.csv')
except FileNotFoundError:
    TEST_CSV = Path("test.csv")

try:
    CORPUS_TEXTS_PARQUET = find_dataset('corpus_texts.parquet')
except FileNotFoundError:
    CORPUS_TEXTS_PARQUET = Path("/kaggle/working/corpus_index/corpus_texts.parquet")

INPUT_JSONL = Path('/kaggle/working/lightgbm_top20.jsonl')
OUTPUT_JSONL = Path('/kaggle/working/qwen_input_top20.jsonl')

def main():
    print("="*60)
    print("[INFO] Stage 03: Fulltext Cache (Top 20 -> Qwen Input)")
    print("="*60)
    
    if not INPUT_JSONL.exists():
        raise FileNotFoundError(f"Missing {INPUT_JSONL}")
        
    candidates = []
    wanted_globals = set()
    with INPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            candidates.append(row)
            wanted_globals.add(row["global_idx"])
                
    print(f"[INFO] Loaded {len(candidates)} candidate rows. Need to fetch texts for {len(wanted_globals)} unique global indices...")
    
    # Fast load using Pandas (bypassing the slow PyArrow RowGroup search)
    print(f"[INFO] Loading chunks from {CORPUS_TEXTS_PARQUET}...")
    corpus_df = pd.read_parquet(CORPUS_TEXTS_PARQUET)
    
    ORIGINAL_TEXTS_PARQUET = CORPUS_TEXTS_PARQUET.parent / "original_texts.parquet"
    print(f"[INFO] Loading original unchunked texts from {ORIGINAL_TEXTS_PARQUET}...")
    try:
        original_df = pd.read_parquet(ORIGINAL_TEXTS_PARQUET)
        original_text_map = dict(zip(original_df['citation'], original_df['text']))
    except Exception as e:
        print(f"[WARNING] Could not load {ORIGINAL_TEXTS_PARQUET}: {e}. Falling back to chunk only.")
        original_text_map = {}
    
    # -------------------------------------------------------------
    # BUILD QWEN INPUT JSONL
    # -------------------------------------------------------------
    grouped = defaultdict(list)
    for row in candidates:
        grouped[row["query_id"]].append(row)
        
    import csv
    queries = {}
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            queries[row["query_id"]] = row["query"]
            
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    missing_text = 0
    
    with OUTPUT_JSONL.open("w", encoding="utf-8") as out:
        for qid in sorted(grouped):
            query = queries.get(qid, "")
            cands = sorted(grouped[qid], key=lambda x: x.get('rank', 999))
            
            for row in cands:
                g_idx = int(row["global_idx"])
                if g_idx < len(corpus_df):
                    chunk_text = corpus_df.iloc[g_idx]['full_injected_text']
                    citation = corpus_df.iloc[g_idx]['citation']
                else:
                    chunk_text = ""
                    citation = ""
                    
                if not chunk_text: 
                    missing_text += 1
                    text = ""
                else:
                    orig_text = str(original_text_map.get(citation, ""))
                    head_len, tail_len = 1000, 1000
                    
                    if len(orig_text) <= head_len + tail_len + len(chunk_text):
                        text = orig_text if orig_text else chunk_text
                    else:
                        head = orig_text[:head_len]
                        tail = orig_text[-tail_len:]
                        text = (
                            "[ANFANG DES DOKUMENTS]\n" + head +
                            "\n\n...[GEKUERZT]...\n\n" +
                            "[RELEVANTER ABSCHNITT]\n" + chunk_text +
                            "\n\n...[GEKUERZT]...\n\n" +
                            "[ENDE DES DOKUMENTS]\n" + tail
                        )
                
                doc_id = row.get("doc_id", citation)
                
                rec = {
                    "query_id": qid,
                    "query": query,
                    "rank": int(row.get("rank") or 999999),
                    "global_idx": g_idx,
                    "doc_id": doc_id,
                    "fusion_score": row.get("fusion_score"),
                    "lgbm_score": row.get("lgbm_score"),
                    "document_text": text,
                    "document_text_len": len(text),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_rows.append(rec)
                
    print(f"[INFO] Generated {len(out_rows)} prompts in {OUTPUT_JSONL}. Missing texts: {missing_text}")

if __name__ == "__main__":
    main()
