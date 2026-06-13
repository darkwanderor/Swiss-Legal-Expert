#!/usr/bin/env python3
import argparse
import json
import re
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
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
    LLM_MODEL = find_dataset('Qwen3-4B-Instruct')
except FileNotFoundError:
    LLM_MODEL = "/kaggle/input/models/urvishp80/qwenqwen3-4b-instruct-2507/transformers/default/1"

INPUT_JSONL = Path('/kaggle/working/qwen_input_top20.jsonl')
OUTPUT_JSONL = Path('/kaggle/working/qwen_scored_top20.jsonl')

SYSTEM_PROMPT = (
    "Du bist ein schneller juristischer Reranker fuer Schweizer Gerichtsentscheide.\n"  # You are a fast legal reranker for Swiss court decisions.
    "\n"
    "Aufgabe: Bewerte, ob der Kandidat das Ursprungdokument zur Kaggle-Query ist. Namen, Orte, Daten und Betraege koennen anonymisiert/paraphrasiert sein. Werte vor allem Rechtsgebiet, Verfahrenskontext, Normen, seltene Sachverhaltskombination und konkrete Rechtsfrage.\n"  # Task: Evaluate whether the candidate is the source document for the Kaggle query. Names, places, dates, and amounts may be anonymized/paraphrased. Evaluate primarily the legal area, procedural context, norms, rare factual combinations, and specific legal question.
    "\n"
    "Score 0-10:\n"
    "10 = praktisch sicher derselbe Fall.\n"  # 10 = practically certain the same case.
    "8-9 = sehr wahrscheinlich.\n"  # 8-9 = very likely.
    "6-7 = naher Schwesterfall.\n"  # 6-7 = close sister case.
    "4-5 = nur gleiche Rechtsfamilie.\n"  # 4-5 = only same legal family.
    "0-3 = wahrscheinlich falsch.\n"  # 0-3 = probably wrong.
    "\n"
    "Antworte NUR als kurzes JSON: {\"score\":0.0,\"confidence\":0.0}\n"  # Answer ONLY as short JSON: {"score":0.0,"confidence":0.0}
)

def clean(s): return re.sub(r"\s+", " ", str(s or "")).strip()

def prompt_for_row(row):
    return "\n".join([
        "ORIGINAL-QUERY / KAGGLE-QUERY:",
        clean(row.get("query", "")),
        "",
        "KANDIDAT:",
        f"doc_id: {row.get('doc_id')}",
        "document:",
        str(row.get("document_text", "")).strip(),
        "",
        "Gib nur JSON zurueck: {\"score\":0.0,\"confidence\":0.0}",
    ])

def make_messages(tokenizer, row):
    user = prompt_for_row(row)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_one(text):
    text = (text or "").strip()
    obj = None
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start: obj = json.loads(text[start:end + 1])
    except: pass
    
    if not isinstance(obj, dict):
        m = re.search(r"(?:score|Score|SCORE)\D{0,20}([0-9]+(?:\.[0-9]+)?)", text)
        obj = {"score": float(m.group(1)) if m else 0.0, "confidence": 0.0}
        
    score = max(0.0, min(10.0, float(obj.get("score", 0) or 0)))
    conf = max(0.0, min(1.0, float(obj.get("confidence", 0) or 0)))
    return {"score": score, "confidence": conf}

def main():
    print("="*60)
    print("[INFO] Stage 04: LLM Pointwise Scorer")
    print("="*60)
    
    if not INPUT_JSONL.exists(): raise FileNotFoundError(f"Missing {INPUT_JSONL}")
    
    rows = []
    with INPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
            
    print(f"[INFO] Loading Qwen3-4B from {LLM_MODEL} for {len(rows)} candidates...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, local_files_only=True, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()
    
    input_device = next(model.parameters()).device
    prompts = [make_messages(tokenizer, r) for r in rows]
    
    texts = []
    t0 = time.time()
    batch_size = 4
    
    print(f"[INFO] Generating scores in batches of {batch_size}...")
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=6000).to(input_device)
        try:
            with torch.inference_mode():
                out = model.generate(**inputs, do_sample=False, max_new_tokens=64, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            new = out[:, inputs["input_ids"].shape[1]:]
            texts.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        except torch.OutOfMemoryError:
            print("[WARN] OOM Error during inference! Recovering batch.")
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            texts.extend(['{"score":0.0,"confidence":0.0}'] * len(batch))
            
    elapsed = time.time() - t0
    parsed = [parse_one(t) for t in texts]
    
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as f:
        for row, item in zip(rows, parsed):
            rec = {
                "query_id": row["query_id"],
                "rank": row["rank"],
                "doc_id": row["doc_id"],
                "fusion_score": row["fusion_score"],
                "lgbm_score": row.get("lgbm_score"),
                "qwen_score": item["score"],
                "qwen_confidence": item["confidence"],
                "global_idx": row["global_idx"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"[INFO] Generated {len(rows)} scores in {elapsed:.1f}s. Saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
