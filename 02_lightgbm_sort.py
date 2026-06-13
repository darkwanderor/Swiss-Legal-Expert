#!/usr/bin/env python3
import json
import os
import re
import pandas as pd
from collections import defaultdict
import lightgbm as lgb

def find_dataset(file_or_dir_name):
    base = "/kaggle/input"
    if not os.path.exists(base): return f"./{file_or_dir_name}"
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return os.path.join(root, file_or_dir_name)
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

try:
    LGBM_WEIGHTS_FILE = find_dataset('lgbm_weights.txt')
except FileNotFoundError:
    LGBM_WEIGHTS_FILE = None

try:
    TEST_CSV = find_dataset('test.csv')
except FileNotFoundError:
    TEST_CSV = "test.csv"

INPUT_JSONL = '/kaggle/working/retrieval_candidates.json'
OUTPUT_JSONL = '/kaggle/working/lightgbm_top20.jsonl'

def extract_advanced_laws(text: str) -> list:
    CODE_MAP = {"CPP": "StPO", "CPC": "ZPO", "CP": "StGB", "CC": "ZGB", "CO": "OR", "LCR": "SVG", "LTF": "BGG", "LPGA": "ATSG", "LAI": "IVG", "LDIP": "IPRG", "Cst.": "BV", "Cst": "BV"}
    LAW_RE_ADV = re.compile(r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)\s*[0-9a-z]+)*)\s*(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b", re.I)
    def norm_law_parts(num, tail, code):
        code = CODE_MAP.get(code.strip(), code.strip())
        tail = re.sub(r"\bal\.\s*", "Abs. ", tail or "", flags=re.I)
        tail = re.sub(r"\blet\.\s*", "lit. ", tail, flags=re.I)
        tail = re.sub(r"\s+", " ", tail).strip(" ,.")
        return f"Art. {num.strip()} {tail} {code}".replace("  ", " ").strip()
    laws = set()
    for m in LAW_RE_ADV.finditer(text): laws.add(norm_law_parts(m.group(1), m.group(2), m.group(3)))
    return list(laws)

def main():
    print("="*60)
    print("[INFO] Stage 02: LightGBM Residual Sort")
    print("="*60)
    
    if not os.path.exists(INPUT_JSONL):
        raise FileNotFoundError(f"Missing {INPUT_JSONL}.")
        
    df_queries = pd.read_csv(TEST_CSV)
    query_dict = dict(zip(df_queries['query_id'], df_queries['query']))
    
    grouped = defaultdict(list)
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                grouped[c['query_id']].append(c)
                
    if not LGBM_WEIGHTS_FILE:
        print("[WARN] No LGBM weights found. Outputting fallback raw RRF Top 20.")
        with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for qid, cands in grouped.items():
                for c in sorted(cands, key=lambda x: x.get('fusion_score', 0), reverse=True)[:20]:
                    f.write(json.dumps(c, ensure_ascii=False) + '\n')
        return
        
    print(f"[INFO] Loading LightGBM weights from {LGBM_WEIGHTS_FILE}...")
    lgb_ranker = lgb.Booster(model_file=LGBM_WEIGHTS_FILE)
    
    out_file = open(OUTPUT_JSONL, 'w', encoding='utf-8')
    
    for qid, cands in grouped.items():
        q = query_dict.get(qid, "")
        literal_cites = set(extract_advanced_laws(q))
        q_lower = q.lower()
        q_is_criminal = any(kw in q_lower for kw in ['stgb', 'stpo'])
        q_is_civil = any(kw in q_lower for kw in ['zgb', 'or', 'contract'])
        q_art_set = set(re.findall(r'art\.?\s*(\d+[a-z]*)', q_lower))
        
        X = []
        for c in cands:
            cit = c.get('metadata', {}).get('citation', '')
            cit_lower = cit.lower()
            
            c_is_criminal = 'stgb' in cit_lower or 'stpo' in cit_lower
            c_is_civil = 'zgb' in cit_lower or ' or ' in cit_lower or cit_lower.endswith(' or')
            domain_clash = 1 if (q_is_criminal and c_is_civil) or (q_is_civil and c_is_criminal) else 0
            
            c_art_matches = re.findall(r'art\.?\s*(\d+[a-z]*)', cit_lower)
            article_match_safety = 1 if set(c_art_matches).intersection(q_art_set) else 0
            is_literal = 1 if cit in literal_cites else 0
            
            fields = {h['field']: h for h in c.get('metadata', {}).get('hits', []) if h['field'] in ['normal_query', 'meta_searchterm', 'keywords', 'fulltext', 'citations', 'sparse']}
            
            feat = [
                c.get('fusion_score', 0.0),
                fields.get('normal_query', {}).get('score', 0.0),
                fields.get('meta_searchterm', {}).get('score', 0.0),
                fields.get('keywords', {}).get('score', 0.0),
                fields.get('fulltext', {}).get('score', 0.0),
                fields.get('citations', {}).get('score', 0.0),
                fields.get('sparse', {}).get('score', 0.0),
                fields.get('normal_query', {}).get('rank', 2000),
                fields.get('meta_searchterm', {}).get('rank', 2000),
                fields.get('keywords', {}).get('rank', 2000),
                fields.get('fulltext', {}).get('rank', 2000),
                fields.get('citations', {}).get('rank', 2000),
                fields.get('sparse', {}).get('rank', 2000),
                is_literal,
                1 if cit.startswith("BGE") else 0,
                1 if 'bgg' in cit_lower else 0,
                domain_clash,
                article_match_safety
            ]
            X.append(feat)
            
        preds = lgb_ranker.predict(X)
        for i, c in enumerate(cands): c['lgbm_score'] = float(preds[i])
        cands.sort(key=lambda x: x['lgbm_score'], reverse=True)
        
        for i, c in enumerate(cands[:20]):
            c['rank'] = i + 1
            out_file.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    out_file.close()
    print(f"[INFO] Saved Top 20 candidates per query to {OUTPUT_JSONL}")

if __name__ == '__main__': main()
