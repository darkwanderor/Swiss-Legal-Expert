#!/usr/bin/env python3
import json
import os
import re
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

# =======================================================================
# KAGGLE INPUT DATASETS (AUTO-DETECTED)
# =======================================================================
def find_dataset(file_or_dir_name):
    base = "/kaggle/input"
    if not os.path.exists(base): return f"./{file_or_dir_name}"
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return os.path.join(root, file_or_dir_name)
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

try:
    TRAIN_CSV = find_dataset('train.csv')
except FileNotFoundError:
    TRAIN_CSV = "train.csv"

CANDIDATES_JSON = '/kaggle/working/retrieval_candidates.json'
OUTPUT_FILE = '/kaggle/working/lgbm_weights.txt'

FEATURE_COLS = [
    'fusion_score',
    'normal_score', 'meta_score', 'keywords_score', 'fulltext_score', 'citations_score', 'sparse_score',
    'normal_rank', 'meta_rank', 'keywords_rank', 'fulltext_rank', 'citations_rank', 'sparse_rank',
    'is_literal', 'is_bge', 'is_bgg', 'domain_clash', 'article_match_safety'
]

# =======================================================================
# HEURISTICS
# =======================================================================
def extract_advanced_laws(text: str) -> list:
    CODE_MAP = {"CPP": "StPO", "CPC": "ZPO", "CP": "StGB", "CC": "ZGB", "CO": "OR", "LCR": "SVG", "LTF": "BGG", "LPGA": "ATSG", "LAI": "IVG", "LDIP": "IPRG", "Cst.": "BV", "Cst": "BV"}
    LAW_RE_ADV = re.compile(r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)\s*[0-9a-z]+)*)\s*(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b", re.I)
    COUR_RES = [re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+\s+E\.\s*[\dA-Za-z_.-]+"), re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+"), re.compile(r"\b\d+[A-Z]_\d+/\d{4}\s+E\.\s*[\dA-Za-z_.-]+"), re.compile(r"\b\d+[A-Z]_\d+/\d{4}")]
    
    def norm_law_parts(num, tail, code):
        code = CODE_MAP.get(code.strip(), code.strip())
        tail = re.sub(r"\bal\.\s*", "Abs. ", tail or "", flags=re.I)
        tail = re.sub(r"\blet\.\s*", "lit. ", tail, flags=re.I)
        tail = re.sub(r"\s+", " ", tail).strip(" ,.")
        return f"Art. {num.strip()} {tail} {code}".replace("  ", " ").strip()
        
    laws = set()
    for m in LAW_RE_ADV.finditer(text):
        laws.add(norm_law_parts(m.group(1), m.group(2), m.group(3)))
    for rx in COUR_RES:
        for m in rx.finditer(text):
            laws.add(re.sub(r"\s+", " ", m.group(0)).strip(" .;:,"))
    return list(laws)

# =======================================================================
# MAIN
# =======================================================================
def main():
    print("="*60)
    print("[INFO] Stage 00b: LightGBM Residual Ranker Training")
    print("="*60)
    
    if not os.path.exists(CANDIDATES_JSON):
        raise FileNotFoundError(f"Missing {CANDIDATES_JSON}.")
        
    df_train = pd.read_csv(TRAIN_CSV)
    df_train['gold_citations'] = df_train['gold_citations'].astype(str)
    
    def parse_golds(s): return [c.strip() for c in str(s).split(";") if c.strip()]
    gold_dict = dict(zip(df_train['query_id'], df_train['gold_citations'].apply(parse_golds)))
    query_dict = dict(zip(df_train['query_id'], df_train['query']))
    
    grouped = defaultdict(list)
    print("[INFO] Loading candidates...")
    with open(CANDIDATES_JSON, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                grouped[c['query_id']].append(c)
        
    X, y, group_qids = [], [], []
    
    for qi, (qid, cands) in enumerate(grouped.items()):
        if qid not in gold_dict: continue
        
        q = query_dict[qid]
        golds_set = set(gold_dict[qid])
        
        literal_cites = set(extract_advanced_laws(q))
        q_lower = q.lower()
        q_is_criminal = any(kw in q_lower for kw in ['stgb', 'stpo', 'murder', 'penalty', 'theft', 'strafe'])
        q_is_civil = any(kw in q_lower for kw in ['zgb', 'or', 'contract', 'marriage', 'divorce', 'liability'])
        q_art_set = set(re.findall(r'art\.?\s*(\d+[a-z]*)', q_lower))
        
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
            y.append(1 if cit in golds_set else 0)
            group_qids.append(qi)
            
    print(f"[INFO] Feature matrix built. X shape: ({len(X)}, {len(X[0]) if X else 0}), Groups: {len(group_qids)}")
    
    train_data = lgb.Dataset(X, label=y, group=group_qids)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [10, 20],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'verbose': -1
    }
    
    print("[INFO] Initiating LightGBM LambdaRank training...")
    lgb_ranker = lgb.train(
        params,
        train_data,
        num_boost_round=150
    )
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    lgb_ranker.save_model(OUTPUT_FILE)
    
    print(f"[INFO] Training Complete. Model weights saved to {OUTPUT_FILE}")

if __name__ == '__main__':
    main()
