#!/usr/bin/env python3
import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter

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
    LAWS_CSV = find_dataset('laws_de.csv')
    COURT_CSV = find_dataset('court_considerations.csv')
except FileNotFoundError:
    LAWS_CSV = Path("laws_de.csv")
    COURT_CSV = Path("court_considerations.csv")

INPUT_JSONL = Path('/kaggle/working/qwen_scored_top20.jsonl')
OUTPUT_CSV = Path('/kaggle/working/submission.csv')
AUDIT_JSON = Path('/kaggle/working/audit.json')

CODE_MAP = {
    "CPP": "StPO", "CPC": "ZPO", "CP": "StGB", "CC": "ZGB", "CO": "OR", "LCR": "SVG",
    "LTF": "BGG", "LOAP": "StBOG", "LPGA": "ATSG", "LAI": "IVG", "LDIP": "IPRG", "Cst.": "BV", "Cst": "BV"
}

LAW_RE = re.compile(
    r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)"
    r"((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)\s*[0-9a-z]+)*)"
    r"\s*(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b",
    re.I,
)
CODE_RE = re.compile(r"\b(StPO|StGB|ZGB|OR|ZPO|SVG|BGG|StBOG|ATSG|IVG|IPRG|BV|CPP|CP|CC|CO|CPC|LCR|LTF|LOAP|LPGA|LAI|LDIP|Cst\.?)\b", re.I)
BARE_ART_IN_SEG_RE = re.compile(
    r"(?:\bArt\.\s*|[,;]|\bet\b|\bund\b|\band\b)\s*([0-9]+[a-z]?(?:/[0-9]+)?)"
    r"((?:\s*(?:Abs\.|al\.|alinéa|lit\.|let\.|Ziff\.|ch\.|cifra)\s*[0-9a-z]+)*)",
    re.I,
)
ABS_RANGE_RE = re.compile(r"(Abs\.|al\.|alinéa)\s*([0-9]+)\s*(?:et|und|and)\s*([0-9]+)", re.I)
ARTICLE_ABS_RANGE_RE = re.compile(
    r"\bArt\.\s*([0-9]+[a-z]?(?:/[0-9]+)?)\s*"
    r"(Abs\.|al\.|alinéa)\s*([0-9]+)\s*(?:et|und|and)\s*([0-9]+)"
    r"(?!\s*(?:Abs\.|al\.|alinéa))",
    re.I,
)
COUR_RES = [
    re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+\s+E\.\s*[\dA-Za-z_.-]+"),
    re.compile(r"\bBGE\s+\d+\s+[IVX]+\s+\d+"),
    re.compile(r"\b\d+[A-Z]_\d+/\d{4}\s+E\.\s*[\dA-Za-z_.-]+"),
    re.compile(r"\b\d+[A-Z]_\d+/\d{4}"),
]
CONSID_LIST = r"[\dA-Za-z_.-]+(?:\s*(?:,|et|und|and)\s*[\dA-Za-z_.-]+)*"
ATF_CONSID_RE = re.compile(r"\bATF\s+(\d+\s+[IVX]+\s+\d+)\s+(?:consid\.|c\.)\s*(" + CONSID_LIST + ")", re.I)
BGE_CONSID_RE = re.compile(r"\bBGE\s+(\d+\s+[IVX]+\s+\d+)\s+(?:consid\.|c\.|E\.)\s*(" + CONSID_LIST + ")", re.I)
CASE_CONSID_RE = re.compile(r"\b([1-9][A-Z]_\d+/\d{4})\b(?:.{0,120}?)(?:consid\.|c\.|E\.)\s*(" + CONSID_LIST + ")", re.I | re.S)

def norm_law_parts(num, tail, code):
    code = CODE_MAP.get(code.strip(), code.strip())
    tail = tail or ""
    tail = re.sub(r"\bal\.\s*", "Abs. ", tail, flags=re.I)
    tail = re.sub(r"\balinéa\s*", "Abs. ", tail, flags=re.I)
    tail = re.sub(r"\blet\.\s*", "lit. ", tail, flags=re.I)
    tail = re.sub(r"\bch\.\s*", "Ziff. ", tail, flags=re.I)
    tail = re.sub(r"\bcifra\s*", "Ziff. ", tail, flags=re.I)
    tail = re.sub(r"\s+", " ", tail).strip(" ,;.")
    return f"Art. {num.strip()} {tail} {code}".replace("  ", " ").strip()

def law_variants(citation):
    variants = {citation}
    m = re.match(r"^(Art\.\s+\S+(?:\s+Abs\.\s+\S+)?)\s+lit\.\s+\S+\s+(\S+)$", citation)
    if m and "Abs." in m.group(1): variants.add(f"{m.group(1)} {m.group(2)}")
    return variants

def split_considerations(raw):
    return [x.strip() for x in re.split(r"\s*(?:,|et|und|and)\s*", raw or "") if x.strip()]

def extract_raw(text):
    laws = set()
    text = text or ""
    for m in LAW_RE.finditer(text):
        norm = norm_law_parts(m.group(1), m.group(2), m.group(3))
        laws.update(law_variants(norm))
        code_norm = CODE_MAP.get(m.group(3).strip(), m.group(3).strip())
        if code_norm == "StGB" and re.search(r"\b(ch\.|Ziff\.)\s*1\b", m.group(2), re.I): laws.add(f"Art. {m.group(1).strip()} Abs. 1 StGB")
        for rm in ABS_RANGE_RE.finditer(m.group(2) or ""):
            tail2 = re.sub(ABS_RANGE_RE, f"{rm.group(1)} {rm.group(3)}", m.group(2), count=1)
            laws.update(law_variants(norm_law_parts(m.group(1), tail2, m.group(3))))
    for cm in CODE_RE.finditer(text):
        code = cm.group(1)
        start = max(text.rfind(";", 0, cm.start()), text.rfind("(", 0, cm.start()))
        seg = text[start + 1 : cm.start()]
        inner_codes = list(CODE_RE.finditer(seg))
        if inner_codes: seg = seg[inner_codes[-1].end() :]
        if not re.search(r"\bart\.", seg, re.I) or len(seg) > 600: continue
        for rm in ARTICLE_ABS_RANGE_RE.finditer(seg):
            laws.update(law_variants(norm_law_parts(rm.group(1), f"{rm.group(2)} {rm.group(3)}", code)))
            laws.update(law_variants(norm_law_parts(rm.group(1), f"{rm.group(2)} {rm.group(4)}", code)))
        for am in BARE_ART_IN_SEG_RE.finditer(seg):
            norm = norm_law_parts(am.group(1), am.group(2), code)
            laws.update(law_variants(norm))
            for rm in ABS_RANGE_RE.finditer(am.group(2) or ""):
                tail2 = re.sub(ABS_RANGE_RE, f"{rm.group(1)} {rm.group(3)}", am.group(2), count=1)
                laws.update(law_variants(norm_law_parts(am.group(1), tail2, code)))

    courts = set()
    for rx in COUR_RES:
        for m in rx.finditer(text): courts.add(re.sub(r"\s+", " ", m.group(0)).strip(" .;,:"))
    for m in ATF_CONSID_RE.finditer(text):
        ref = re.sub(r"\s+", " ", m.group(1)).strip()
        for cons in split_considerations(m.group(2)): courts.add(f"BGE {ref} E. {cons}")
    for m in BGE_CONSID_RE.finditer(text):
        ref = re.sub(r"\s+", " ", m.group(1)).strip()
        for cons in split_considerations(m.group(2)): courts.add(f"BGE {ref} E. {cons}")
    for m in CASE_CONSID_RE.finditer(text):
        for cons in split_considerations(m.group(2)): courts.add(f"{m.group(1).strip()} E. {cons}")
    return laws, courts

def load_vocab(path):
    vals = set()
    try:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                c = (row.get("citation") or "").strip()
                if c: vals.add(c)
    except: pass
    return vals

def main():
    print("="*60)
    print("[INFO] Stage 05: Adaptive Threshold Submission (Alpha = 0.80)")
    print("="*60)
    
    if not INPUT_JSONL.exists(): raise FileNotFoundError(f"Missing {INPUT_JSONL}")
    
    qids = []
    with TEST_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f): qids.append(row["query_id"])
        
    law_vocab = load_vocab(LAWS_CSV)
    court_vocab = load_vocab(COURT_CSV)
    
    candidates = defaultdict(list)
    with INPUT_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                candidates[row["query_id"]].append(row)
                
    predictions = {}
    audit = []
    
    ALPHA = 0.80
    MIN_SCORE = 5.0
    FALLBACK_K = 3
    FALLBACK_VOTES = 2
    
    # Load text cache from Stage 03 to extract exact citations
    
    INPUT_TEXT_JSONL = Path('/kaggle/working/qwen_input_top20.jsonl')
    texts_by_doc = {}
    if INPUT_TEXT_JSONL.exists():
        with INPUT_TEXT_JSONL.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    texts_by_doc[r["doc_id"]] = r.get("document_text", "")
                    
    citation_cache = {}
    
    for qid in qids:
        scored = candidates.get(qid, [])
        scored.sort(key=lambda s: (s["qwen_score"], -s["rank"]), reverse=True)
        
        if not scored:
            predictions[qid] = []
            audit.append({"query_id": qid, "action": "empty"})
            continue
            
        for s in scored:
            doc_id = s["doc_id"]
            if doc_id not in citation_cache:
                txt = texts_by_doc.get(doc_id, "")
                raw_laws, raw_courts = extract_raw(txt)
                citation_cache[doc_id] = (raw_laws & law_vocab) | (raw_courts & court_vocab)
                
        top = scored[0]
        max_score = top["qwen_score"]
        adaptive_threshold = max(MIN_SCORE, max_score * ALPHA)
        
        selected = [s for s in scored if s["qwen_score"] >= adaptive_threshold]
        
        pred = set()
        if selected:
            for s in selected:
                pred.update(citation_cache.get(s["doc_id"], set()))
            action = f"adaptive_alpha_{ALPHA}_count_{len(selected)}"
        else:
            # Fallback to Top 3 Voting if everything is < 5.0
            fallback_cands = scored[:FALLBACK_K]
            counts = Counter()
            for s in fallback_cands:
                for cite in citation_cache.get(s["doc_id"], set()):
                    counts[cite] += 1
            pred = {cite for cite, n in counts.items() if n >= FALLBACK_VOTES}
            action = "low_score_vote"
            
        if not pred:
            # Absolute fallback to Top 1's citations
            for s in scored:
                cites = citation_cache.get(s["doc_id"], set())
                if cites:
                    pred = cites
                    action = "fallback_top1_any"
                    break
                    
        predictions[qid] = sorted(pred)
        audit.append({
            "query_id": qid,
            "action": action,
            "max_score": max_score,
            "adaptive_threshold": adaptive_threshold,
            "predicted_count": len(pred),
            "predicted_citations": ";".join(sorted(pred))
        })
        
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        for qid in qids:
            writer.writerow({"query_id": qid, "predicted_citations": ";".join(predictions.get(qid, []))})
            
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(json.dumps({
        "submission": str(OUTPUT_CSV),
        "queries": len(qids),
        "total_predicted_citations": sum(len(v) for v in predictions.values())
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
