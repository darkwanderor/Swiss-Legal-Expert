#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def split_cites(s):
    return {x.strip() for x in str(s or "").split(";") if x.strip()}


def metrics(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


def find_dataset(file_or_dir_name):
    import os
    base = "/kaggle/input"
    if not os.path.exists(base): return Path(f"./{file_or_dir_name}")
    for root, dirs, files in os.walk(base):
        if file_or_dir_name in files or file_or_dir_name in dirs:
            return Path(root) / file_or_dir_name
    raise FileNotFoundError(f"Could not find '{file_or_dir_name}' in {base}")

def main():
    print("="*60)
    print("[INFO] Stage 06: Evaluation (Metrics)")
    print("[INFO] Target: Strict Zero-Leakage Validation Set")
    print("="*60)
    
    try:
        GOLD_CSV = find_dataset('test.csv')
    except FileNotFoundError:
        try:
            GOLD_CSV = find_dataset('train.csv')
        except:
            GOLD_CSV = Path("test.csv")
            
    SUBMISSION = Path('/kaggle/working/submission.csv')
    OUT_JSON = Path('/kaggle/working/metrics.json')
    OUT_CSV = Path('/kaggle/working/metrics.csv')
    
    if not SUBMISSION.exists():
        print(f"[WARN] {SUBMISSION} not found. Skipping evaluation.")
        return

    gold = {}
    with GOLD_CSV.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gold[row["query_id"]] = split_cites(row.get("gold_citations", ""))
            
    pred = {}
    with SUBMISSION.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pred[row["query_id"]] = split_cites(row.get("predicted_citations", ""))

    rows = []
    for qid in sorted(gold):
        m = metrics(pred.get(qid, set()), gold[qid])
        rows.append({
            "query_id": qid,
            **m,
            "pred_count": len(pred.get(qid, set())),
            "gold_count": len(gold[qid]),
            "matched": sorted(pred.get(qid, set()) & gold[qid]),
            "missing": sorted(gold[qid] - pred.get(qid, set())),
            "extra": sorted(pred.get(qid, set()) - gold[qid]),
        })
    summary = {
        "queries": len(rows),
        "macro_f1": sum(r["f1"] for r in rows) / len(rows) if rows else 0,
        "macro_precision": sum(r["precision"] for r in rows) / len(rows) if rows else 0,
        "macro_recall": sum(r["recall"] for r in rows) / len(rows) if rows else 0,
        "tp": sum(r["tp"] for r in rows),
        "fp": sum(r["fp"] for r in rows),
        "fn": sum(r["fn"] for r in rows),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "f1", "precision", "recall", "tp", "fp", "fn", "pred_count", "gold_count"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
