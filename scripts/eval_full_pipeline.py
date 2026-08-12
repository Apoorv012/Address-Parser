"""
Scores the combined pipeline (spaCy NER + enrichment + regex fallback [+ optional LLM
refinement]) against eval/test_set.json. This is the "everything together" version of the
idea: pipeline/full_pipeline.py.

Usage:
    python scripts/eval_full_pipeline.py --no-llm      # spaCy+enrichment+regex only
    python scripts/eval_full_pipeline.py                # full stage, needs Ollama + a pulled model
    python scripts/eval_full_pipeline.py --model mistral

Writes eval/results_full.json (or eval/results_full_no_llm.json with --no-llm).
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.full_pipeline import run_rule_based_stage, run_full_pipeline

TEST_SET = REPO_ROOT / "eval" / "test_set.json"
FIELDS = ["house_number", "sublocality", "poi", "locality", "city", "state", "pincode"]

KEY_MAP = {
    "houseno": "house_number",
    "sublocality": "sublocality",
    "poi": "poi",
    "locality": "locality",
    "city": "city",
    "state": "state",
    "pincode": "pincode",
}


def normalize(v):
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null"):
        return None
    return re.sub(r"\s+", " ", s)


def fields_match(pred, expected) -> bool:
    p, e = normalize(pred), normalize(expected)
    if e is None:
        return p is None
    if p is None:
        return False
    return p == e or p in e or e in p


def to_canonical(raw_dict: dict) -> dict:
    out = {f: None for f in FIELDS}
    for k, v in (raw_dict or {}).items():
        canon = KEY_MAP.get(k)
        if canon:
            out[canon] = v
    return out


def predict_no_llm(raw_address):
    rule_based = run_rule_based_stage(raw_address)
    return to_canonical(rule_based), {}


def predict_full(raw_address, model):
    refined, rule_based, llm_output = run_full_pipeline(raw_address, model=model)
    if refined is None:
        return to_canonical(rule_based), {"parse_failed": True}
    return to_canonical(refined), {"parse_failed": False}


def score(test_set, predict_fn):
    per_field_correct = {f: 0 for f in FIELDS}
    per_field_total = {f: 0 for f in FIELDS}
    per_style = {}
    exact_matches = 0
    total_fields_correct = 0
    total_fields_expected = 0
    details = []

    for i, item in enumerate(test_set):
        expected = item["expected"]
        print(f"[{i+1}/{len(test_set)}] {item['id']} ({item['style']})", file=sys.stderr)
        try:
            predicted, meta = predict_fn(item["raw_address"])
        except Exception as e:
            print(f"  [ERROR] {item['id']}: {e!r} -- scoring as all-null and continuing", file=sys.stderr)
            predicted, meta = {f: None for f in FIELDS}, {"error": repr(e)}

        style = item["style"]
        per_style.setdefault(style, {"correct": 0, "total": 0, "exact": 0, "n": 0})
        per_style[style]["n"] += 1

        row_correct = 0
        row_total = 0
        all_ok = True
        for f in FIELDS:
            exp_v = expected.get(f)
            pred_v = predicted.get(f)
            if exp_v is not None:
                row_total += 1
                per_field_total[f] += 1
                ok = fields_match(pred_v, exp_v)
                if ok:
                    row_correct += 1
                    per_field_correct[f] += 1
                else:
                    all_ok = False
            else:
                if normalize(pred_v) is not None:
                    all_ok = False

        total_fields_correct += row_correct
        total_fields_expected += row_total
        per_style[style]["correct"] += row_correct
        per_style[style]["total"] += row_total
        if all_ok:
            exact_matches += 1
            per_style[style]["exact"] += 1

        details.append({
            "id": item["id"],
            "style": style,
            "raw_address": item["raw_address"],
            "expected": expected,
            "predicted": predicted,
            "field_accuracy": round(row_correct / row_total, 3) if row_total else None,
            "exact_match": all_ok,
            "meta": meta,
        })

    n = len(test_set)
    summary = {
        "n_addresses": n,
        "field_level_accuracy": round(total_fields_correct / total_fields_expected, 4) if total_fields_expected else None,
        "exact_match_accuracy": round(exact_matches / n, 4) if n else None,
        "per_field_accuracy": {
            f: (round(per_field_correct[f] / per_field_total[f], 4) if per_field_total[f] else None)
            for f in FIELDS
        },
        "per_style_accuracy": {
            s: {
                "field_level": round(v["correct"] / v["total"], 4) if v["total"] else None,
                "exact_match": round(v["exact"] / v["n"], 4) if v["n"] else None,
                "n": v["n"],
            }
            for s, v in per_style.items()
        },
    }
    return summary, details


def print_report(title, summary):
    print(f"\n=== {title} ===")
    print(f"Addresses evaluated: {summary['n_addresses']}")
    print(f"Field-level accuracy: {summary['field_level_accuracy']:.1%}")
    print(f"Exact-match accuracy: {summary['exact_match_accuracy']:.1%}")
    print("Per-field accuracy:")
    for f, acc in summary["per_field_accuracy"].items():
        print(f"  {f:15s} {acc:.1%}" if acc is not None else f"  {f:15s} n/a")
    print("Per-style accuracy (field-level):")
    for s, v in summary["per_style_accuracy"].items():
        print(f"  {s:16s} field={v['field_level']:.1%}  exact={v['exact_match']:.1%}  n={v['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Score spaCy+enrichment+regex only, no Ollama needed")
    ap.add_argument("--model", default="mistral")
    args = ap.parse_args()

    test_set = json.loads(TEST_SET.read_text(encoding="utf-8"))

    if args.no_llm:
        summary, details = score(test_set, predict_no_llm)
        out_path = REPO_ROOT / "eval" / "results_full_no_llm.json"
        title = "Combined pipeline: spaCy + enrichment + regex (no LLM)"
    else:
        summary, details = score(test_set, lambda addr: predict_full(addr, args.model))
        out_path = REPO_ROOT / "eval" / "results_full.json"
        title = f"Combined pipeline: spaCy + enrichment + regex + LLM refinement (model={args.model})"

    print_report(title, summary)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
