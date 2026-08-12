"""
Builds a fixed accuracy-eval benchmark (eval/test_set.json) from data/sample_addrs.csv.

Reassembles real structured Indian address rows into free-text raw addresses the way a
user might actually type them: some comma-delimited, some with no punctuation at all,
some missing fields, some phrased around a landmark ("near X"), some in non-standard
field order. Each entry is tagged with its `style` so downstream eval scripts can report
accuracy segmented by style, not just an overall number.

Run once (from repo root): python scripts/build_eval_set.py
"""
import csv
import json
import random
from pathlib import Path

SRC = Path("data/sample_addrs.csv")
OUT = Path("eval/test_set.json")

# Canonical field set used for ground truth / scoring across all pipelines.
FIELDS = ["house_number", "sublocality", "poi", "locality", "city", "state", "pincode"]

POI_PHRASES = ["near", "beside", "opposite", "behind", "in front of"]


def load_rows():
    with SRC.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        rows = list(reader)
    return rows


def extract_fields(row):
    return {
        "house_number": row["house number"].strip(),
        "road": row["road"].strip(),
        "sublocality": row["sublocality"].strip(),
        "locality": row["locality"].strip(),
        "poi": row["poi"].strip(),
        "city": row["city"].strip(),
        "state": row["state"].strip(),
        "pincode": row["pincode"].strip(),
    }


def usable(f):
    # Need enough to build a sensible address and a meaningful ground truth.
    return bool(f["city"] and f["state"] and f["pincode"] and f["locality"])


def expected_from(f, drop=()):
    exp = {}
    for k in FIELDS:
        if k in drop:
            exp[k] = None
        else:
            exp[k] = f[k] or None
    return exp


def make_comma(f, rng):
    parts = [f[k] for k in ["house_number", "road", "sublocality", "locality", "city", "state", "pincode"] if f[k]]
    if f["poi"]:
        parts.append(f["poi"])
    return ", ".join(parts), expected_from(f)


def make_no_comma(f, rng):
    parts = [f[k] for k in ["house_number", "road", "sublocality", "locality", "city", "state", "pincode"] if f[k]]
    if f["poi"]:
        parts.append(f["poi"])
    return " ".join(parts), expected_from(f)


def make_missing_fields(f, rng):
    # Simulate a real user who just doesn't type everything they know.
    drop = rng.sample(["house_number", "sublocality"], k=rng.choice([1, 2]))
    keep_order = ["house_number", "road", "sublocality", "locality", "city", "state", "pincode"]
    parts = [f[k] for k in keep_order if f[k] and k not in drop]
    if f["poi"] and rng.random() < 0.5:
        parts.append(f["poi"])
    else:
        drop = list(drop) + (["poi"] if f["poi"] else [])
    return ", ".join(parts), expected_from(f, drop=drop)


def make_poi_phrase(f, rng):
    if not f["poi"]:
        return None
    phrase = rng.choice(POI_PHRASES)
    parts = [f[k] for k in ["house_number", "road", "sublocality"] if f[k]]
    lead = ", ".join(parts)
    landmark = f"{phrase} {f['poi']}"
    tail = ", ".join([f[k] for k in ["locality", "city", "state", "pincode"] if f[k]])
    text = ", ".join([p for p in [lead, landmark, tail] if p])
    return text, expected_from(f)


def make_mixed_order(f, rng):
    parts = [f[k] for k in ["city", "state", "locality", "sublocality", "road", "house_number"] if f[k]]
    if f["poi"]:
        parts.append(f["poi"])
    if f["pincode"]:
        parts.append(f["pincode"])
    return ", ".join(parts), expected_from(f)


BUILDERS = {
    "comma": make_comma,
    "no_comma": make_no_comma,
    "missing_fields": make_missing_fields,
    "poi_phrase": make_poi_phrase,
    "mixed_order": make_mixed_order,
}

# How many of each style to produce.
QUOTA = {
    "comma": 8,
    "no_comma": 8,
    "missing_fields": 6,
    "poi_phrase": 5,
    "mixed_order": 5,
}


def main():
    rng = random.Random(42)
    rows = load_rows()
    rng.shuffle(rows)

    usable_rows = [extract_fields(r) for r in rows if usable(extract_fields(r))]

    test_set = []
    idx = 0
    for style, quota in QUOTA.items():
        builder = BUILDERS[style]
        made = 0
        while made < quota and idx < len(usable_rows):
            f = usable_rows[idx]
            idx += 1
            result = builder(f, rng)
            if result is None:
                continue
            text, expected = result
            test_set.append({
                "id": f"{style}_{made+1}",
                "style": style,
                "raw_address": text,
                "expected": expected,
            })
            made += 1

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(test_set)} test addresses to {OUT}")
    for style in QUOTA:
        n = sum(1 for t in test_set if t["style"] == style)
        print(f"  {style}: {n}")


if __name__ == "__main__":
    main()
