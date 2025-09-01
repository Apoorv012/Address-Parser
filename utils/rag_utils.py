import subprocess
import json
from utils.lookup_utils import lookup_pincode_info, lookup_city_state

def get_candidates(parsed_json, address):
    candidates = {
        "localities": [],
        "city": [],
        "state": [],
        "pincode": []
    }

    # 1. Localities from pincode DB
    if parsed_json.get("pincode"):
        pin_df = lookup_pincode_info(parsed_json["pincode"])
        if not pin_df.empty:
            candidates["localities"] = list(set(pin_df["officename"].str.lower()))

    # 2. City/State from DB
    city_state = lookup_city_state(address)
    if city_state:
        if city_state.get("city"):
            candidates["city"].append(city_state["city"].lower())
        if city_state.get("state"):
            candidates["state"].append(city_state["state"].lower())

    # 3. Always include parsed values
    if parsed_json.get("locality"):
        candidates["localities"].append(parsed_json["locality"].lower())
    if parsed_json.get("city"):
        candidates["city"].append(parsed_json["city"].lower())
    if parsed_json.get("state"):
        candidates["state"].append(parsed_json["state"].lower())
    if parsed_json.get("pincode"):
        candidates["pincode"].append(parsed_json["pincode"])

    # Deduplicate
    for k in candidates:
        candidates[k] = list(set([c for c in candidates[k] if c]))

    return candidates


def refine_with_llm(raw_address, parsed_json, candidates, model="phi3"):
    prompt = f"""
You are correcting a parsed Indian address.

STRICT RULES:
- Output must be valid JSON
- Use the exact schema below
- Only fill/correct values using the raw address and candidates
- Do not invent pincodes, only pick from candidates

Schema:
{{
  "careof": string or null,
  "houseno": string or null,
  "sublocality": string or null,
  "poi": string or null,
  "locality": string or null,
  "city": string or null,
  "state": string or null,
  "pincode": string or null
}}

Raw Address:
{raw_address}

Parsed JSON (first pass):
{json.dumps(parsed_json, indent=2)}

Candidates:
{json.dumps(candidates, indent=2)}

Return only the corrected JSON object.
"""

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    return result.stdout.decode("utf-8")
