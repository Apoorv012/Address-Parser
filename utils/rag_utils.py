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

    if parsed_json.get("pincode"):
        pin_df = lookup_pincode_info(parsed_json["pincode"])
        if not pin_df.empty:
            candidates["localities"] = list(set(pin_df["officename"].str.lower()))

    city_state = lookup_city_state(address)
    if city_state:
        if city_state.get("city"):
            candidates["city"].append(city_state["city"].lower())
        if city_state.get("state"):
            candidates["state"].append(city_state["state"].lower())

    if parsed_json.get("locality"):
        candidates["localities"].append(parsed_json["locality"].lower())

    for k in candidates:
        candidates[k] = list(set([c for c in candidates[k] if c]))

    return candidates


def refine_with_llm(raw_address, parsed_json, candidates, model="mistral"):
    # Lock stable fields
    locked = {
        "city": parsed_json.get("city"),
        "state": parsed_json.get("state"),
        "pincode": parsed_json.get("pincode")
    }

    prompt = f"""
You are parsing Indian address.

Rules:
- Output must be valid JSON
- Use the exact schema shown
- If unsure, leave a field null
- Rule based json might be wrong, so correct it according to you
- If you feel that the address is missing something, check the candidates
- Do what you feel is best

Schema:
{{
  "careof": string or null,
  "houseno": string or null,
  "sublocality": string or null,
  "poi": string or null,
  "locality": string or null,
  "city": string,
  "state": string,
  "pincode": string
}}

Raw Address:
{raw_address}

Rule-based result:
{parsed_json}

Candidates:
{candidates}

Reply ONLY with the corrected JSON.
"""

    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )

    return result.stdout.decode("utf-8")
