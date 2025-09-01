from utils.regex_utils import extract_pincode, extract_house_number
from utils.lookup_utils import (
    lookup_pincode_info,
    lookup_city_state,
    get_all_localities,
    get_all_states
)
from utils.poi_utils import extract_poi
import re

def extract_sublocality(addr: str) -> str | None:
    patterns = [
        r"\b(?:[A-Za-z]+\s+)?Sector\s*\d+[A-Z]?\b",
        r"\b(?:[A-Za-z]+\s+)?Block[-\s]?[A-Z]\b(?:\s+[A-Za-z]+ Nagar)?",  # Block-B Green Nagar
        r"\b(?:[A-Za-z]+\s+)?Pocket[-\s]?[A-Z]\b",
        r"\b(?:[A-Za-z]+\s+)?Phase\s*\d+\b",
    ]
    for pat in patterns:
        match = re.search(pat, addr, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None



def parse_address(address: str) -> dict:
    result = {
        "careof": None,
        "houseno": None,
        "poi": None,
        "locality": None,
        "sublocality": None,
        "city": None,
        "state": None,
        "pincode": None,
    }

    # Step 1: Extract pincode
    pincode = extract_pincode(address)
    if pincode:
        result["pincode"] = pincode
        pin_df = lookup_pincode_info(pincode)

        if not pin_df.empty:
            result["locality"] = pin_df.iloc[0]["officename"]
            result["city"] = pin_df.iloc[0]["district"]
            result["state"] = pin_df.iloc[0]["statename"]

            # Disambiguate multiple localities
            for _, row in pin_df.iterrows():
                if row["officename"].lower() in address.lower():
                    result["locality"] = row["officename"]
                    break

    # Step 2: House number
    houseno = extract_house_number(address)
    if houseno:
        result["houseno"] = houseno

    # Step 2.5: Sublocality
    sublocality = extract_sublocality(address)
    if sublocality:
        result["sublocality"] = sublocality

    # Step 3: City/State fallback
    city_state = lookup_city_state(address)
    if city_state:
        result.update(city_state)

    # Step 4: POI (improved, not comma dependent)
    known_localities = get_all_localities()
    known_states = get_all_states()
    poi = extract_poi(address, known_localities, known_states)
    if poi:
        result["poi"] = poi

    # Step 5: Care of
    careof_match = re.search(r"\b(c/?o|care of)\s+([A-Za-z\s]+)", address, re.IGNORECASE)
    if careof_match:
        result["careof"] = careof_match.group(2).strip()

    return result
