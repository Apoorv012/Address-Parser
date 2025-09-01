import re
from typing import List, Optional

def extract_poi(address: str, known_localities: List[str], known_states: List[str]) -> Optional[str]:
    """
    Extract POI after 'near', 'opposite', etc., stopping before locality/state/pincode.
    """
    poi_pattern = re.compile(
        r"\b(near|opp\.?|opposite|beside|behind|adjacent to|in front of)\s+(.*)",
        re.IGNORECASE
    )
    match = poi_pattern.search(address)
    if not match:
        return None

    after_keyword = match.group(2).strip().lower()
    known_localities = [l.lower() for l in known_localities]
    known_states = [s.lower() for s in known_states]

    poi_tokens = []
    words = after_keyword.split()
    for i in range(len(words)):
        remainder = " ".join(words[i:])
        if any(remainder.startswith(loc) for loc in known_localities):
            break
        if any(remainder.startswith(state) for state in known_states):
            break
        if re.match(r"^\d{6}", words[i]):
            break
        poi_tokens.append(words[i])

    return " ".join(poi_tokens).strip() if poi_tokens else None

