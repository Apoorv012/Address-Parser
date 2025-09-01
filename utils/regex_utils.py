import re
from typing import Optional

def extract_pincode(addr: str) -> Optional[str]:
    match = re.search(r"\b\d{6}\b", addr)
    return match.group(0) if match else None

def extract_house_number(addr: str) -> Optional[str]:
    match = re.search(r"\b\d+[a-zA-Z0-9\-\/]*\b", addr)
    return match.group(0) if match else None
