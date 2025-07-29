import re
from spacy.tokens import Doc
from spacy.language import Language

class AddressDetailsParser:
    def __init__(self, nlp, name):
        self.name = name
        if not Doc.has_extension("kb_info"):
            Doc.set_extension("kb_info", default=None)

    def __call__(self, doc):
        kb_info = doc._.kb_info.copy() if doc._.kb_info else {}
        text = doc.text

        # --- care_of ---
        care_of_match = re.search(
            r"\b([CSDW]/O|Care of)\s*[:\-]?\s*([\w\s.]+?[\w]+)(?=[,\n]|$)",
            text, re.IGNORECASE | re.UNICODE
        )
        if care_of_match:
            kb_info['care_of'] = care_of_match.group(2).strip().rstrip(",.")
        else:
            fallback_match = re.search(
                r"^\s*(Mr\.?|Mrs\.?|Miss|Ms\.?|Smt\.?|Dr\.?)\s+[A-Z][\w.\s]+?(?=[,\n]|$)",
                text, re.IGNORECASE | re.UNICODE
            )
            if fallback_match:
                kb_info['care_of'] = fallback_match.group(0).strip().rstrip(",.")

        # --- house_number ---
        house_no_match = re.search(
            r"\b(?:H\.?\s*No\.?|House Number|Flat No\.?|Plot No\.?|#)?\s*[:\-]?\s*([A-Z]?\s*\d+[A-Z]?(?:[-\/]?[A-Z0-9]+)*)",
            text, re.IGNORECASE | re.UNICODE
        )
        if house_no_match:
            house_number = house_no_match.group(1).strip().replace(" ", "").rstrip(",.")
            kb_info['house_number'] = house_number

        # --- sub_locality ---
        sub_locality_match = re.search(
            r"\b(Sector[-\s]?\d+[A-Z]?|Pocket[-\s]?[A-Z]|Block[-\s]?[A-Z]|\w+\s(?:Nagar|Bazar|Bazaar))\b",
            text, re.IGNORECASE | re.UNICODE
        )
        if sub_locality_match:
            kb_info['sub_locality'] = sub_locality_match.group(0).strip().rstrip(",.")

        # --- road ---
        road_match = re.search(
            r"\b([\w\s]+?)\s+(Road|Rd\.|Street|St\.|Lane|Ln\.|Marg|Avenue|Ave\.|Bypass|Highway|Hwy\.)\b",
            text, re.IGNORECASE | re.UNICODE
        )
        if road_match:
            kb_info['road'] = road_match.group(0).strip().rstrip(",.")

        # --- poi ---
        poi_match = re.search(
            r"\b(Near|Opp\.?|Opposite|Beside|Behind|Adjacent to|In front of)\s+([^\.,\n]+)",
            text, re.IGNORECASE | re.UNICODE
        )
        if poi_match:
            kb_info['poi'] = poi_match.group(2).strip().rstrip(",.")
        else:
            # Match names like "Green Heights Apartments", "Palm Grove Villas", etc.
            named_poi_match = re.search(
                r"\b([A-Z][\w\s&'-]{2,})\s+(Apartments|Apartment|Layout|Residency|Tower|Heights|Villas|Enclave|Complex|Plaza|Mansion|Arcade|Building|Homes|Court|Garden|Society)\b",
                text, re.IGNORECASE | re.UNICODE
            )
            if named_poi_match:
                kb_info['poi'] = named_poi_match.group(0).strip().rstrip(",.")

        doc._.kb_info = kb_info
        return doc

@Language.factory("address_details_parser")
def create_address_details_parser(nlp: Language, name: str):
    return AddressDetailsParser(nlp, name)


# Optional test: run this file directly for quick check
if __name__ == "__main__":
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("address_details_parser")

    tests = [
        "C/O Mr. Sharma, A-31/F1, Pocket B, Shiv Nagar, Near Metro Station",
        "7B, Block-A, Krishna Nagar",
        "Mr. Manish Mittal, 180/2, Block-B, Green Nagar, Behind HDFC Bank",
        "House Number 60, Pocket-A, Near ABC Mall"
    ]

    for text in tests:
        doc = nlp(text)
        print(f"Input: {text}\nParsed: {doc._.kb_info}\n")
