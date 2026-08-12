"""
The combined pipeline: spaCy NER -> regex/DB enrichment (kb_info) -> regex fallback parser
-> LLM refinement -> final structured JSON. This is the full original idea, stitched together
from the `main` branch's trained spaCy model + custom enrichment pipes (copied in via
`git checkout main -- address_parser_model address_details_parser.py ...`) and the RAG
branch's regex parser + Ollama refinement step.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import spacy

from pincode_centric_parser import PincodeCentricParser  # noqa: F401
from cities_state_parser import CitiesStateParser  # noqa: F401
from locality_based_parser import LocalityBasedParser  # noqa: F401
from address_details_parser import AddressDetailsParser  # noqa: F401

from parser.parse_address import parse_address as regex_parse_address
from utils.rag_utils import get_candidates, refine_with_llm, extract_json_from_llm_output

_nlp = None


def _load_nlp():
    global _nlp
    if _nlp is None:
        # cities_state_parser's config points at "indian_cities.csv" relative to cwd.
        cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            _nlp = spacy.load(REPO_ROOT / "address_parser_model")
        finally:
            os.chdir(cwd)
    return _nlp


def preprocess_text(text: str) -> str:
    text = re.sub(r'([a-zA-Z]+)-(\d{6})', r'\1 \2', text)
    text = re.sub(r'(\w+)\s*\(\s*([^)]+?)\s*\)', r'\1 ( \2 )', text)
    return text


def run_spacy_stage(raw_address: str) -> dict:
    """Returns a dict using main branch's key names (house_number, sub_locality, care_of, ...)."""
    nlp = _load_nlp()
    text = preprocess_text(raw_address)
    doc = nlp(text)

    parsed_data = {}
    for ent in doc.ents:
        label = ent.label_.lower()
        if label not in parsed_data:
            parsed_data[label] = ent.text

    if doc._.kb_info:
        kb_data = doc._.kb_info
        for key in ['pincode', 'state', 'district', 'locality', 'city', 'care_of', 'house_number', 'road', 'poi', 'sub_locality']:
            if kb_data.get(key):
                parsed_data[key] = kb_data.get(key)

    return parsed_data


def to_rag_style(spacy_out: dict, regex_out: dict) -> dict:
    """
    Merge spaCy+enrichment output with parser.parse_address()'s regex output into the schema
    utils.rag_utils.refine_with_llm expects. spaCy+enrichment wins when both have a value,
    since it's backed by the trained NER model + knowledge-base lookups; the regex parser
    fills in whatever spaCy's stage missed.
    """
    merged = dict(regex_out)  # careof, houseno, poi, locality, sublocality, city, state, pincode
    key_map = {
        "care_of": "careof",
        "house_number": "houseno",
        "sub_locality": "sublocality",
        "poi": "poi",
        "locality": "locality",
        "city": "city",
        "state": "state",
        "pincode": "pincode",
    }
    for spacy_key, rag_key in key_map.items():
        val = spacy_out.get(spacy_key)
        if val:
            merged[rag_key] = val
    return merged


def run_rule_based_stage(raw_address: str) -> dict:
    """spaCy + enrichment + regex fallback, merged -- no LLM call."""
    spacy_out = run_spacy_stage(raw_address)
    regex_out = regex_parse_address(raw_address)
    return to_rag_style(spacy_out, regex_out)


def run_full_pipeline(raw_address: str, model: str = "mistral"):
    """
    spaCy NER -> enrichment -> regex fallback -> LLM refinement -> final JSON.
    Returns (refined_dict_or_None, rule_based_dict, raw_llm_output).
    """
    rule_based = run_rule_based_stage(raw_address)
    candidates = get_candidates(rule_based, raw_address)
    llm_output = refine_with_llm(raw_address, rule_based, candidates, model=model)
    refined = extract_json_from_llm_output(llm_output)
    return refined, rule_based, llm_output
