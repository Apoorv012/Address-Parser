from parser.parse_address import parse_address
from utils.rag_utils import get_candidates, refine_with_llm
import json

def pprint_dict(d: dict):
    print("{\n" + ",\n".join([f"\t\"{k}\" : \"{v}\"" for k, v in d.items()]) + "\n}")

if __name__ == "__main__":
    addr = "7-B, Pocket C, near astha child clinic Sector 24 Rohini, Delhi 110085"
    print("Address: ", addr)

    # Stage 1: Rule-based parser
    parsed = parse_address(addr)
    print("Rule-based result:")
    pprint_dict(parsed)

    # Stage 2: Candidate building
    candidates = get_candidates(parsed, addr)
    print("Candidates:")
    pprint_dict(candidates)

    # Stage 3: LLM refinement (local via Ollama)
    refined_output = refine_with_llm(addr, parsed, candidates, model="phi3")
    
    print("\nCorrected by LLM:")
    try:
        refined_json = json.loads(refined_output)
        pprint_dict(refined_json)
    except:
        print(refined_output)
