import re
from spacy.tokens import Doc
from spacy.language import Language
import pandas as pd

class LocalityBasedParser:
    def __init__(self, nlp, name, pincode_dataset_path=None, locality_db=None):
        self.name = name
        if locality_db is not None:
            print("Using already loaded pincode_db")
            self.locality_db = locality_db
        elif pincode_dataset_path is not None:
            self.locality_db = self._load_locality_database(pincode_dataset_path)
        else:
            raise ValueError("Must provide either locality_db or pincode_dataset_path")
        if not Doc.has_extension("kb_info"): Doc.set_extension("kb_info", default=None)

    def _load_locality_database(self, pincode_dataset_path):
        print(f"Loading locality knowledge base from {pincode_dataset_path}...")
        try:
            df = pd.read_csv(pincode_dataset_path, low_memory=False)
            df.dropna(subset=['pincode', 'officename', 'district', 'statename'], inplace=True)
            df['statename'] = df['statename'].str.title()
            df['district'] = df['district'].str.title()
            df['officename'] = df['officename'].str.title()
            df['officename_lower'] = df['officename'].str.strip().str.lower()
            df['pincode'] = df['pincode'].astype(str)
            locality_db = {}
            for _, row in df.iterrows():
                details = {"locality": str(row['officename']).strip(), "district": str(row['district']).strip(), "state": str(row['statename']).strip(), "pincode": str(row['pincode'])}
                locality_name = row['officename_lower']
                if locality_name not in locality_db: locality_db[locality_name] = []
                locality_db[locality_name].append(details)
            print("Locality knowledge base loaded successfully.")
            return locality_db
        except FileNotFoundError:
            print(f"Error: Pincode CSV not found at {pincode_dataset_path}. The parser will be limited.")
            return {}

    def __call__(self, doc):
        # Only run if kb_info is not already set (i.e., pincode was not found)
        if doc._.kb_info:
            return doc
        for locality_name, possible_details in self.locality_db.items():
            if not locality_name:
                continue
            for match in re.finditer(r'\b' + re.escape(locality_name) + r'\b', doc.text, re.IGNORECASE):
                if len(possible_details) == 1:
                    details = possible_details[0]
                else:
                    details = None
                    for potential_detail in possible_details:
                        district_clue = r'\b' + re.escape(potential_detail['district']) + r'\b'
                        state_clue = r'\b' + re.escape(potential_detail['state']) + r'\b'
                        if re.search(district_clue, doc.text, re.IGNORECASE) or re.search(state_clue, doc.text, re.IGNORECASE):
                            details = potential_detail
                            break
                if details:
                    doc._.kb_info = details
                    break
            if doc._.kb_info:
                break
        print("Ending LocalityBasedParser with doc._.kb_info:", doc._.kb_info)
        return doc

@Language.factory("locality_based_parser", default_config={"pincode_dataset_path": None, "locality_db": None})
def create_locality_based_parser(nlp: Language, name: str, pincode_dataset_path: str = None, locality_db=None):
    if pincode_dataset_path is None and locality_db is None:
        raise ValueError("The 'pincode_dataset_path' or 'locality_db' for the locality parser is not set in the config.")
    return LocalityBasedParser(nlp, name, pincode_dataset_path=pincode_dataset_path, locality_db=locality_db) 