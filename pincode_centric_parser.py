import spacy
import re
import pandas as pd
from spacy.tokens import Span, Doc
from spacy.language import Language


def clean_office_name(name):
    """Removes common post office suffixes (like S.O, H.O, B.O, SO, etc.) from the end of the string."""
    # This regex looks for S, H, or B, followed by optional dots and spaces, ending with O,
    # and ensures this pattern is at the very end of the string ($).
    name = re.sub(r'\s*(?:[SHB]\.?\s?O\.?)$', '', name, flags=re.IGNORECASE)
    return name.strip()


class PincodeCentricParser:
    def __init__(self, nlp, name, pincode_dataset_path, cities_dataset_path):
        self.name = name
        self.pincode_db, self.locality_db = self._load_pincode_database(pincode_dataset_path)
        self.cities_db = self._load_cities_database(cities_dataset_path)
        if not Doc.has_extension("kb_info"): Doc.set_extension("kb_info", default=None)

    def _load_pincode_database(self, pincode_dataset_path):
        print(f"Loading knowledge base from {pincode_dataset_path}...")
        try:
            df = pd.read_csv(pincode_dataset_path, low_memory=False)
            df.dropna(subset=['pincode', 'officename', 'district', 'statename'], inplace=True)
            
            # NEW: Convert relevant columns to Title Case
            df['statename'] = df['statename'].str.title()
            df['district'] = df['district'].str.title()
            df['officename'] = df['officename'].str.title()

            # Clean the officename column to remove suffixes before any processing
            df['officename'] = df['officename'].apply(clean_office_name)

            df['officename_lower'] = df['officename'].str.strip().str.lower()
            df['pincode'] = df['pincode'].astype(str)
            
            pincode_db, locality_db = {}, {}
            for pincode, group in df.groupby('pincode'):
                pincode_db[pincode] = {
                    "state": group['statename'].iloc[0],
                    "district": group['district'].iloc[0],
                    "localities": group['officename'].str.strip().unique().tolist()
                }
            for _, row in df.iterrows():
                details = {"locality": str(row['officename']).strip(), "district": str(row['district']).strip(), "state": str(row['statename']).strip(), "pincode": str(row['pincode'])}
                locality_name = row['officename_lower']
                if locality_name not in locality_db: locality_db[locality_name] = []
                locality_db[locality_name].append(details)
            print("Knowledge base loaded successfully.")
            return pincode_db, locality_db
        except FileNotFoundError:
            print(f"Error: Pincode CSV not found at {pincode_dataset_path}. The parser will be limited.")
            return {}, {}

    def _load_cities_database(self, cities_dataset_path):
        print(f"Loading knowledge base from {cities_dataset_path}...")
        try:
            df = pd.read_csv(cities_dataset_path, low_memory=False)
            df.dropna(inplace=True)
            
            # NEW: Convert relevant columns to Title Case
            df['State/UT'] = df['State/UT'].str.title()
            df['City/Town'] = df['City/Town'].str.title()

            cities_db = {}
            for _, row in df.iterrows():
                state = row['State/UT']
                city = row['City/Town']
                cities_db[city] = state
            print("Knowledge base loaded successfully.")
            return cities_db
        except FileNotFoundError:
            print(f"Error: Cities CSV not found at {cities_dataset_path}. The parser will be limited.")
            return {}

    def _find_pincode(self, doc):
        match = re.search(r'\b(\d{6})\b', doc.text)
        if match:
            print(f"Pincode found: {match.group(1)}")
            span = doc.char_span(match.start(1), match.end(1), label="PINCODE")
            return match.group(1), span
        return None, None

    def __call__(self, doc):
        pincode, pincode_span = self._find_pincode(doc)
        ents = []
        doc._.kb_info = None
        if pincode and pincode in self.pincode_db:
            known_details = self.pincode_db[pincode]
            doc._.kb_info = {**known_details, "pincode": pincode}
            if pincode_span:
                ents.append(pincode_span)
            for label, text_to_find in [("STATE", known_details['state']), ("DISTRICT", known_details['district']), ("CITY", known_details['district'])]:
                for match in re.finditer(r'\b' + re.escape(text_to_find) + r'\b', doc.text, re.IGNORECASE):
                    span = doc.char_span(match.start(), match.end(), label=label)
                    if span:
                        ents.append(span)

            for locality in known_details['localities']:
                for match in re.finditer(r'\b' + re.escape(locality) + r'\b', doc.text, re.IGNORECASE):
                    span = doc.char_span(match.start(), match.end(), label="LOCALITY")
                    if span:
                        ents.append(span)

        else:
            # REVERSE LOOKUP LOGIC
            print("No pincode found. Attempting reverse lookup by locality.")
            for locality_name, possible_details in self.locality_db.items():
                if not locality_name: continue # Skip empty locality names
                for match in re.finditer(r'\b' + re.escape(locality_name) + r'\b', doc.text, re.IGNORECASE):
                    
                    if len(possible_details) == 1:
                        # Unambiguous match, we can be confident.
                        details = possible_details[0]
                    else:
                        # Ambiguous match, look for more clues (district or state).
                        print(f"Ambiguous locality '{locality_name}' found. Searching for clues...")
                        details = None
                        for potential_detail in possible_details:
                            # Check if the district or state for this potential match is in the text
                            district_clue = r'\b' + re.escape(potential_detail['district']) + r'\b'
                            state_clue = r'\b' + re.escape(potential_detail['state']) + r'\b'
                            if re.search(district_clue, doc.text, re.IGNORECASE) or re.search(state_clue, doc.text, re.IGNORECASE):
                                print(f"Disambiguated with clue: {potential_detail['district']}/{potential_detail['state']}")
                                details = potential_detail
                                break # Found the correct detail
                    
                    if details:
                        print(f"Found known locality: '{locality_name}'. Filling details.")
                        doc._.kb_info = details
                        span = doc.char_span(match.start(), match.end(), label="LOCALITY")
                        if span is not None: ents.append(span)
                        break # Found a match, stop searching this locality
                if doc._.kb_info: break # Found a definitive match, stop all searching
        
        doc.ents = spacy.util.filter_spans(ents)
        return doc
    
@Language.factory("pincode_centric_parser", default_config={"pincode_dataset_path": None, "cities_dataset_path": None})
def create_pincode_parser(nlp: Language, name: str, pincode_dataset_path: str, cities_dataset_path: str):
    """
    This factory function tells spaCy how to build the PincodeCentricParser component.
    It takes the 'pincode_dataset_path' and 'cities_dataset_path' from the config and passes it to the class.
    """
    if pincode_dataset_path is None:
        raise ValueError("The 'pincode_dataset_path' for the pincode parser is not set in the config.")
    if cities_dataset_path is None:
        raise ValueError("The 'cities_dataset_path' for the pincode parser is not set in the config.")

    return PincodeCentricParser(nlp, name, pincode_dataset_path, cities_dataset_path)