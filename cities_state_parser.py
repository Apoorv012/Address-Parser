import spacy
import re
import pandas as pd
from spacy.tokens import Span, Doc
from spacy.language import Language
from spacy.matcher import PhraseMatcher


class CitiesStateParser:
    def __init__(self, nlp, name, cities_dataset_path):
        self.name = name
        self.cities_db = self._load_cities_database(cities_dataset_path)
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

        if not Doc.has_extension("kb_info"): Doc.set_extension("kb_info", default=None)

        patterns = [nlp.make_doc(city) for city in self.cities_db]
        self.matcher.add("CITY", patterns)


    def _load_cities_database(self, cities_dataset_path):
        print(f"Loading knowledge base from {cities_dataset_path}...")
        try:
            df = pd.read_csv(cities_dataset_path, low_memory=False)
            df.dropna(inplace=True)
            
            # Convert relevant columns to Title Case
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

    def __call__(self, doc):
        print("Starting Cities:", doc._.kb_info)
        ents = []
        
        # Preserve existing kb_info from previous parsers (like pincode parser)
        existing_kb_info = doc._.kb_info if doc._.kb_info else {}
        # Check if we already have state information from previous parsers
        existing_state = None
        
        # Check both existing entities and kb_info
        for ent in doc.ents:
            if ent.label_ == "STATE":
                existing_state = ent.text
                break
        
        # Also check if state exists in kb_info from previous parsers
        if not existing_state and 'state' in existing_kb_info:
            existing_state = existing_kb_info['state']
            print(f"State already exists in kb_info: '{existing_state}'")
        
        # Find cities in the address
        found_city = None
        found_state = None

        print(doc.text)
        print([token.text for token in doc])

        matches = self.matcher(doc)
        print('matches:', matches)
        for match_id, start, end in matches:
            span = doc[start:end]
            city = span.text.title()
            state = self.cities_db.get(city)

            if state:
                ents.append(Span(doc, start, end, label="CITY"))
                found_city = city
                found_state = state
                print(f"Found city: '{city}' in state: '{state}'")
                break

        
        # If we found a city and don't have state information yet, add the state
        if found_city and not existing_state:
            # Check if the state is already mentioned in the text
            state_mentioned = False
            for match in re.finditer(r'\b' + re.escape(found_state) + r'\b', doc.text, re.IGNORECASE):
                span = doc.char_span(match.start(), match.end(), label="STATE")
                if span:
                    ents.append(span)
                    state_mentioned = True
                    print(f"Found state mentioned in text: '{found_state}'")
                    break
            
            # If state is not mentioned in text, we can still store it in kb_info for enrichment
            if not state_mentioned:
                print(f"State '{found_state}' not found in text, will be available in kb_info")
        
        # Merge knowledge base info
        if found_city:
            # Only add city and state if they don't already exist
            if 'city' not in existing_kb_info:
                existing_kb_info['city'] = found_city
            if 'state' not in existing_kb_info:
                existing_kb_info['state'] = found_state
            
            doc._.kb_info = existing_kb_info
            print(f"CitiesParser: Updated kb_info: {doc._.kb_info}")
        
        doc.ents = spacy.util.filter_spans(ents)
        return doc
    
@Language.factory("cities_state_parser", default_config={"cities_dataset_path": None})
def create_cities_parser(nlp: Language, name: str, cities_dataset_path: str):
    """
    This factory function tells spaCy how to build the CitiesStateParser component.
    It takes the 'cities_dataset_path' from the config and passes it to the class.
    """
    if cities_dataset_path is None:
        raise ValueError("The 'cities_dataset_path' for the pincode parser is not set in the config.")

    return CitiesStateParser(nlp, name, cities_dataset_path)