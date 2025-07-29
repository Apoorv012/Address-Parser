import pandas as pd

CSV_FILE = "your_address_data.csv"  # Your new structured address dataset
LABEL_MAPPING = {
    "house number": "HOUSE_NUMBER",
    "road": "ROAD",
    "sublocality": "SUBLOCALITY",
    "locality": "LOCALITY",
    "city": "CITY",
    "sub_district": "SUB_DISTRICT",
    "district": "DISTRICT",
    "state": "STATE",
    "poi": "POI",
    "pincode": "PINCODE"
}

def build_ner_training_data(csv_file):
    df = pd.read_csv(csv_file, dtype=str, sep="|").fillna("")  # Fill empty fields with empty string
    ner_data = []

    for _, row in df.iterrows():
        parts = []
        entities = []
        cursor = 0

        for col in [
            "house number", "road", "sublocality", "locality",
            "sub_district", "city", "district", "state", "poi", "pincode"
        ]:
            val = row[col].strip()
            if val:
                if parts:  # Add separator if not first
                    parts.append(", ")
                    cursor += 2

                start = cursor
                parts.append(val)
                cursor += len(val)
                end = cursor

                label = LABEL_MAPPING.get(col)
                if label:
                    entities.append((start, end, label))

        full_text = ''.join(parts)
        if full_text:
            ner_data.append((full_text, {"entities": entities}))

    return ner_data
