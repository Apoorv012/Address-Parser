import spacy
import random
import spacy.lang
from spacy.training.example import Example
from pathlib import Path
from pincode_centric_parser import PincodeCentricParser
from cities_state_parser import CitiesStateParser
from address_csv_converter import build_ner_training_data

# The format is: (text, {"entities": [(start_char, end_char, LABEL)]})
# You can use tools like Doccano or Prodigy to annotate more data easily.
# NER_TRAIN_DATA = [
#     ("Shop No. 5, Ground Floor, The Great India Place, Sector 38, Noida", {"entities": [(0, 25, "HOUSE_NUMBER"), (27, 46, "POI"), (48, 58, "LOCALITY")]}),
#     ("123, Sunshine Apartments, 1st Main Road, Jayanagar, Bangalore", {"entities": [(0, 3, "HOUSE_NUMBER"), (5, 25, "POI"), (27, 41, "ROAD")]}),
#     ("C-45, Connaught Place, near Odeon Cinema, New Delhi", {"entities": [(0, 4, "HOUSE_NUMBER"), (24, 41, "POI")]}),
#     ("Office 7B, 3rd Floor, Cyber Towers, Hitec City, Madhapur", {"entities": [(0, 22, "HOUSE_NUMBER"), (24, 37, "POI")]}),
#     ("Khasra No. 14/2, Village Ghitorni, Tehsil Vasant Vihar", {"entities": [(0, 15, "HOUSE_NUMBER")]}),
# ]

def build_and_train_hybrid_pipeline(pincode_dataset_path, cities_dataset_path, training_data, output_dir, iterations=30):
    """Builds a hybrid pipeline and trains the NER component."""
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir()

    nlp = spacy.blank("en")
    nlp.add_pipe("pincode_centric_parser", config={"pincode_dataset_path": pincode_dataset_path})
    nlp.add_pipe("cities_state_parser", config={"cities_dataset_path": cities_dataset_path})
    ner = nlp.add_pipe("ner")

    for _, annotations in training_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    print("Starting training of the statistical NER component...")
    with nlp.select_pipes(disable=unaffected_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(training_data)
            losses = {}
            examples = []
            for text, annotations in training_data:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            
            nlp.update(examples, drop=0.35, sgd=optimizer, losses=losses)
            print(f"Iteration {itn+1}/{iterations}, Losses: {losses}")

    nlp.to_disk(output_path)
    print(f"\nHybrid pipeline saved to '{output_path}'")


if __name__ == '__main__':
    # --- Configuration ---
    PINCODE_DATASET_FILE_PATH = 'pincode_dataset.csv'
    CITIES_DATASET_FILE_PATH = 'indian_cities.csv'
    MODEL_OUTPUT_DIR = "./address_parser_model"
    CSV_ADDRESS_PATH = "Sample_data_Interns.csv"

    NER_TRAIN_DATA = build_ner_training_data(CSV_ADDRESS_PATH)

    # # To check if the NER_TRAIN_DATA is correct
    # for text, ann in NER_TRAIN_DATA[:5]:
    #     print(text)
    #     for start, end, label in ann['entities']:
    #         print(f"{label}: '{text[start:end]}'")
    #     print()

    build_and_train_hybrid_pipeline(PINCODE_DATASET_FILE_PATH, CITIES_DATASET_FILE_PATH, NER_TRAIN_DATA, MODEL_OUTPUT_DIR)
