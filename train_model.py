import spacy
import random
import spacy.lang
from spacy.training.example import Example
from pathlib import Path
from pincode_centric_parser import PincodeCentricParser
from cities_state_parser import CitiesStateParser
from address_details_parser import AddressDetailsParser
from address_csv_converter import build_ner_training_data


def build_and_train_hybrid_pipeline(pincode_dataset_path, cities_dataset_path, training_data, output_dir, iterations=30):
    """Builds a hybrid pipeline and trains the NER component."""
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir()

    nlp = spacy.blank("en")
    nlp.add_pipe("pincode_centric_parser", config={"pincode_dataset_path": pincode_dataset_path})
    nlp.add_pipe("cities_state_parser", config={"cities_dataset_path": cities_dataset_path})
    nlp.add_pipe("address_details_parser")  # Added new parser for road, house_number, care_of, poi
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

    build_and_train_hybrid_pipeline(PINCODE_DATASET_FILE_PATH, CITIES_DATASET_FILE_PATH, NER_TRAIN_DATA, MODEL_OUTPUT_DIR)
