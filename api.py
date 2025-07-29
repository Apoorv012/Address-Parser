import re
import spacy
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from pathlib import Path


# Configuration using Pydantic Settings
class Settings(BaseSettings):
    """Defines application settings."""
    model_dir: Path = "./address_parser_model" # Default value if not set in .env
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create an instance of the settings
settings = Settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler() # Logs to the console
    ]
)

# --- Pydantic Models for Request and Response ---
class AddressRequest(BaseModel):
    """Request model for the raw address string."""
    raw_address: str | None = None

class ParsedAddress(BaseModel):
    """Response model for the structured address components."""
    care_of: str | None = None
    house_number: str | None = None
    poi: str | None = None
    road: str | None = None
    sublocality: str | None = None
    locality: str | None = None
    district: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None


# --- Application Setup ---
app = FastAPI(
    title="Address Parser API",
    description="An API to parse raw Indian addresses using a hybrid spaCy model.",
    version="2.0.0", # Version updated for new architecture
)


# --- Load the spaCy Model ---
nlp = None
try:
    # Need to import the custom components so spaCy knows about them when loading
    from pincode_centric_parser import PincodeCentricParser
    from cities_state_parser import CitiesStateParser
    logging.info(f"Loading model from {settings.model_dir}...")
    nlp = spacy.load(settings.model_dir)
    logging.info("Model loaded successfully.")
    
except (OSError, ImportError) as e:
    logging.error(f"Could not load model: {e}")
    logging.warning("Please ensure 'pincode_centric_parser.py' and 'cities_state_parser.py' exist and the model is trained.")
    # The app will run but the /parse endpoint will fail gracefully.


def preprocess_text(text: str) -> str:
    # Split things like Delhi-110095 into Delhi 110095
    return re.sub(r'([a-zA-Z]+)-(\d{6})', r'\1 \2', text)


# Parse Address Endpoint
@app.post("/parse", response_model=ParsedAddress, tags=["Parsing the address"])
async def parse_address(request: AddressRequest):
    """
    Parses a raw address string and returns its components, enriched with knowledge base data.
    """
    logging.info(f"Received request to parse address: '{request.raw_address}'")

    # if raw_address is not present, respond accordingly
    if not request.raw_address or not request.raw_address.strip():
        logging.warning("Validation error: 'raw_address' field is empty.")
        raise HTTPException(status_code=422, detail="The 'raw_address' field cannot be an empty string.")

    if nlp is None:
        logging.error("Attempted to use /parse endpoint but model is not loaded.")
        raise HTTPException(status_code=503, detail="Model is not loaded. Please ensure the model is trained and available.")

    text = preprocess_text(request.raw_address)

    # Process the raw address with the loaded spaCy model
    doc = nlp(text)

    # Create a dictionary to collect all entities.
    parsed_data = {}

    # 1. Get entities found by the pipeline (both rule-based and statistical)
    for ent in doc.ents:
        label = ent.label_.lower()
        if label not in parsed_data:
            parsed_data[label] = ent.text

    # 2. Enrich the results with data from the knowledge base, if available
    if doc._.kb_info:
        kb_data = doc._.kb_info
        logging.info(f"Enriching response with Knowledge Base data: {kb_data}")
        # Overwrite with Knowledge Base data if available (kb_info takes priority over NER)
        for key in ['pincode', 'state', 'district', 'locality', 'city']:
            if kb_data.get(key):
                parsed_data[key] = kb_data.get(key)

    logging.info(f"Successfully parsed address. Final Result: {parsed_data}")
    response = ParsedAddress(**parsed_data)
    return response


# Health check endpoint
@app.get("/", tags=["Health Check"])
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "model_loaded": nlp is not None}

