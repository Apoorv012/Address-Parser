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

# --- 1. Pydantic Models for Request and Response ---

class AddressRequest(BaseModel):
    """Request model for the raw address string."""
    raw_address: str | None = None

class ParsedAddress(BaseModel):
    """Response model for the structured address components."""
    house_number: str | None = None
    poi: str | None = None
    road: str | None = None
    sublocality: str | None = None
    locality: str | None = None
    sub_district: str | None = None
    district: str | None = None
    city: str | None = None
    state: str | None = None
    pincode: str | None = None

# --- 2. Application Setup ---

# Create the FastAPI app instance
app = FastAPI(
    title="Address Parser API",
    description="An API to parse raw Indian addresses into structured components using a custom spaCy NER model.",
    version="1.0.0",
)


# Load the spaCy Model ---

# Define the path to the saved model
MODEL_DIR = Path("./address_parser_model")

# Load the trained spaCy model from disk
nlp = None
try:
    logging.info(f"Loading model from {settings.model_dir}...")
    nlp = spacy.load(settings.model_dir)
    logging.info("Model loaded successfully.")
except OSError:
    logging.error(f"Could not find model at {settings.model_dir}.")
    logging.warning("Please train the model, and save the model first.")
    # The app will run but the /parse endpoint will fail gracefully.


# Parse Address Endpoint
@app.post("/parse", response_model=ParsedAddress, tags=["Parsing the address"])
async def parse_address(request: AddressRequest):
    """
    Parses a raw address string and returns its components.
    """
    logging.info(f"Received request to parse address: '{request.raw_address}'")

    # if raw_address is not present, respond accordingly
    if not request.raw_address or not request.raw_address.strip():
        logging.warning("Validation error: 'raw_address' field is empty.")
        raise HTTPException(
            status_code=422,
            detail="The 'raw_address' field cannot be an empty string."
        )

    if nlp is None:
        logging.error("Attempted to use /parse endpoint but model is not loaded.")
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. Please ensure the model is trained and available."
        )

    # Process the raw address with the loaded spaCy model
    doc = nlp(request.raw_address)

    # Use a dictionary to collect entities, handling multiple matches by taking the first one
    parsed_data = {}
    for ent in doc.ents:
        label = ent.label_.lower()
        if label not in parsed_data: # Only take the first entity for a given label
            parsed_data[label] = ent.text

    logging.info(f"Successfully parsed address. Result: {parsed_data}")

    # Create the response object using the Pydantic model
    response = ParsedAddress(**parsed_data)

    return response


# Health check endpoint
@app.get("/", tags=["Health Check"])
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "model_loaded": nlp is not None}

