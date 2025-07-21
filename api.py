import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path


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
try:
    print(f"Loading model from {MODEL_DIR}...")
    nlp = spacy.load(MODEL_DIR)
    print("Model loaded successfully.")
except OSError:
    print(f"Error: Could not find model at {MODEL_DIR}.")
    print("Please run `python train_model.py` first to train and save the model.")
    nlp = None # Set to None if model loading fails


# Parse Address Endpoint
@app.post("/parse", response_model=ParsedAddress, tags=["Parsing the address"])
async def parse_address(request: AddressRequest):
    """
    Parses a raw address string and returns its components.
    """
    # if raw_address is not present, respond accordingly
    if not request.raw_address or not request.raw_address.strip():
        raise HTTPException(
            status_code=422,
            detail="The 'raw_address' field cannot be an empty string."
        )

    if nlp is None:
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

    # Create the response object using the Pydantic model
    response = ParsedAddress(**parsed_data)

    return response


# Health check endpoint
@app.get("/", tags=["Health Check"])
async def health_check():
    """A simple health check endpoint."""
    return {"status": "ok", "model_loaded": nlp is not None}

