# Address Parser API using spaCy and FastAPI

This project provides a simple API to parse raw Indian addresses into structured JSON objects. It uses `pydantic-settings` for configuration management and includes request logging.

## Project Structure

```
.
├── address_parser_model/   <-- This is the spacy model
├── .env                    <-- Your environment configuration file
├── api.py                  <-- The FastAPI application
└── requirements.txt        <-- Python dependencies
```

## Setup Instructions


### 1. Set up a Virtual Environment (Recommended)

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

Install all the required libraries from your updated `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## How to Use

### Step 1: Run the FastAPI Server

Start the API server using `uvicorn`.

```bash
uvicorn api:app --reload
```

When you run the server, you will now see log messages in your console, including the model loading status and details for each request you make.

**Example Log Output:**

```
INFO:     Uvicorn running on [http://127.0.0.1:8000](http://127.0.0.1:8000) (Press CTRL+C to quit)
INFO:     Started reloader process [28789] using StatReload
INFO:     Started server process [28791]
INFO:     Waiting for application startup.
2023-10-27 12:30:00,123 - INFO - Loading model from address_parser_model...
2023-10-27 12:30:01,456 - INFO - Model loaded successfully.
INFO:     Application startup complete.
```

### Step 3: Test the API

Go to the interactive documentation at **`http://127.0.0.1:8000/docs`** to test the API. When you send a request, check your terminal to see the new log messages for that request.
