# Address Parser API using spaCy and FastAPI
This project provides a simple API to parse raw Indian addresses into structured JSON objects.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Start the FastAPI app using `uvicorn` with auto-reload:

```bash
uvicorn api:app --reload
```

- The API will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000)
- The interactive docs will be at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
