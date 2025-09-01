import pandas as pd
import re
from typing import Dict, List

# Load datasets
cities_df = pd.read_csv("data/cities.csv")
pincode_df = pd.read_csv("data/pincodes.csv", usecols=["pincode", "officename", "district", "statename"])

# --- Normalize cities dataset ---
cities_df.rename(columns={"City/Town": "city", "State/UT": "state"}, inplace=True)
cities_df["city"] = cities_df["city"].astype(str).str.lower().str.strip()
cities_df["state"] = cities_df["state"].astype(str).str.lower().str.strip()

# --- Normalize pincode dataset ---
for col in ["officename", "district", "statename", "pincode"]:
    if col in pincode_df.columns:
        pincode_df[col] = pincode_df[col].astype(str).str.lower().str.strip()

# --- Clean office names ---
def clean_officename(name: str) -> str:
    """
    Remove suffixes like S.O, SO, S O, B.O, BO, B O, H.O, HO, H O, PO, P.O., P O from officename.
    """
    return re.sub(r"\s*(s\.?\s*o|b\.?\s*o|h\.?\s*o|p\.?\s*o)$", "", name.strip(), flags=re.IGNORECASE)

pincode_df["officename"] = pincode_df["officename"].apply(clean_officename)


# --- Lookup functions ---
def lookup_pincode_info(pincode: str) -> pd.DataFrame:
    return pincode_df[pincode_df["pincode"] == str(pincode)]


def lookup_city_state(text: str) -> Dict:
    text = text.lower()
    for _, row in cities_df.iterrows():
        if row["city"] in text:
            return {"city": row["city"], "state": row["state"]}
    return {}


def get_all_localities() -> List[str]:
    return pincode_df["officename"].dropna().unique().tolist()


def get_all_states() -> List[str]:
    return pincode_df["statename"].dropna().unique().tolist()
