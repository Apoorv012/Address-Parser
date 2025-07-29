import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Fetch page
url = "https://www.britannica.com/topic/list-of-cities-and-towns-in-India-2033033"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Sections
sections = soup.find_all('section', id=re.compile("^ref328"))
# print("sections:", sections)

city_state_pairs = []

# Extract data from each section
for section in sections:
    # State name is inside <h3>
    state_heading = section.find('h2')
    if not state_heading:
        continue
    state_name = state_heading.get_text(strip=True)
    # Remove any text in parentheses from the state name
    state_name = re.sub(r'\s*\([^)]*\)', '', state_name)

    # Cities/towns are listed in <ul><li>
    ul = section.find('ul')
    if ul:
        for li in ul.find_all('li'):
            city = li.get_text(strip=True)
            city_state_pairs.append({'State/UT': state_name, 'City/Town': city})
    # break

# Convert to DataFrame
df = pd.DataFrame(city_state_pairs)

# Save to CSV
df.to_csv("indian_cities.csv", index=False)
print("Saved indian_cities.csv with", len(df), "records.")

# I have manually updated Dadra and Nagar Haveli and Daman and Diu fields
# I have manually added Gautam Buddha Nagar and Noida
# I have manually added Bangalore too