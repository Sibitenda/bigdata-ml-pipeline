import os
from dotenv import load_dotenv
from google.cloud import bigquery

# Load .env
load_dotenv()

# Explicit credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("PROJECT_ID")

# Create client
client = bigquery.Client(project=PROJECT_ID)

query = """
SELECT country_region, date, confirmed
FROM `bigquery-public-data.covid19_jhu_csse.summary`
LIMIT 5
"""

df = client.query(query).to_dataframe()

print(df.head())