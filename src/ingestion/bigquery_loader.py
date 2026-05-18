import os
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = \
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = os.getenv("PROJECT_ID")

client = bigquery.Client(project=PROJECT_ID)

def fetch_data(limit=2000):

    query = f"""
    SELECT country_region, date, confirmed
    FROM `bigquery-public-data.covid19_jhu_csse.summary`
    WHERE date >= '2021-01-01'
    LIMIT {limit}
    """

    df = client.query(query).to_dataframe()

    return df