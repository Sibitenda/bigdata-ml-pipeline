import json
import time
import pandas as pd
import numpy as np

from kafka import KafkaProducer

from src.ingestion.bigquery_loader import (
    fetch_data
)

from src.preprocessing.preprocess import (
    preprocess_pipeline
)


print("\nStarting Kafka Producer")


# Create Kafka producer
producer = KafkaProducer(

    bootstrap_servers="localhost:9092",

    value_serializer=lambda v: json.dumps(v).encode("utf-8")

)


print("\nConnected to Kafka Broker")


# Kafka topic
TOPIC_NAME = "covid-stream"


# Number of streaming batches
NUM_BATCHES = 5

# Rows per batch
BATCH_SIZE = 100


print("\nStarting Real-Time Event Streaming")


# Simulate real-time streaming batches
for batch_num in range(1, NUM_BATCHES + 1):

    print("\n")
    print("=" * 60)
    print(f"KAFKA STREAM BATCH {batch_num}")
    print("=" * 60)

    # Fetch dataset from BigQuery
    df = fetch_data(
        limit=BATCH_SIZE
    )

    # Preprocess data
    df, _, _, _ = preprocess_pipeline(df)

    # Add streaming variation
    df["confirmed"] = (

        df["confirmed"]

        + np.random.randint(
            0,
            1000,
            len(df)
        )

    )

    df["log_confirmed"] = np.log(
        df["confirmed"] + 1
    )

    # Stream rows one-by-one
    for _, row in df.iterrows():

        event = {

            "country_region":
                str(
                    row["country_region"]
                ),

            "confirmed":
                int(
                    row["confirmed"]
                ),

            "log_confirmed":
                float(
                    row["log_confirmed"]
                ),

            "previous_day_cases":
                float(
                    row["previous_day_cases"]
                ),

            "rolling_avg_cases":
                float(
                    row["rolling_avg_cases"]
                ),

            "growth_rate":
                float(
                    row["growth_rate"]
                ),

            "day":
                int(
                    row["day"]
                ),

            "month":
                int(
                    row["month"]
                ),

            "risk_category":
                str(
                    row["risk_category"]
                )

        }

        # Send event to Kafka
        producer.send(
            TOPIC_NAME,
            value=event
        )

        print(
            f"Sent Event → "
            f"{event['country_region']} | "
            f"Cases: {event['confirmed']}"
        )

        # Simulate event delay
        time.sleep(0.2)

    print(
        f"\nCompleted Batch "
        f"{batch_num}"
    )

    print(
        "\nWaiting before next "
        "streaming batch...\n"
    )

    # Delay between batches
    time.sleep(5)


# Flush remaining events
producer.flush()

# Close producer
producer.close()


print("\nKafka Producer Pipeline Complete")