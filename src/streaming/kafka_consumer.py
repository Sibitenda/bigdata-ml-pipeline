import os
import json
import time
import pandas as pd

# Windows Hadoop configuration
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"

# Spark networking configuration
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python"

from kafka import KafkaConsumer

from pyspark.sql import SparkSession

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)

from pyspark.ml.classification import (
    RandomForestClassifier
)

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator
)

from src.ingestion.bigquery_loader import (
    fetch_data
)

from src.preprocessing.preprocess import (
    preprocess_pipeline
)


# Create output directories
os.makedirs(
    "data/kafka_predictions",
    exist_ok=True
)


print("\nStarting Kafka Consumer Pipeline")


# Create Spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("KafkaDistributedInference") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()


print("\nLoading Historical Training Dataset")


# Load historical dataset
df = fetch_data(limit=5000)


print("\nRunning Preprocessing Pipeline")


# Preprocess dataset
df, scaler, corr, selected_features = (
    preprocess_pipeline(df)
)


print("\nConverting Training Data to Spark DataFrame")


spark_df = spark.createDataFrame(df)


# Convert labels to numeric
indexer = StringIndexer(
    inputCol="risk_category",
    outputCol="label"
)

spark_df = indexer.fit(
    spark_df
).transform(spark_df)


# Assemble features
assembler = VectorAssembler(

    inputCols=selected_features,
    outputCol="features"

)

spark_df = assembler.transform(
    spark_df
)


print("\nTraining Distributed Spark Model")


# Train Spark ML model
model = RandomForestClassifier(

    labelCol="label",
    featuresCol="features",
    numTrees=20

)

fitted_model = model.fit(
    spark_df
)


print("\nDistributed Model Ready")


# Evaluation metric
evaluator = MulticlassClassificationEvaluator(

    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"

)


print("\nConnecting to Kafka Topic")


# Create Kafka consumer
consumer = KafkaConsumer(

    "covid-stream",

    bootstrap_servers="localhost:9092",

    auto_offset_reset="earliest",

    enable_auto_commit=True,

    group_id="covid-ml-group",

    value_deserializer=lambda x:
        json.loads(x.decode("utf-8"))

)


print("\nKafka Consumer Connected")


# Store incoming events
microbatch_events = []

# Store metrics
batch_metrics = []

# Micro-batch settings
MICROBATCH_SIZE = 20

batch_number = 1


print("\nStarting Real-Time Distributed Inference")


try:

    for message in consumer:

        # Read Kafka event
        event = message.value

        print("\nIncoming Kafka Event")

        print(event)

        # Add event to batch
        microbatch_events.append(event)

        # Trigger micro-batch inference
        if len(microbatch_events) >= MICROBATCH_SIZE:

            print("\n")
            print("=" * 60)
            print(f"KAFKA MICRO-BATCH {batch_number}")
            print("=" * 60)

            # Convert events → pandas DataFrame
            batch_df = pd.DataFrame(
                microbatch_events
            )

            print("\nMicro-Batch Preview")

            print(
                batch_df.head()
            )

            # Convert → Spark DataFrame
            batch_spark_df = spark.createDataFrame(
                batch_df
            )

            # Convert labels
            batch_spark_df = indexer.fit(
                batch_spark_df
            ).transform(batch_spark_df)

            # Assemble features
            batch_spark_df = assembler.transform(
                batch_spark_df
            )

            print("\nRunning Distributed Inference")

            # Run inference
            predictions = fitted_model.transform(
                batch_spark_df
            )

            # Evaluate predictions
            accuracy = evaluator.evaluate(
                predictions
            )

            batch_metrics.append({

                "batch":
                    batch_number,

                "accuracy":
                    round(accuracy, 4)

            })

            print(
                f"\nBatch Accuracy: "
                f"{accuracy:.4f}"
            )

            print("\nPrediction Results")

            predictions.select(

                "country_region",
                "prediction",
                "probability"

            ).show(10, truncate=False)

            # Event-driven monitoring
            high_risk_cases = predictions.filter(
                predictions["prediction"] == 2
            ).count()

            moderate_risk_cases = predictions.filter(
                predictions["prediction"] == 1
            ).count()

            low_risk_cases = predictions.filter(
                predictions["prediction"] == 0
            ).count()

            print("\nEvent Monitoring")

            print(
                f"Low Risk Cases: "
                f"{low_risk_cases}"
            )

            print(
                f"Moderate Risk Cases: "
                f"{moderate_risk_cases}"
            )

            print(
                f"High Risk Cases: "
                f"{high_risk_cases}"
            )

            # Alert logic
            if high_risk_cases > 10:

                print(
                    "\nHIGH ALERT: "
                    "Elevated outbreak activity detected"
                )

            elif moderate_risk_cases > 10:

                print(
                    "\nWARNING: "
                    "Moderate outbreak activity detected"
                )

            else:

                print(
                    "\nNORMAL: "
                    "System operating safely"
                )

            # Save predictions
            output_path = (

                f"data/kafka_predictions/"
                f"kafka_batch_{batch_number}.csv"

            )

            predictions.select(

                "country_region",
                "prediction"

            ).toPandas().to_csv(

                output_path,
                index=False

            )

            print(
                f"\nSaved predictions to:"
                f"\n{output_path}"
            )

            # Reset micro-batch
            microbatch_events = []

            batch_number += 1

            print(
                "\nWaiting for next "
                "Kafka events...\n"
            )

            time.sleep(2)

except KeyboardInterrupt:

    print("\nStopping Kafka Consumer")


finally:

    # Save batch metrics
    metrics_df = pd.DataFrame(
        batch_metrics
    )

    metrics_df.to_csv(

        "data/kafka_predictions/kafka_metrics.csv",
        index=False

    )

    print(
        "\nSaved Kafka metrics to:"
        "\ndata/kafka_predictions/kafka_metrics.csv"
    )

    # Close consumer
    consumer.close()

    # Stop Spark session
    spark.stop()

    print("\nKafka Consumer Pipeline Complete")