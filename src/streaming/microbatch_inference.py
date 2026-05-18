import os
import time
import pandas as pd
import numpy as np

# Windows Hadoop configuration
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"

# Spark networking configuration
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python"

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
    "data/microbatch",
    exist_ok=True
)


# Create Spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("MicroBatchInferencePipeline") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()


print("\nLoading Initial Training Dataset")

# Load training dataset
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


# Store batch results
batch_results = []


print("\nStarting Micro-Batch Real-Time Inference")


# Simulate continuous real-time batches
for batch_num in range(1, 6):

    print("\n")
    print("=" * 60)
    print(f"MICRO-BATCH {batch_num}")
    print("=" * 60)

    # Simulate incoming live data
    incoming_df = fetch_data(
        limit=500
    )

    # Preprocess new batch
    incoming_df, _, _, _ = (
        preprocess_pipeline(incoming_df)
    )

    # Add synthetic streaming variation
    incoming_df["confirmed"] = (

        incoming_df["confirmed"]

        + np.random.randint(
            0,
            1000,
            len(incoming_df)
        )

    )

    incoming_df["log_confirmed"] = np.log(
        incoming_df["confirmed"] + 1
    )

    # Convert to Spark DataFrame
    batch_spark_df = spark.createDataFrame(
        incoming_df
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

    # Perform distributed inference
    predictions = fitted_model.transform(
        batch_spark_df
    )

    # Evaluate batch
    accuracy = evaluator.evaluate(
        predictions
    )

    batch_results.append({

        "batch": batch_num,
        "accuracy": round(accuracy, 4)

    })

    print(
        f"\nBatch Accuracy: "
        f"{accuracy:.4f}"
    )

    print("\nPrediction Preview")

    predictions.select(

        "risk_category",
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

    # Alert system
    if high_risk_cases > 50:

        print(
            "\nHIGH ALERT: "
            "Large number of high-risk "
            "cases detected"
        )

    elif high_risk_cases > 20:

        print(
            "\nWARNING: "
            "Elevated high-risk activity detected"
        )

    else:

        print(
            "\nNORMAL: "
            "System operating within safe range"
        )

    # Save batch predictions
    output_path = (

        f"data/microbatch/"
        f"batch_{batch_num}_predictions.csv"

    )

    predictions.select(

        "risk_category",
        "prediction"

    ).toPandas().to_csv(

        output_path,
        index=False

    )

    print(
        f"\nSaved predictions to:"
        f"\n{output_path}"
    )

    # Simulate real-time delay
    print("\nWaiting for next incoming batch...\n")

    time.sleep(5)


print("\n")
print("=" * 60)
print("MICRO-BATCH STREAMING SUMMARY")
print("=" * 60)

summary_df = pd.DataFrame(
    batch_results
)

print(summary_df)


# Save summary metrics
summary_df.to_csv(

    "data/microbatch/microbatch_metrics.csv",
    index=False

)

print(
    "\nSaved micro-batch metrics to:"
    "\ndata/microbatch/microbatch_metrics.csv"
)


print("\nMicro-Batch Streaming Pipeline Complete")


# Stop Spark session
spark.stop()