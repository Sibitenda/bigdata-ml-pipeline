import os
import pandas as pd

# Windows Hadoop configuration
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["hadoop.home.dir"] = "C:\\hadoop"

# Spark networking configuration
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python"

from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    rand,
    expr,
    when,
    current_timestamp
)

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)

from pyspark.ml.classification import (
    RandomForestClassifier
)

from src.ingestion.bigquery_loader import (
    fetch_data
)

from src.preprocessing.preprocess import (
    preprocess_pipeline
)


# Create output directories
os.makedirs(
    "data/streaming",
    exist_ok=True
)


# Create Spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("RealTimeInferencePipeline") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config(
        "spark.hadoop.fs.file.impl",
        "org.apache.hadoop.fs.LocalFileSystem"
    ) \
    .config(
        "spark.sql.streaming.forceDeleteTempCheckpointLocation",
        "true"
    ) \
    .config(
        "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version",
        "2"
    ) \
    .getOrCreate()


print("\nLoading dataset from BigQuery")

# Load dataset
df = fetch_data(limit=5000)


print("\nRunning preprocessing pipeline")

# Run preprocessing
df, scaler, corr, selected_features = (
    preprocess_pipeline(df)
)


print("\nConverting to Spark DataFrame")

# Convert pandas DataFrame → Spark DataFrame
spark_df = spark.createDataFrame(df)


print("\nSpark Dataset Preview")

spark_df.show(5)


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

# Train distributed ML model
model = RandomForestClassifier(

    labelCol="label",
    featuresCol="features",
    numTrees=20

)

fitted_model = model.fit(
    spark_df
)


print("\nDistributed Model Ready")


# ==========================================
# REAL-TIME STREAMING PIPELINE
# ==========================================

print("\nStarting Real-Time Streaming")


# Simulated streaming source
stream_df = spark.readStream \
    .format("rate") \
    .option("rowsPerSecond", 5) \
    .load()


# Generate streaming features
stream_df = (

    stream_df

    .withColumn(
        "confirmed",
        (rand() * 100000).cast("int")
    )

    .withColumn(
        "log_confirmed",
        expr("log(confirmed + 1)")
    )

    .withColumn(
        "previous_day_cases",
        (rand() * 50000)
    )

    .withColumn(
        "rolling_avg_cases",
        (rand() * 70000)
    )

    .withColumn(
        "growth_rate",
        rand()
    )

    .withColumn(
        "day",
        (rand() * 30 + 1).cast("int")
    )

    .withColumn(
        "month",
        (rand() * 12 + 1).cast("int")
    )

)


print("\nApplying Feature Engineering to Stream")

# Apply feature vectorization
stream_features = assembler.transform(
    stream_df
)


print("\nRunning Real-Time Inference")

# Apply trained model
stream_predictions = fitted_model.transform(
    stream_features
)


# Event-driven alert system
stream_predictions = stream_predictions.withColumn(

    "alert",

    when(
        stream_predictions["prediction"] == 0,
        "LOW RISK"
    )

    .when(
        stream_predictions["prediction"] == 1,
        "MODERATE RISK"
    )

    .when(
        stream_predictions["prediction"] == 2,
        "HIGH RISK"
    )

    .otherwise(
        "CRITICAL RISK"
    )
)


# Add processing timestamp
stream_predictions = stream_predictions.withColumn(

    "processed_time",
    current_timestamp()

)


print("\nStarting Streaming Console Output")


# Console streaming output
console_query = (

    stream_predictions.select(

        "timestamp",
        "confirmed",
        "prediction",
        "alert",
        "processed_time"

    )

    .writeStream

    .outputMode("append")

    .format("console")

    .option("truncate", False)

    .option("checkpointLocation", "temp_checkpoint")

    .start()

)


print("\nStreaming Inference Running")

print(
    "\nStreaming will stop automatically "
    "after 60 seconds"
)


# Run stream for 60 seconds
console_query.awaitTermination(60)


# Stop stream
console_query.stop()


print("\nStreaming Pipeline Complete")


# Stop Spark session
spark.stop()