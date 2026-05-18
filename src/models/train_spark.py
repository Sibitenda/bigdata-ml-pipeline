import os
import time
import pandas as pd

os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python"

from pyspark.sql import SparkSession

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler
)

from pyspark.ml.classification import (
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier
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
    "models/spark",
    exist_ok=True
)

os.makedirs(
    "data/processed",
    exist_ok=True
)


# Create Spark session
spark = SparkSession.builder \
    .master("local[2]") \
    .appName("DistributedMLPipeline") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()


print("\nLoading dataset from BigQuery")

# Load dataset
df = fetch_data(limit=10000)


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


print("\nSpark Dataset Schema")

spark_df.printSchema()


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


print("\nFeature Vector Preview")

spark_df.select(
    "features",
    "label"
).show(5, truncate=False)


# Train/test split
train_df, test_df = (
    spark_df.randomSplit(
        [0.8, 0.2],
        seed=42
    )
)

print("\nTraining Rows")

print(train_df.count())

print("\nTesting Rows")

print(test_df.count())


# Initialize distributed ML models
models = {

    "Random Forest":
        RandomForestClassifier(
            labelCol="label",
            featuresCol="features",
            numTrees=20
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            labelCol="label",
            featuresCol="features"
        ),

    "Logistic Regression":
        LogisticRegression(
            labelCol="label",
            featuresCol="features",
            maxIter=100
        )
}


# Evaluation metrics
accuracy_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
)

f1_evaluator = (
    MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    )
)


# Store results
results = {}

results_list = []

all_predictions = []


print("\nStarting Distributed Training")


# Train and evaluate models
for name, model in models.items():

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

    start_time = time.time()

    # Train distributed model
    fitted_model = model.fit(
        train_df
    )

    # Generate predictions
    predictions = fitted_model.transform(
        test_df
    )

    # Evaluate accuracy
    accuracy = accuracy_evaluator.evaluate(
        predictions
    )

    # Evaluate F1 score
    f1_score = f1_evaluator.evaluate(
        predictions
    )

    # Store results in dictionary
    results[name] = {

        "accuracy": accuracy,
        "f1_score": f1_score

    }

    # Store results in list
    results_list.append({

        "model": name,
        "accuracy": round(accuracy, 4),
        "f1_score": round(f1_score, 4)

    })

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"F1 Score: {f1_score:.4f}"
    )

    # Display predictions
    print("\nPrediction Preview")

    predictions.select(
        "risk_category",
        "prediction",
        "probability"
    ).show(5, truncate=False)

    # Save sample predictions
    sample_preds = predictions.select(
        "risk_category",
        "prediction"
    ).limit(20).toPandas()

    sample_preds["model"] = name

    all_predictions.append(
        sample_preds
    )

    # Event-driven monitoring
    critical_cases = predictions.filter(
        predictions["prediction"] == 0
    ).count()

    if critical_cases > 10:

        print(
            f"\nALERT: "
            f"{critical_cases} "
            f"critical cases detected"
        )

    # Skip Spark model persistence
    print(
        "\nSkipping Spark model persistence "
        "on Windows environment"
    )

    end_time = time.time()

    print(
        f"\nTraining Time: "
        f"{end_time - start_time:.2f} seconds"
    )


print("\n")
print("=" * 50)
print("FINAL MODEL COMPARISON")
print("=" * 50)


for name, metrics in results.items():

    print(
        f"{name} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"F1 Score: {metrics['f1_score']:.4f}"
    )


# Save metrics
metrics_df = pd.DataFrame(
    results_list
)

metrics_path = (
    "data/processed/"
    "spark_model_results.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print(
    f"\nSaved metrics to "
    f"{metrics_path}"
)


# Save predictions
predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_path = (
    "data/processed/"
    "spark_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)

print(
    f"Saved predictions to "
    f"{predictions_path}"
)


# Display summary table
print("\nSpark Model Evaluation Summary")

print(metrics_df)


print("\nDistributed Spark ML Pipeline Complete")


# Stop Spark session
spark.stop()