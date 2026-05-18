import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from sklearn.preprocessing import StandardScaler


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# Validate dataset structure
def validate_schema(df):

    required_columns = [
        "country_region",
        "date",
        "confirmed"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    logger.info("Schema validation passed")


# Clean raw dataset
def clean_data(df):

    logger.info("Starting data cleaning")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove missing values
    df = df.dropna()

    # Remove invalid case counts
    df = df[df["confirmed"] > 0]

    logger.info(
        f"Rows after cleaning: {len(df)}"
    )

    return df


# Generate engineered features
def engineer_features(df):

    logger.info("Generating features")

    # Convert date column
    df["date"] = pd.to_datetime(
        df["date"]
    )

    # Time-based features
    df["day"] = df["date"].dt.day

    df["month"] = df["date"].dt.month

    df["year"] = df["date"].dt.year

    df["week_day"] = (
        df["date"]
        .dt.dayofweek
    )

    # Log transformation
    df["log_confirmed"] = np.log1p(
        df["confirmed"]
    )

    # Previous day statistics
    df["previous_day_cases"] = (
        df["confirmed"]
        .shift(1)
    )

    # Rolling average
    df["rolling_avg_cases"] = (
        df["confirmed"]
        .rolling(
            window=7,
            min_periods=1
        )
        .mean()
    )

    # Growth rate
    df["growth_rate"] = (
        df["confirmed"]
        .pct_change()
        .fillna(0)
    )

    # Replace infinity values
    df["growth_rate"] = (
        df["growth_rate"]
        .replace(
            [np.inf, -np.inf],
            0
        )
    )

    logger.info(
        "Feature engineering complete"
    )

    return df


# Create multiclass risk labels
def classify_risk(cases):

    if cases <= 1000:
        return 0

    elif cases <= 10000:
        return 1

    elif cases <= 100000:
        return 2

    return 3


# Generate target labels
def create_target(df):

    logger.info(
        "Creating target labels"
    )

    # Future cases prediction target
    df["future_cases"] = (
        df["confirmed"]
        .shift(-1)
    )

    # Remove missing rows
    df = df.dropna()

    # Create multiclass target
    df["risk_level"] = (
        df["future_cases"]
        .apply(classify_risk)
    )

    # Human-readable labels
    risk_labels = {

        0: "Low",
        1: "Moderate",
        2: "High",
        3: "Critical"

    }

    df["risk_category"] = (
        df["risk_level"]
        .map(risk_labels)
    )

    logger.info(
        "Risk labels created"
    )

    return df


# Remove extreme outliers
def remove_outliers(df):

    logger.info("Removing outliers")

    q1 = df["confirmed"].quantile(0.25)

    q3 = df["confirmed"].quantile(0.75)

    iqr = q3 - q1

    lower = q1 - 1.5 * iqr

    upper = q3 + 1.5 * iqr

    before = len(df)

    df = df[
        (df["confirmed"] >= lower) &
        (df["confirmed"] <= upper)
    ]

    after = len(df)

    logger.info(
        f"Removed {before - after} outliers"
    )

    return df


# Scale numerical features
def scale_features(df):

    logger.info(
        "Scaling numerical features"
    )

    scaler = StandardScaler()

    numerical_cols = [

        "confirmed",
        "log_confirmed",
        "previous_day_cases",
        "rolling_avg_cases",
        "growth_rate",
        "day",
        "month"

    ]

    df[numerical_cols] = (
        scaler.fit_transform(
            df[numerical_cols]
        )
    )

    return df, scaler


# Generate exploratory analysis
def exploratory_analysis(df):

    logger.info(
        "Running exploratory analysis"
    )

    print("\nDataset Summary")

    print(df.describe())

    # Distribution plot
    plt.figure(figsize=(10, 5))

    plt.hist(
        df["confirmed"],
        bins=30
    )

    plt.title(
        "Confirmed Cases Distribution"
    )

    plt.xlabel("Cases")

    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        "data/processed/case_distribution.png"
    )

    plt.close()

    # Correlation matrix
    feature_cols = [

        "confirmed",
        "log_confirmed",
        "rolling_avg_cases",
        "growth_rate"

    ]

    correlation_matrix = (
        df[feature_cols]
        .corr()
    )

    print("\nCorrelation Matrix")

    print(correlation_matrix)

    return correlation_matrix


# Select final ML features
def select_features():

    selected_features = [

        "confirmed",
        "log_confirmed",
        "previous_day_cases",
        "rolling_avg_cases",
        "growth_rate",
        "day",
        "month"

    ]

    logger.info(
        f"Selected features: "
        f"{selected_features}"
    )

    return selected_features


# Save processed dataset
def save_processed_data(df):

    output_path = (
        "data/processed/"
        "covid_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        f"Processed dataset saved to "
        f"{output_path}"
    )


# Complete preprocessing workflow
def preprocess_pipeline(df):

    logger.info(
        "Starting preprocessing pipeline"
    )

    # Validate dataset
    validate_schema(df)

    # Clean raw dataset
    df = clean_data(df)

    # Generate engineered features
    df = engineer_features(df)

    # Create target labels
    df = create_target(df)

    # Remove outliers
    df = remove_outliers(df)

    # Exploratory analysis
    correlation_matrix = (
        exploratory_analysis(df)
    )

    # Select ML features
    selected_features = (
        select_features()
    )

    # Scale numerical features
    df, scaler = scale_features(df)

    # Save processed dataset
    save_processed_data(df)

    # Display outputs
    print("\nProcessed Dataset Preview")

    print(df.head())

    print("\nProcessed Dataset Shape")

    print(df.shape)

    print("\nSelected Features")

    print(selected_features)

    print("\nRisk Category Distribution")

    print(
        df["risk_category"]
        .value_counts()
    )

    print("\nCorrelation Matrix")

    print(correlation_matrix)

    logger.info(
        "Preprocessing pipeline complete"
    )

    return (

        df,
        scaler,
        correlation_matrix,
        selected_features

    )