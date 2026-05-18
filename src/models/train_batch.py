import os
import time
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

# Regression models
from sklearn.linear_model import (
    LinearRegression
)

from sklearn.ensemble import (
    RandomForestRegressor
)

# Classification models
from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.tree import (
    DecisionTreeClassifier
)

# Regression metrics
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Classification metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.ingestion.bigquery_loader import (
    fetch_data
)

from src.preprocessing.preprocess import (
    preprocess_pipeline
)


# Create output folders
os.makedirs(
    "models",
    exist_ok=True
)

os.makedirs(
    "data/processed",
    exist_ok=True
)


print("\nLoading dataset from BigQuery")

# Load dataset
df = fetch_data(limit=5000)


print("\nRunning preprocessing pipeline")

# Run preprocessing
df, scaler, corr, selected_features = (
    preprocess_pipeline(df)
)


print("\nProcessed Dataset")
print(df.head())

print("\nDataset Shape")
print(df.shape)


# Select features
X = df[selected_features]


 
# REGRESSION PIPELINE
 

print("\n")
print("=" * 50)
print("REGRESSION MODELS")
print("=" * 50)

# Regression target
y_reg = df["future_cases"]


# Train/test split
X_train_r, X_test_r, y_train_r, y_test_r = (
    train_test_split(
        X,
        y_reg,
        test_size=0.2,
        random_state=42
    )
)


# Initialize regression models
regression_models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest Regressor":
        RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
}


# Evaluate regression models
def evaluate_regression(
    name,
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    mse = mean_squared_error(
        y_true,
        predictions
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    print(f"\n{name}")

    print("-" * 30)

    print(
        "MAE:",
        round(mae, 2)
    )

    print(
        "MSE:",
        round(mse, 2)
    )

    print(
        "R2 Score:",
        round(r2, 2)
    )


# Train regression models
for name, model in regression_models.items():

    print(f"\nTraining {name}")

    start_time = time.time()

    # Train model
    model.fit(
        X_train_r,
        y_train_r
    )

    # Predictions
    predictions = model.predict(
        X_test_r
    )

    # Evaluation
    evaluate_regression(
        name,
        y_test_r,
        predictions
    )

    # Cross validation
    cv_scores = cross_val_score(
        model,
        X,
        y_reg,
        cv=5,
        scoring="r2"
    )

    print(
        f"Cross-validation R2: "
        f"{cv_scores.mean():.4f}"
    )

    end_time = time.time()

    print(
        f"Training Time: "
        f"{end_time - start_time:.2f} seconds"
    )

    # Save model
    model_path = (
        f"models/"
        f"{name.replace(' ', '_')}_reg.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to {model_path}"
    )

    # Prediction visualization
    plt.figure(figsize=(10, 5))

    plt.plot(
        y_test_r.values[:50],
        label="Actual"
    )

    plt.plot(
        predictions[:50],
        label="Predicted"
    )

    plt.title(
        f"{name} Predictions"
    )

    plt.xlabel("Samples")
    plt.ylabel("Cases")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"data/processed/"
        f"{name.replace(' ', '_')}_predictions.png"
    )

    plt.close()


# Feature importance
forest_regressor = (
    regression_models[
        "Random Forest Regressor"
    ]
)

importance = (
    forest_regressor
    .feature_importances_
)

importance_df = pd.DataFrame({

    "Feature": selected_features,
    "Importance": importance

}).sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance_df)


# Feature importance graph
plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.title(
    "Feature Importance"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "data/processed/feature_importance.png"
)

plt.close()


 
# CLASSIFICATION PIPELINE
 

print("\n")
print("=" * 50)
print("MULTICLASS CLASSIFICATION")
print("=" * 50)

# Classification target
y_class = df["risk_level"]


# Train/test split
X_train_c, X_test_c, y_train_c, y_test_c = (
    train_test_split(
        X,
        y_class,
        test_size=0.2,
        random_state=42
    )
)


# Initialize classification models
classification_models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        )
}


classification_results = {}


# Evaluate classification models
def evaluate_classification(
    name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        predictions,
        average="weighted"
    )

    print(f"\n{name}")

    print("-" * 30)

    print(
        "Accuracy:",
        round(accuracy, 4)
    )

    print(
        "Precision:",
        round(precision, 4)
    )

    print(
        "Recall:",
        round(recall, 4)
    )

    print(
        "F1 Score:",
        round(f1, 4)
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_true,
            predictions
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_true,
            predictions
        )
    )

    return accuracy


# Train classification models
for name, model in classification_models.items():

    print(f"\nTraining {name}")

    start_time = time.time()

    # Train
    model.fit(
        X_train_c,
        y_train_c
    )

    # Predict
    predictions = model.predict(
        X_test_c
    )

    # Evaluate
    accuracy = evaluate_classification(
        name,
        y_test_c,
        predictions
    )

    classification_results[name] = (
        accuracy
    )

    # Cross validation
    cv_scores = cross_val_score(
        model,
        X,
        y_class,
        cv=5,
        scoring="accuracy"
    )

    print(
        f"Cross-validation Accuracy: "
        f"{cv_scores.mean():.4f}"
    )

    end_time = time.time()

    print(
        f"Training Time: "
        f"{end_time - start_time:.2f} seconds"
    )

    # Event-driven monitoring
    critical_cases = (
        predictions >= 3
    ).sum()

    if critical_cases > 10:

        print(
            f"ALERT: "
            f"{critical_cases} "
            f"critical predictions detected"
        )

    # Save model
    model_path = (
        f"models/"
        f"{name.replace(' ', '_')}_clf.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"Model saved to {model_path}"
    )


# Best model
best_model_name = max(
    classification_results,
    key=classification_results.get
)

best_model = classification_models[
    best_model_name
]

best_predictions = best_model.predict(
    X_test_c
)


# Confusion matrix graph
cm = confusion_matrix(
    y_test_c,
    best_predictions
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title(
    f"Confusion Matrix - "
    f"{best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "data/processed/confusion_matrix.png"
)

plt.close()


# Model comparison graph
plt.figure(figsize=(8, 5))

plt.bar(
    classification_results.keys(),
    classification_results.values()
)

plt.title(
    "Classification Model Comparison"
)

plt.ylabel("Accuracy")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "data/processed/model_comparison.png"
)

plt.close()


print("\nFinal Classification Results")

for name, score in classification_results.items():

    print(
        f"{name}: "
        f"{score:.4f}"
    )


print("\nBatch ML Pipeline Complete")