# Big Data MLOps Pipeline Documentation

## 1. Project Overview

This project implements an end-to-end Big Data MLOps Pipeline using:

* Python
* Apache Spark
* Kafka
* FastAPI
* Streamlit
* Docker
* Kubernetes
* BigQuery

The pipeline supports:

| Component              | Purpose                      |
| ---------------------- | ---------------------------- |
| Batch Processing       | Traditional ML training      |
| Distributed Processing | Spark-based large-scale ML   |
| Micro-Batch Streaming  | Real-time Spark inference    |
| Kafka Streaming        | Event-driven data pipelines  |
| API Serving            | Prediction endpoints         |
| Dashboard Monitoring   | Visualization and monitoring |
| Containerization       | Docker deployment            |
| Orchestration          | Kubernetes deployment        |

---

# 2. Project Architecture

```text
BigQuery Dataset
        ↓
Data Ingestion
        ↓
Preprocessing Pipeline
        ↓
Batch ML Training
        ↓
Spark Distributed Training
        ↓
Micro-Batch Streaming
        ↓
Kafka Producer
        ↓
Kafka Broker
        ↓
Kafka Consumer
        ↓
Real-Time Predictions
        ↓
FastAPI Service
        ↓
Streamlit Dashboard
        ↓
Docker Containers
        ↓
Kubernetes Cluster
```

---

# 3. Project Structure

```text
bigdata-ml-pipeline/
│
├── data/
│
├── models/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── models/
│   ├── streaming/
│   ├── api/
│   └── dashboard/
│
├── event_streaming/
│
├── docker/
│
├── k8s/
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 4. Environment Setup

## 4.1 Create Virtual Environment

```powershell
python -m venv bigdata_real
```

Activate:

```powershell
bigdata_real\Scripts\activate
```

---

# 5. Install Dependencies

## 5.1 requirements.txt

```text
pandas
numpy
matplotlib
scikit-learn
pyspark==3.5.1
kafka-python
streamlit
fastapi
uvicorn
google-cloud-bigquery
google-auth
db-dtypes
pyarrow
python-dotenv
```

Install:

```powershell
pip install -r requirements.txt
```

---

# 6. BigQuery Setup

## 6.1 Install Google Cloud SDK

Download:

* [https://cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install)

---

## 6.2 Authenticate

```powershell
gcloud auth application-default login
```

---

## 6.3 Set Credentials

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="credentials/service-account.json"
```

---

# 7. Batch Processing Pipeline

## 7.1 Objective

Traditional machine learning workflow using:

* pandas
* scikit-learn

---

## 7.2 Batch Pipeline Steps

| Step       | Description                 |
| ---------- | --------------------------- |
| Ingestion  | Load BigQuery dataset       |
| Cleaning   | Handle missing values       |
| Encoding   | Encode categorical features |
| Scaling    | Normalize numeric features  |
| Training   | Train ML model              |
| Evaluation | Generate metrics            |

---

## 7.3 Run Batch Training

```powershell
python -m src.models.train_batch
```

---

# 8. Distributed Spark Processing

## 8.1 Objective

Use Apache Spark for:

* distributed preprocessing
* distributed training
* scalable analytics

---

# 8.2 Spark Setup

## Download Spark

Download:

* [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)

Recommended:

* Spark 3.5.x
* Hadoop 3

---

## Extract Spark

Example:

```text
C:\spark\spark-3.5.8-bin-hadoop3
```

---

## Configure Environment Variables

```powershell
$env:SPARK_HOME="C:\spark\spark-3.5.8-bin-hadoop3"
```

Add:

```text
C:\spark\spark-3.5.8-bin-hadoop3\bin
```

to PATH.

---

# 8.3 Verify Spark

```powershell
spark-submit --version
```

---

# 8.4 Spark Preprocessing

Run:

```powershell
python -m src.preprocessing.preprocessing
```

Pipeline includes:

* null handling
* feature engineering
* scaling
* Spark DataFrame transformations

---

# 8.5 Spark Distributed Training

Run:

```powershell
python -m src.models.train_spark
```

Includes:

* distributed ML training
* RandomForestClassifier
* Spark ML pipelines
* evaluation metrics

---

# 9. Micro-Batch Streaming

## 9.1 Objective

Real-time inference using:

* Spark Structured Streaming
* micro-batches

---

# 9.2 Streaming Architecture

```text
Incoming Data
      ↓
Spark Structured Streaming
      ↓
Micro-Batch Processing
      ↓
Prediction Engine
      ↓
Console / Dashboard Output
```

---

# 9.3 Run Micro-Batch Inference

```powershell
python -m src.streaming.microbatch_inference
```

---

# 10. Kafka Streaming Setup

## 10.1 Objective

Implement event-driven architecture using Kafka.

---

# 10.2 Download Kafka

Download:

* [https://kafka.apache.org/downloads](https://kafka.apache.org/downloads)

Recommended:

* Kafka 3.x

Extract to:

```text
C:\kafka
```

---

# 10.3 Start Zookeeper

```powershell
.\bin\windows\zookeeper-server-start.bat .\config\zookeeper.properties
```

---

# 10.4 Start Kafka Server

```powershell
.\bin\windows\kafka-server-start.bat .\config\server.properties
```

---

# 10.5 Create Kafka Topic

```powershell
.\bin\windows\kafka-topics.bat ^
--create ^
--topic covid-stream ^
--bootstrap-server localhost:9092 ^
--partitions 1 ^
--replication-factor 1
```

---

# 11. Kafka Producer

## 11.1 Purpose

Simulates real-time streaming events.

---

# 11.2 Run Producer

```powershell
python -m event_streaming.producer
```

Producer:

* loads batch data
* streams records to Kafka
* publishes prediction events

---

# 12. Kafka Consumer

## 12.1 Purpose

Consumes streaming events and performs inference.

---

# 12.2 Run Consumer

```powershell
python -m src.streaming.kafka_consumer
```

Consumer:

* receives Kafka events
* preprocesses data
* generates predictions
* logs outputs

---

# 13. FastAPI Prediction Service

## 13.1 Run API

```powershell
uvicorn src.api.main:app --reload
```

---

# 13.2 Access API Docs

Open:

```text
http://localhost:8000/docs
```

---

# 14. Streamlit Dashboard

## 14.1 Objective

Visual monitoring dashboard for:

* batch metrics
* streaming metrics
* Kafka events
* micro-batches

---

# 14.2 Run Dashboard

```powershell
streamlit run src/dashboard/dashboard.py
```

---

# 15. Docker Setup

## 15.1 Build Docker Image

```powershell
docker build -t bigdata-ml-producer .
```

---

## 15.2 Run Container

```powershell
docker run -p 8000:8000 bigdata-ml-producer
```

---

# 16. Docker Compose

## 16.1 Start Multi-Service Environment

```powershell
docker compose up
```

Services:

* Kafka
* Zookeeper
* API
* Dashboard
* Producer
* Consumer

---

# 17. Kubernetes Deployment

## 17.1 Apply Kubernetes Configurations

```powershell
kubectl apply -f k8s/
```

---

## 17.2 Verify Pods

```powershell
kubectl get pods
```

---

## 17.3 Verify Services

```powershell
kubectl get services
```

---

# 18. Common Errors and Fixes

## Spark Java Gateway Error

Error:

```text
JAVA_GATEWAY_EXITED
```

Fix:

* install Java 17
* configure SPARK_HOME correctly

---

## Missing db-dtypes

Error:

```text
Please install the 'db-dtypes' package
```

Fix:

```powershell
pip install db-dtypes
```

---

## Kafka Import Error

Error:

```text
cannot import name 'KafkaConsumer'
```

Fix:

```powershell
pip install kafka-python
```

---

## Kubernetes ImagePullBackOff

Cause:

* incorrect image names
* missing local Docker images

Fix:

* rebuild images
* verify tags
* use imagePullPolicy: Never

---

# 19. Future Improvements

| Feature             | Planned              |
| ------------------- | -------------------- |
| CI/CD               | GitHub Actions       |
| Cloud Deployment    | GKE / AWS EKS        |
| Monitoring          | Prometheus + Grafana |
| Model Registry      | MLflow               |
| Feature Store       | Feast                |
| Experiment Tracking | Weights & Biases     |

---

# 20. Conclusion

This project demonstrates a complete:

* Big Data pipeline
* MLOps workflow
* distributed ML architecture
* streaming inference system
* cloud-native deployment stack

using modern data engineering and machine learning tools.
