import os

# Force localhost networking
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"
os.environ["PYSPARK_PYTHON"] = "python"

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[*]") \
    .appName("BigDataML") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

print(" Spark started successfully!")

data = [
    ("Uganda", 100),
    ("Kenya", 200),
    ("Tanzania", 300)
]

df = spark.createDataFrame(data, ["country", "cases"])

df.show()

spark.stop()