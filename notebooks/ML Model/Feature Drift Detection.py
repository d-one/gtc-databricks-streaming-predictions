# Databricks notebook source
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType
import pyspark.sql.functions as f
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# reading the last 2 versions of our golden table
wind_turbine_gold_sdf = spark.read.table("konstantinos_ninas.gold.wind_turbines_predictions")
latest_batch_timestamp = (
    wind_turbine_gold_sdf
    .select(f.max("load_timestamp").alias("max_timestamp"))
)
previous_batch_timestamp = (
    wind_turbine_gold_sdf
    .join(
        latest_batch_timestamp,
        [latest_batch_timestamp["max_timestamp"] != wind_turbine_gold_sdf["load_timestamp"],]
    )
    .select(f.max("load_timestamp").alias("max_timestamp"))
    )

# COMMAND ----------

historical_data = (
    wind_turbine_gold_sdf
    .join(
        previous_batch_timestamp
        .withColumnRenamed("max_timestamp", "load_timestamp"),
        "load_timestamp",
        "inner"
    )
    .toPandas()
)
streaming_data = (
    wind_turbine_gold_sdf
    .join(
        latest_batch_timestamp
        .withColumnRenamed("max_timestamp", "load_timestamp"),
        "load_timestamp",
        "inner"
    )
    .toPandas()
)
features = [col for col in historical_data.columns if col not in ["subtraction", "measured_at", "wt_sk", "prediction", "load_timestamp"]]

# COMMAND ----------

# DBTITLE 1,Population Stability Index calculation
def calculate_psi(expected, actual, buckets=10):
    expected_percents, actual_percents = [], []
    
    # Create the quantiles for the buckets (10 buckets)
    quantiles = np.percentile(expected, np.linspace(0, 100, buckets + 1))  # +1 to include the last boundary
    
    for i in range(buckets):
        # Calculate the fraction of values in each bucket for expected and actual
        expected_bucket = ((expected >= quantiles[i]) & (expected < quantiles[i + 1])).mean()
        actual_bucket = ((actual >= quantiles[i]) & (actual < quantiles[i + 1])).mean()

        # Prevent division by zero in the case where a bucket has no actual data
        actual_bucket = max(actual_bucket, 0.0001)
        expected_bucket = max(expected_bucket, 0.0001)

        expected_percents.append(expected_bucket)
        actual_percents.append(actual_bucket)
    
    # Calculate PSI for each bucket
    psi_values = (np.array(actual_percents) - np.array(expected_percents)) * np.log(np.array(actual_percents) / np.array(expected_percents))
    
    return np.sum(psi_values)

# COMMAND ----------

# DBTITLE 1,Kullback-Leibler Divergence
def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    
    # Avoid division by zero and log(0) issues
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    
    return np.sum(p * np.log(p / q))

# COMMAND ----------

kl_dict, psi_dict = {}, {}
for feat in features:
    training_feature = historical_data[feat].values
    streaming_feature = streaming_data[feat].values

    # Calculate PSI
    psi_value = calculate_psi(streaming_feature, training_feature)
    psi_dict[feat] = psi_value
    
    # Calculate KL Divergence
    # Step 1: Define common bins for both distributions
    num_bins = 50
    min_val = min(training_feature.min(), streaming_feature.min())
    max_val = max(training_feature.max(), streaming_feature.max())

    # Create the bins
    bins = np.linspace(min_val, max_val, num_bins)

    # Step 2: Create histograms (counts) for both training and streaming data
    training_hist, _ = np.histogram(training_feature, bins=bins, density=True)
    streaming_hist, _ = np.histogram(streaming_feature, bins=bins, density=True)

    # Step 3: Normalize the histograms to convert them to probability distributions
    training_prob = training_hist / np.sum(training_hist)
    streaming_prob = streaming_hist / np.sum(streaming_hist)

    # Step 4: Calculate KL Divergence
    kl_div = kl_divergence(training_prob, streaming_prob)
    kl_dict[feat] = kl_div

# COMMAND ----------

# DBTITLE 1,drift dataframe
psi_drift_threshold = 0.2
kl_drift_threshold = 0.1

schema = StructType([
    StructField("feature", StringType(), True),
    StructField("psi", DoubleType(), True),
    StructField("kl_divergence", DoubleType(), True)
])

combined_dict = {k: (kl_dict[k], psi_dict[k]) for k in kl_dict}
data_list = [(k, float(v[0]), float(v[1])) for k, v in combined_dict.items()]
drift_sdf = (
    spark.createDataFrame(data_list, schema=schema)
    .withColumn("drift_detected", f.when((f.col("psi") > psi_drift_threshold) & (f.col("kl_divergence") > kl_drift_threshold), 1).otherwise(0))
    .withColumn("drift_detected", f.col("drift_detected").cast(BooleanType()))
)

# COMMAND ----------

# writing the results of drift detection
drift_sdf.write.mode("overwrite").saveAsTable("konstantinos_ninas.gold.drift_detection")
