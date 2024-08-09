# Databricks notebook source
import mlflow
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# COMMAND ----------

wind_turbines_sdf = spark.read.table("konstantinos_ninas.silver.wind_turbines")

# COMMAND ----------

# Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = wind_turbines_sdf.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.drop("subtraction")
response_df = train_df.select("subtraction")

# COMMAND ----------

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------


# Covert data to pandas dataframes
X_train_pdf = features_df.drop("wt_sk").toPandas()
Y_train_pdf = response_df.toPandas()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Use 3-level namespace for model name
model_name = f"konstantinos_ninas.gold.decision_tree_ml_model" 

with mlflow.start_run(run_name="wind-turbines-decision-tree") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(X_train_pdf, Y_train_pdf)

    # Log model and push to registry
    signature = infer_signature(X_train_pdf, Y_train_pdf)
    mlflow.sklearn.log_model(
        clf,
        artifact_path="decision_tree",
        signature=signature,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Baseline)
    client.set_registered_model_alias(model_name, "Baseline", get_latest_model_version(model_name))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # deployment left - maybe also calculate here the data drift and use it a as a source in power bi. Streaming table - direct query in pbi

# COMMAND ----------


