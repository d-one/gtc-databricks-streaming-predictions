# Databricks notebook source
# MAGIC %md 
# MAGIC # Generating Live Predictions by utilizing Delta Live Tables (DLT) #
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ## Data Serving
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ##### This is the third and final part of the DLT pipeline that we are going to build. At this point, we will serve live predictions from a binary classification model and store them in a Databricks Streaming Table.
# MAGIC -------------------------------------------------------------------------------------------------------

# COMMAND ----------

# importing all the necessary libraries
import pyspark.sql.functions as f
import mlflow
import mlflow.pyfunc
import dlt

# COMMAND ----------

# set up catalog name either by workflow parameters or by using current user's id
user_email = spark.sql('select current_user() as user').collect()[0]['user']
try:
    catalog_name = dbutils.widgets.get("CATALOG_NAME")
except:
    catalog_name = user_email.split('@')[0].replace(".", "_").replace("-", "_")

# COMMAND ----------

# set registry to your UC
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

model_name = f"konstantinos_ninas.gold.decision_tree_ml_model"
latest_version = get_latest_model_version(model_name)
model_version_uri = f"models:/{model_name}/{latest_version}"

print(f"Loading registered model version from URI: '{model_version_uri}'")
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

predict_func = mlflow.pyfunc.spark_udf(
    spark,
    model_version_uri)

# COMMAND ----------

# loading silver dataset
wind_turbines_silver_sdf = spark.read.table(f"{catalog_name}.silver.wind_turbines_curated")

# defining all columns to be selected
columns_to_be_selected = wind_turbines_silver_sdf.columns

# dropping the label, to produce predictions
wind_turbines_clean_sdf = wind_turbines_silver_sdf.drop("subtraction")

# produce prediction on new data
prediction_sdf = (wind_turbines_clean_sdf
                .withColumn("prediction", predict_func(*wind_turbines_clean_sdf
                                                        .drop("wt_sk", "measured_at")
                                                        .columns)
                            )
)

# create a table that includes both the label and the prediction to make it available downstream
output_sdf = (
    wind_turbines_silver_sdf
    .select("wt_sk", "subtraction", "measured_at")
    .join(
      prediction_sdf
      ,on=["wt_sk", "measured_at"]
      ,how="inner"
    )
    .select(*columns_to_be_selected, "prediction")
)




# COMMAND ----------

# writing the gold layer table
tableExists=spark.catalog.tableExists(f"{catalog_name}.gold.wind_turbines_predictions")
if tableExists:
    output_sdf.write.mode("append").saveAsTable(f"{catalog_name}.gold.wind_turbines_predictions")
else:
    output_sdf.write.mode("overwrite").saveAsTable(f"{catalog_name}.gold.wind_turbines_predictions")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to load a trained model from the Model Registry using mlflow
# MAGIC * To create user-defined-functions (udf) on the loaded model
# MAGIC * Apply the functions to generate live predictions
# MAGIC
# MAGIC **Next:** Demo to show how to keep track of the model's prediction performance
