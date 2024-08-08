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

# COMMAND ----------

# ********* workflow parameters ********* #
# set parameters here only if running notebook, for example:
# dbutils.widgets.text("CATALOG_NAME", "konstantinos_ninas")

# COMMAND ----------

# set up catalog name either by workflow parameters or by using current user's id
user_email = spark.sql('select current_user() as user').collect()[0]['user']
try:
    catalog_name = dbutils.widgets.get("CATALOG_NAME")
except:
    catalog_name = user_email.split('@')[0].replace(".", "_").replace("-", "_")

# COMMAND ----------

wind_turbines_clean_sdf = spark.table(f"{catalog_name}.silver.wind_turbines")
display(wind_turbines_clean_sdf)

# COMMAND ----------

# set registry to your UC
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

model_name = f"konstantinos_ninas.gold.decision_tree_ml_model" 
model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------



# COMMAND ----------

# prepare test dataset
test_features_df = test_df.drop("subtraction")

# make prediction
prediction_df = test_features_df.withColumn("prediction", predict_func(*test_features_df.drop("wt_sk").columns))

display(prediction_df)

# COMMAND ----------

wind_turbine_predictions_sdf = (
    test_df
    .select("wt_sk", "wind_speed", "power", "subtraction")
    .join(
        prediction_df
        ,["wt_sk", "wind_speed", "power"]
        ,"inner"
    )
)

# COMMAND ----------

wind_turbine_predictions_sdf.write.format("delta").mode("overwrite").saveAsTable("konstantinos_ninas.gold.wind_turbinea_predictions")


# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
