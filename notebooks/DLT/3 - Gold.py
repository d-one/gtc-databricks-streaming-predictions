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

# set registry to your UC
mlflow.set_registry_uri("databricks-uc")

model_name = f"konstantinos_ninas.gold.decision_tree_ml_model" 
model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

predict_func = mlflow.pyfunc.spark_udf(
    spark,
    model_version_uri)

# COMMAND ----------

@dlt.table(
  name=f"{catalog_name}.gold.model_predictions",
    comment="Serving table",
    table_properties={
    "quality": "gold"
}
)

def model_predictions():
    # load silver dataset dataset

    wind_turbines_features_sdf = (dlt
                                 .read("wind_turbines_curated")
                                 .drop("subtraction")
    )

    # make prediction
    prediction_sdf = (wind_turbines_features_sdf
                    .withColumn("prediction", predict_func(*wind_turbines_features_sdf
                                                            .drop("wt_sk", "measured_at")
                                                            .columns)
                                )
    )

    return (
        dlt.read("wind_turbines_curated")
        .select("wt_sk", "measured_at")
        .join(
          prediction_sdf
          ,["wt_sk", "measured_at"]
          ,"inner"
        )
    )


# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to load a trained model from the Model Registry using mlflow
# MAGIC * To create functions on the loaded model 
# MAGIC * Apply the functions to generate live predictions
# MAGIC
# MAGIC **Next:** Demo to show how to keep track of the model's prediction performance
