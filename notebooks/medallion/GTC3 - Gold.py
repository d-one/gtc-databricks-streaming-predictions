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

# MAGIC %md
# MAGIC Let's review the clean data

# COMMAND ----------

wind_turbines_clean_sdf = spark.table(f"{catalog_name}.silver.wind_turbines")
display(wind_turbines_clean_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generating Live predictions in the streaming data
# MAGIC First, let's load the pretrained model to produce our predictions

# COMMAND ----------

# set registry to your UC
mlflow.set_registry_uri("databricks-uc")

model_name = f"{catalog_name}.gold.decision_tree_ml_model" 
model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

predict_func = mlflow.pyfunc.spark_udf(
    spark,
    model_version_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC Following, we will apply the prediction function in our dataframe to generate the predictions 

# COMMAND ----------

# prepare test dataset
wind_turbines_features_df = wind_turbines_clean_sdf.drop("subtraction")

# make prediction
prediction_df = (wind_turbines_features_df
                 .withColumn("prediction", predict_func(*wind_turbines_features_df
                                                        .drop("wt_sk")
                                                        .columns)
                             )
)

display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's see the predictions alongside the actual labels

# COMMAND ----------

wind_turbine_predictions_sdf = (
    wind_turbines_clean_sdf
    .select("wt_sk", "wind_speed", "power", "subtraction")
    .join(
        prediction_df
        ,["wt_sk", "wind_speed", "power"]
        ,"inner"
    )
)

wind_turbine_predictions_sdf.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Finally, we can write the data in our gold table in the Unity Catalog to serve the data in a Power BI dashboard

# COMMAND ----------

wind_turbine_predictions_sdf.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.gold.wind_turbines_predictions")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's load the data to validate that they have been actually saved

# COMMAND ----------

schema_name = "gold"
table_name = "wind_turbines_predictions"

wind_turbine_predictions_sdf.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

display(wind_turbine_predictions_sdf)

# COMMAND ----------

dbutils.notebook.exit("End of notebook when running as a workflow task")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to load a trained model from the Model Registry using mlflow
# MAGIC * To create functions on the loaded model 
# MAGIC * Apply the functions to generate live predictions
# MAGIC
# MAGIC **Next:** Demo to show how to keep track of the model's prediction performance
