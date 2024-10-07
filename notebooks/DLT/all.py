# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC %pip install azure-storage-blob
# MAGIC %pip install azure-storage-file-datalake

# COMMAND ----------

# MAGIC %md 
# MAGIC # Generating Live Predictions by utilizing Delta Live Tables (DLT)
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ## Data ingestion
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ##### This is the first part of the DLT pipeline that we are going to build. Starting, we will create the bronze table of the medallion architecture where we will ingest the streaming data in the unity catalog in a streaming delta table.
# MAGIC -------------------------------------------------------------------------------------------------------

# COMMAND ----------

# importing all the necessary libraries
import pyspark.sql.functions as f
import mlflow
import mlflow.pyfunc
import dlt
import pyspark.sql.types as t

# COMMAND ----------

# ********* workflow parameters ********* #
# set parameters here only if running notebook, for example:
dbutils.widgets.text("CATALOG_NAME", "konstantinos_ninas")

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

model_name = f"{catalog_name}.gold.decision_tree_ml_model" 
model_version_uri = f"models:/{model_name}/1"

print(f"Loading registered model version from URI: '{model_version_uri}'")
loaded_model = mlflow.pyfunc.load_model(model_version_uri)

predict_func = mlflow.pyfunc.spark_udf(
    spark,
    model_version_uri)

# COMMAND ----------

# receive secrets to access the storage container of the data
storage_account_access_key = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_access_key')
storage_account_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_name')
container_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='container_name')

# COMMAND ----------

# # Mounting the blob storage
# dbutils.fs.mount(
# source = f"wasbsa://{container_name}@{storage_account_name}.blob.core.windows.net/",
# mount_point = f"/mnt/{container_name}",
# extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
# )

# COMMAND ----------

# specifying the path the streaming files will be found in
source_path = (f"dbfs:/mnt/{container_name}/")

# specify the expected schema of the csv files in the Blob Storage
schema = (t.StructType()
      .add("wt_sk",t.IntegerType(),True)
      .add("measured_at",t.TimestampType(),True)
      .add("wind_speed",t.DoubleType(),True)
      .add("power",t.DoubleType(),True)
      .add("nacelle_direction",t.DoubleType(),True)
      .add("wind_direction",t.IntegerType(),True)
      .add("rotor_speed",t.TimestampType(),True)
      .add("generator_speed",t.DoubleType(),True)
      .add("temp_environment",t.DoubleType(),True)
      .add("temp_hydraulic_oil",t.DoubleType(),True)
      .add("temp_gear_bearing",t.IntegerType(),True)
      .add("cosphi",t.TimestampType(),True)
      .add("blade_angle_avg",t.DoubleType(),True)
      .add("hydraulic_pressure",t.DoubleType(),True)
      .add("subtraction",t.DoubleType(),True)
      .add("categories_sk",t.DoubleType(),True)
)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC use catalog konstantinos_ninas

# COMMAND ----------

# MAGIC %sql
# MAGIC create or replace table wind_turbines_raw using live

# COMMAND ----------

@dlt.table(
  name=f"streaming_wind_turbines_raw",
  comment="Raw inputs table",
  table_properties={
    "quality": "bronze"
  } 
)

def streaming_wind_turbines_raw():
  wind_turbines_raw_sdf = (
      spark
      .read
      .schema(schema)
      .csv(source_path, 
           recursiveFileLookup=True, 
           pathGlobFilter="*.csv",
           header=True,
           multiLine=True
           )
      .fillna(
            {
                "subtraction": 0
            }
            )
  )
  return wind_turbines_raw_sdf


# COMMAND ----------

@dlt.table(
  name=f"streaming_wind_turbines_curated",
    comment="Post processing table",
    table_properties={
    "quality": "silver"
  } 
)

@dlt.expect_or_drop("valid_dates", "year(measured_at)=2020 and measured_at is not null")

def wind_turbines_silver():
  wind_turbines_silver_sdf = (
      dlt
      .read(f"streaming_wind_turbines_raw")
      .dropDuplicates()
      .drop(f.col("categories_sk"))
  )
  return wind_turbines_silver_sdf

# COMMAND ----------



# COMMAND ----------

@dlt.table(
  name=f"streaming_model_predictions",
    comment="Serving table",
    table_properties={
    "quality": "gold"
}
)

def streaming_model_predictions():
    # load silver dataset dataset
    wind_turbines_columns = (dlt
                            .read(f"streaming_wind_turbines_curated")
                            .columns
    )

    # load silver dataset dataset
    wind_turbines_features_sdf = (dlt
                                 .read(f"streaming_wind_turbines_curated")
                                 .drop("subtraction")
    )

    # make prediction
    prediction_sdf = (wind_turbines_features_sdf
                    .withColumn("prediction", predict_func(*wind_turbines_features_sdf
                                                            .drop("wt_sk", "measured_at")
                                                            .columns)
                                )
    )

    output_sdf = (
        dlt.read(f"streaming_wind_turbines_curated")
        .select("wt_sk", "subtraction", "measured_at")
        .join(
          prediction_sdf
          ,on=["wt_sk", "measured_at"]
          ,how="inner"
        )
        .select(*wind_turbines_columns, "prediction")
    )

    return output_sdf


# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select *
# MAGIC from live(wind_turbines_raw)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select *
# MAGIC from live(konstantinos_ninas.bronze.wind_turbines_raw)

# COMMAND ----------

@dlt.table(
  name="wind_turbines_raw",
  comment="Raw inputs table",
  table_properties={
    "quality": "bronze"
  } 
)

def wind_turbines_raw():
  wind_turbines_raw_sdf = (
      spark
      .read
      .schema(schema)
      .csv(source_path, 
           recursiveFileLookup=True, 
           pathGlobFilter="*.csv",
           header=True,
           multiLine=True
           )
      .fillna(
            {
                "subtraction": 0
            }
            )
  )
  return wind_turbines_raw_sdf

dlt.create_target_table()


# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to read csv files from an Azure Container
# MAGIC * Read a csv file and store it in a spark dataframe
# MAGIC * Create your own catalog & schemas (databases)
# MAGIC * Write the dataframe as a delta table inside your own schema
# MAGIC
# MAGIC **Next:** Go to the Silver Notebook (2 - Silver) and continue from there
