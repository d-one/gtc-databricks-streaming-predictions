# Databricks notebook source
# MAGIC %md 
# MAGIC # Generating Live Predictions by utilizing Delta Live Tables (DLT)
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ## Data ingestion
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ##### This is the first part of the DLT pipeline that we are going to build. Starting, we will create the bronze table of the medallion architecture where we will ingest the streaming data in the unity catalog in a streaming delta table.
# MAGIC -------------------------------------------------------------------------------------------------------

# COMMAND ----------

# import libraries
import dlt
import pyspark.sql.functions as f
import pyspark.sql.types as t

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

# receive secrets to access the storage container of the data
storage_account_access_key = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_access_key')
storage_account_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_name')
container_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='container_name')

# COMMAND ----------

# Mounting the blob storage
dbutils.fs.mount(
source = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/",
mount_point = f"/mnt/{container_name}",
extra_configs = {f"fs.azure.account.key.{storage_account_name}.blob.core.windows.net": storage_account_access_key}
)

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

@dlt.table(
  name=f"{catalog_name}.bronze.wind_turbines_raw",
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


# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to read csv files from an Azure Container
# MAGIC * Read a csv file and store it in a spark dataframe
# MAGIC * Create your own catalog & schemas (databases)
# MAGIC * Write the dataframe as a delta table inside your own schema
# MAGIC
# MAGIC **Next:** Go to the Silver Notebook (2 - Silver) and continue from there
