# Databricks notebook source
import dlt
import pyspark.sql.functions as f

# COMMAND ----------

# create custom path using user email
# path = f"file:/Workspace/Repos/{user_email}/gtc-databricks-streaming-predictions/data"
onefile = "file:/Workspace/Repos/konstantinos.ninas@ms.d-one.ai/gtc-databricks-streaming-predictions/data/batch0/2020/1/10.csv"

# listing all files & folders in that path
# dbutils.fs.ls(path)

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
      .csv(
          onefile,
          header=True,
          inferSchema=True, 
          multiLine=True
          )
      # when we change quota
      # spark.read.csv(path, recursiveFileLookup=True, pathGlobFilter="*.csv",header=True,inferSchema=True, multiLine=True)
  )
  return wind_turbines_raw_sdf

@dlt.table(
    name="wind_turbines_curated",
    comment="Post processing table",
    table_properties={
    "quality": "silver"
  } 
)
def wind_turbines_silver():
  wind_turbines_silver_sdf = (
      dlt
      .read("wind_turbines_raw")
      .dropDuplicates()
      .drop(f.col("measured_at"), f.col("categories_sk"))
          .fillna(
            {
                "subtraction": 0
            }
            )
  )
  return wind_turbines_silver_sdf

@dlt.table(
    name="model_predictions",
    comment="Serving table",
    table_properties={
    "quality": "gold"
}
)
def model_predictions():
    return (
        dlt.read("wind_turbines_curated")
        .withColumn("prediction", f.lit(""))
                    # loaded_model_udf(struct(features)))
    )

