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
    # load silver dataset dataset

    wind_turbines_features_sdf = (dlt
                                 .read("wind_turbines_curated")
                                 .drop("subtraction")
    )

    # make prediction
    prediction_sdf = (wind_turbines_features_sdf
                    .withColumn("prediction", predict_func(*wind_turbines_features_sdf
                                                            .drop("wt_sk")
                                                            .columns)
                                )
    )

    return (
        dlt.read("wind_turbines_curated")
        .select("wt_sk", "wind_speed", "power", "subtraction")
        .join(
          prediction_sdf
          ,["wt_sk", "wind_speed", "power"]
          ,"inner"
        )
    )

