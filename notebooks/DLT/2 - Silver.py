# Databricks notebook source
# MAGIC %md
# MAGIC # Generating Live Predictions by utilizing Delta Live Tables (DLT) #
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ## Data preparation & preprocessing
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ##### This is the second part of the DLT pipeline that we are going to build. At this point, we will investigate the data and apply data cleansing & transformations where needed.
# MAGIC -------------------------------------------------------------------------------------------------------

# COMMAND ----------

# importing all the necessary libraries
import dlt
import pyspark.sql.functions as f

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

@dlt.view
def wind_turbines_raw():
  return spark.read.table(f"{catalog_name}.bronze.wind_turbines_raw")

@dlt.table(
  name=f"wind_turbines_curated",
    comment="Post processing table",
    table_properties={
    "quality": "silver"
  } 
)

@dlt.expect_or_drop("valid_dates", "year(measured_at)=2020 and measured_at is not null")

def wind_turbines_curated():
  wind_turbines_silver_sdf = (
      dlt
      .read(f"wind_turbines_raw")
      .dropDuplicates()
      .drop(f.col("categories_sk"))
  )
  return wind_turbines_silver_sdf

# COMMAND ----------

# MAGIC %md
# MAGIC # Exercise
# MAGIC
# MAGIC Time to test your knewly acquired knowledge!
# MAGIC
# MAGIC Try enriching the above step of the medallion architecture by providing expectations on the data to ensure data quality.
# MAGIC Specifically:
# MAGIC   - For the column wind_speed, only allow values within the range of 0-1
# MAGIC   - For the column rotor_speed, only allow positive values (>0)
# MAGIC   - The wind turbines we are tracking have the ids 1, 2, 3 & 4. Also, they should not be null. We should only allow generators with those values to pass.
# MAGIC   - For the column subtraction (our prediction target) we only want to allow values 0 & 1. If we receive any other value, we should stop the pipeline. hint: use @dlt.expect_or_fail() function
# MAGIC
# MAGIC _In any occurence of a violation of the above constraints, we need to drop these rows so as to exclude them from being included downstream._
# MAGIC
# MAGIC Hint 1: You should stop running the DLT pipeline to apply these changes
# MAGIC
# MAGIC Hint 2: You should use the @dlt.expect_or_drop function

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * Apply Transformations on data using pyspark
# MAGIC * Clean up unnecessary rows/columns of a dataset
# MAGIC * Impute missing data
# MAGIC
# MAGIC **Next:** Go to the Gold Notebook (3 - Gold) and continue from there
