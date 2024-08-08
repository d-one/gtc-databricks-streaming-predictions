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
import pyspark.sql.functions as f
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

# ********* workflow parameters ********* #
# set parameters here only if running notebook, for example:
# dbutils.widgets.text("CATALOG_NAME", "konstantinos_ninas")
# dbutils.widgets.text("OVERWRITE_TABLE", "False")

# COMMAND ----------

# set up catalog name either by workflow parameters or by using current user's id
user_email = spark.sql('select current_user() as user').collect()[0]['user']
try:
    catalog_name = dbutils.widgets.get("CATALOG_NAME")
except:
    catalog_name = user_email.split('@')[0].replace(".", "_").replace("-", "_")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's review the ingested data

# COMMAND ----------

wind_turbines_bronze_sdf = spark.table(f"{catalog_name}.bronze.wind_turbines")
display(wind_turbines_bronze_sdf)

# COMMAND ----------

wind_turbines_bronze_sdf.printSchema()

# COMMAND ----------

dbutils.data.summarize(wind_turbines_bronze_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Data preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's validate that there are no duplicate rows in our data. 

# COMMAND ----------

duplicate_rows = (wind_turbines_bronze_sdf
 .groupBy(wind_turbines_bronze_sdf.columns)
 .count()
 .filter(f.col("count")>1)
 .count()
)

wind_turbines_sdf = wind_turbines_bronze_sdf.dropDuplicates()

print(f"{duplicate_rows} in the dataframe will be dropped")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The feature measured_at represents the timestamp in UTC format corresponding to the measurement time. The number of unique entries is lower than the total count as the dataset contains entries for multiple wind turbines, recorded at the same time.
# MAGIC
# MAGIC Since we want to do row-level predictions, we will remove the timestamps from the analysis.
# MAGIC
# MAGIC Also, "subtraction" column is binary and indicates if an error occured, and column "categories_sk" shows the erorr type. We only need to know if an error occured, so we can drop the column "categories_sk" as well. 

# COMMAND ----------

wind_turbines_sdf = wind_turbines_sdf.drop(f.col("measured_at"), f.col("categories_sk"))

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC It seems like only the column "subtraction" has nulls. In this case, we know that nulls mean that turbine did not have any error so we can impute it with 0.

# COMMAND ----------

wind_turbines_sdf = (
    wind_turbines_sdf
    .fillna(
        {
            "subtraction": 0
        }
    )
)

wind_turbines_sdf.display()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Finally, let's check if there is any correlation between the columns.

# COMMAND ----------

wind_turbines_df = wind_turbines_sdf.toPandas()

plt.figure(figsize=(15,10))
sns.heatmap(wind_turbines_df.corr(), annot=True, cmap="coolwarm", linewidth=.5)
plt.title("Correlation heatmap for wind turbines dataset")

# COMMAND ----------

# MAGIC %md 
# MAGIC A strong correlation correlation is noticed between the some column pairs, such as
# MAGIC - wind_speed & power
# MAGIC - power & generator_speed and more..

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC Save the dataframe as a delta table inside the unity catalog

# COMMAND ----------

schema_name = "silver"
table_name = "wind_turbines"

wind_turbines_sdf.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

# COMMAND ----------

dbutils.notebook.exit("End of notebook when running as a workflow task")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to conduct Exploratory Data Analysis (EDA) on a dataset
# MAGIC * Apply Transformations on data using pyspark
# MAGIC * Clean up unnecessary rows/columns of a dataset
# MAGIC * Impute missing data
# MAGIC
# MAGIC **Next:** Go to the Gold Notebook and continue from there

# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
