# Databricks notebook source
# MAGIC %md 
# MAGIC # Generating Live Predictions by utilizing Delta Live Tables (DLT)
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ## Data ingestion
# MAGIC -------------------------------------------------------------------------------------------------------
# MAGIC ##### This is the first part of the DLT pipeline that we are going to build. Starting, we will create the bronze table of the medallion architecture where we will ingest the streaming data in the unity catalog in a streaming delta table.
# MAGIC -------------------------------------------------------------------------------------------------------

# COMMAND ----------

# importing libraries
import base64

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

# create custom path using user email
path = f"file:/Workspace/Repos/{user_email}/gtc-databricks-streaming-predictions/data"

# listing all files & folders in that path
dbutils.fs.ls(path)

# COMMAND ----------

# create custom path using user email
specific_folder_path = f"file:/Workspace/Repos/{user_email}/gtc-databricks-streaming-predictions/data/batch0/2020/1"

# listing the top 5 files in a specific folder
dbutils.fs.ls(specific_folder_path)[:5]

# COMMAND ----------

# MAGIC %md 
# MAGIC Now lets load the data into a spark dataframe & display it.

# COMMAND ----------

try:
    wind_turbines_raw_sdf =(
      spark.read.format("csv")
      .option("recursiveFileLookup", "true")
      .option("pathGlobFilter","*.csv")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(path,header=True)
      )
except:
    print("File does not exist, please make sure that your path is correct and that you have pulled the repository to databricks repos")

# COMMAND ----------

wind_turbines_raw_sdf.display()

# COMMAND ----------

# Define the path to the image
image_path = "/Workspace/Repos/konstantinos.ninas@ms.d-one.ai/gtc-databricks-streaming-predictions/utils/raw_data_info.jpg"

# Read and encode the image to base64 (since displayHTML needs HTML input)
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create an HTML string to display the image
html = f'<img src="data:image/png;base64,{encoded_image}" alt="Description" style="width:800px;"/>'

# Display the image
displayHTML(html)


# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create our own catalog where we will store all of our data

# COMMAND ----------

# we use spark to assign a dynamic name to our catalog
spark.sql(
    f"""
    CREATE CATALOG IF NOT EXISTS {catalog_name}
    """
)

# COMMAND ----------

# we use spark to create the bronze schema in our catalog
spark.sql(
    f"""
    CREATE SCHEMA IF NOT EXISTS {catalog_name}.bronze
    """
)

# COMMAND ----------

# MAGIC %md 
# MAGIC # Writing a delta table to Unity Catalog
# MAGIC
# MAGIC Save the dataframe as a delta table inside the unity catalog

# COMMAND ----------

schema_name = "bronze"
table_name = "wind_turbines"

wind_turbines_raw_sdf.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC With the command above we are:
# MAGIC * Specifying the format to be `delta`
# MAGIC * Specifying `mode` to `overwrite` which will write over any existing data on the table (if any, otherwise it will get created)
# MAGIC
# MAGIC
# MAGIC What would be the difference between `append` and `overwrite` in terms of the:
# MAGIC   * How your table would look like?
# MAGIC   * The space used for the data? (Think about how historization and vacuum works)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exit notebook when running as a workflow task

# COMMAND ----------

dbutils.notebook.exit("End of notebook when running as a workflow task")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Read the data
# MAGIC Load the data from the Unity Catalog to see that the table is actually created.

# COMMAND ----------

wind_turbines_bronze_sdf = spark.table(f"{catalog_name}.{schema_name}.{table_name}")
display(wind_turbines_bronze_sdf)

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion & lessons learned
# MAGIC * In this notebook you learned how to read csv files from the Repos
# MAGIC * Read a csv file and store it in a spark dataframe
# MAGIC * Create your own catalog & schemas (databases)
# MAGIC * Write the dataframe as a delta table inside your own schema
# MAGIC
# MAGIC **Next:** Go to the Silver Notebook and continue from there

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exercise (optional)
# MAGIC * Create a new table called `wind_turbines_dev` in the same catalog and schema
# MAGIC * Write the `wind_turbines_raw_sdf` to the table `wind_turbines_dev` using both `append` and `overwrite` to create some history.
# MAGIC * Check the table history 
# MAGIC   - HINT: Check out the [Describe History](https://docs.databricks.com/en/sql/language-manual/delta-describe-history.html) command

# COMMAND ----------


