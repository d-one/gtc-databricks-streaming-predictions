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
import pyspark.sql.types as t

# COMMAND ----------

# set up catalog name either by workflow parameters or by using current user's id
user_email = spark.sql('select current_user() as user').collect()[0]['user']
catalog_name = user_email.split('@')[0].replace(".", "_").replace("-", "_")

# COMMAND ----------

# receive secrets to access the storage container of the data
storage_account_access_key = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_access_key')
storage_account_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='storage_account_name')
container_name = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='container_name')
sas_token = dbutils.secrets.get(scope='gtc-workshop-streaming-predictions', key='sas_token')

# specifying the path the streaming files will be found in
source_path = (f"/mnt/{container_name}/")

# Create the source URL with the SAS token
source_url = f"wasbs://{container_name}@{storage_account_name}.blob.core.windows.net/"

# COMMAND ----------

# Mount the storage
try:
  dbutils.fs.mount(
    source = source_url,
    mount_point = source_path,
    extra_configs = {f"fs.azure.sas.{container_name}.{storage_account_name}.blob.core.windows.net": sas_token}
  )
  print("Blob storage succesfully mounted to dbfs")
except:
  print("Blob already mounted!")

# COMMAND ----------

display(dbutils.fs.ls("/mnt/{container_name}/data/batch2/2020/"))

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
      .load(f"dbfs:{source_path}",header=True)
      )
except:
    print("File does not exist, please make sure that your path is correct and that you have pulled the repository to databricks repos")

# COMMAND ----------

wind_turbines_raw_sdf.display()

# COMMAND ----------

# Define the path to the image
image_path = f"/Workspace/Repos/{user_email}/databricks-streaming-predictions/utils/raw_data_info.jpg"

# Read and encode the image to base64 (since displayHTML needs HTML input)
with open(image_path, "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

# Create an HTML string to display the image
html = f'<img src="data:image/png;base64,{encoded_image}" alt="Description" style="width:800px;"/>'

# Display the image
displayHTML(html)


# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will create our own catalog where we will store all of our data later

# COMMAND ----------

# we use spark to assign a dynamic name to our catalog
spark.sql(
    f"""
    CREATE CATALOG IF NOT EXISTS {catalog_name}
    """
)

# COMMAND ----------

schemas_needed = ["bronze", "silver", "gold"]

for schema_name in schemas_needed:
    # we use spark to create the bronze, silver and gold schemas in our catalog
    spark.sql(
        f"""
        CREATE SCHEMA IF NOT EXISTS {catalog_name}.{schema_name}
        """
    )
    print(f"{schema_name} schema created under {catalog_name} catalog")
