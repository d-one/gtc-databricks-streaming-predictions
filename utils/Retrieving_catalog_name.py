# Databricks notebook source
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
