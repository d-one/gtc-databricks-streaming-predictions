# Databricks notebook source
import mlflow
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV

# COMMAND ----------

user_email = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
catalog_name = user_email.split("@")[0].replace(".", "_")

# COMMAND ----------

wind_turbines_sdf = spark.read.table(f"{catalog_name}.silver.wind_turbines")

# COMMAND ----------

# Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = wind_turbines_sdf.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.drop("subtraction")
response_df = train_df.select("subtraction")

test_features_df = test_df.drop("subtraction")
test_response_df = test_df.select("subtraction")

# COMMAND ----------

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

train_df, test_df = wind_turbines_sdf.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.drop("subtraction")
response_df = train_df.select("subtraction")

test_features_df = test_df.drop("subtraction")
test_response_df = test_df.select("subtraction")


# Covert data to pandas dataframes
X_train_pdf = features_df.drop("wt_sk").toPandas()
Y_train_pdf = response_df.toPandas()

X_test_pdf = test_features_df.drop("wt_sk").toPandas()
Y_test_pdf = test_response_df.toPandas()

clf = DecisionTreeClassifier(random_state=42)

# Use 3-level namespace for model name
model_name = f"{catalog_name}.gold.decision_tree_ml_model" 

# Define the hyperparameter grid
param_grid = {
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Set up GridSearchCV to tune hyperparameters
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, 
                           scoring='f1', cv=5, n_jobs=-1, verbose=2)

with mlflow.start_run(run_name="wind-turbines-decision-tree") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    grid_search.fit(X_train_pdf, Y_train_pdf)

    # Get the best estimator
    best_clf = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred = best_clf.predict(X_test_pdf)

    # Log model and push to registry
    signature = infer_signature(X_train_pdf, Y_train_pdf)
    mlflow.sklearn.log_model(
        best_clf,
        artifact_path="decision_tree",
        signature=signature,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Baseline)
    client.set_registered_model_alias(model_name, "Baseline", get_latest_model_version(model_name))
