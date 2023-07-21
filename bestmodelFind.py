from mlflow.tracking import MlflowClient
import mlflow 
import numpy as np

# If you called your experiment something else, replace here
current_experiment=dict(mlflow.get_experiment_by_name("text_classification_ktrain"))
experiment_id=current_experiment['experiment_id']

# To access MLFlow stuff we need to work with MlflowClient
client = MlflowClient()

# Searches runs for a specific attribute
runs = client.search_runs([experiment_id])
print("total runs", len(runs), "experiment id", experiment_id)

# Select the best run according to test_accuracy metric
best_run = np.argmax([f.data.metrics['accuracy'] for f in runs])
best_auc = np.round(runs[best_run].data.metrics['accuracy'], 4)
best_runname = runs[best_run].info.run_name
best_runid = runs[best_run].info.run_id
print(f"Experiment had {len(runs)} rounds")
print(f"Best run name - {best_runname} with run id - {best_runid} has the accuracy of {best_auc}")