import sys, warnings

import ktrain
from ktrain import text
from sklearn import model_selection
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    try:
        df = pd.read_csv("data/sample.csv")
        classes = list(set(df.label.tolist()))
    except Exception as e:
        logger.exception(
            "Unable to load training & test CSV, check the file path. Error: %s", e
        )

    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(df['content'].tolist(), df['label'].tolist())

    trn, val, preproc = text.texts_from_array(x_train=train_x, y_train=train_y,
                                                x_test=valid_x, y_test=valid_y,
                                                class_names=classes,
                                                preprocess_mode='distilbert',
                                                maxlen=256, 
                                                max_features=10000)

    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)
    learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=6, use_multiprocessing=True)

    # learner.lr_find(max_epochs=2) # finding the learning rate
    # learner.lr_plot()

    lrate = float(sys.argv[1]) if len(sys.argv) > 1 else 2e-5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    print("training ktrain distilbert model (lrate={:f}, epochs={:f}):".format(lrate, epochs))
    mlflow.set_experiment("text_classification_ktrain")
    experiment = mlflow.get_experiment_by_name("text_classification_ktrain")
    with mlflow.start_run(experiment_id=experiment.experiment_id, nested=True):
        learner.fit_onecycle(lrate, epochs)
        predictor = ktrain.get_predictor(learner.model, preproc)
        y_pred = predictor.predict(valid_x)
        accuracy = accuracy_score(valid_y, y_pred)
        f1 = f1_score(valid_y, y_pred, average="weighted")
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("f1-score", round(f1, 4))
        mlflow.log_param("lrate", lrate)
        mlflow.log_param("epochs", epochs)
        # save the model
        mlflow.sklearn.log_model(learner, "text_classification1", pyfunc_predict_fn="predict_proba")

