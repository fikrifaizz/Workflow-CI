import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

# === CONFIG ===
mlflow.set_tracking_uri("https://fikrifaizz:0c53f4378e077d3ab698ce58b32bc9e3b5e5e915@dagshub.com/fikrifaizz/SMSML_Fikri-Faiz-Zulfadhli.mlflow")
mlflow.set_experiment("MLFlow Tuning with DagsHub")

# === LOAD DATA ===
import os
dataset_path = os.path.join(os.path.dirname(__file__), "ai_dev_productivity_processed.csv")
df = pd.read_csv(dataset_path)
X = df.drop("task_success", axis=1)
y = df["task_success"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === HYPERPARAMETER TUNING ===
n_estimators_list = [50, 100, 200]
max_depth_list = [5, 10, None]

for n in n_estimators_list:
    for d in max_depth_list:
        with mlflow.start_run():
            model = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)

            start_time = time.time()
            model.fit(X_train, y_train)
            end_time = time.time()
            elapsed = end_time - start_time

            y_pred = model.predict(X_test)

            # Log parameters
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", n)
            mlflow.log_param("max_depth", d if d is not None else "None")

            # Log metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("training_time_sec", elapsed)

            # Simpan model
            mlflow.sklearn.log_model(model, "model")