import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
import os

mlflow.set_experiment("wine_quality")

with mlflow.start_run():
    print(os.getcwd())
    # df = pd.read_csv(r'C:\Users\Антон\Documents\Innopolis\Practical Machine Learning and Deep Learning\MLOps-assignment1\data\raw\winequality-red.csv', sep=';')
    df = pd.read_csv('../../data/raw/winequality-red.csv', sep = ';')
    X = df.drop('quality', axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(model, "model")
    os.makedirs('../../models', exist_ok=True)
    joblib.dump(model, '../../models/wine_model.joblib')
    joblib.dump(scaler, '../../models/wine_scaler.joblib')
    print("Модель обучена, accuracy:", acc)
