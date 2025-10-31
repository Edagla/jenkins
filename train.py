import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Deney adını belirliyoruz
try:
    mlflow.create_experiment("Jenkins_Lokal_Deneyi")
except mlflow.exceptions.RestException:
    pass # Zaten varsa hata verir, önemseme
    
mlflow.set_experiment("Jenkins_Lokal_Deneyi")

print("MLflow Çalışması (Run) başlıyor...")

with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")

    # Basit veri oluşturma
    X = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
    y = np.array([2, 4, 6, 8, 10, 12, 14])
    
    # Model eğitimi
    model = LinearRegression()
    model.fit(X, y)
    rmse = np.sqrt(mean_squared_error(model.predict(X), y))

    # MLflow'a parametre, metrik ve modeli kaydet
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model", registered_model_name="BasitRegresyonModeli")
    
    print(f"Model kaydedildi. RMSE: {rmse}")

print("Çalışma tamamlandı.")
