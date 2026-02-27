from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialisation de l'API
app = FastAPI(
    title="API de Maintenance Prédictive",
    description="API REST pour prédire les pannes de machines industrielles via XGBoost.",
    version="1.0.0"
)

# 2. Chargement des modèles au démarrage
try:
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgboost_model.joblib')
except Exception as e:
    print(f"Erreur lors du chargement des modèles : {e}")

# 3. Définition du format de données attendu (Pydantic)
# Cela force l'utilisateur à envoyer les bonnes données
class MachineData(BaseModel):
    vibration_rms: float
    temperature_motor: float
    rpm: int
    pressure_level: float
    operating_mode: str

# --- ENDPOINTS ---

# Endpoint 1 : Vérifier que l'API est en vie (Health Check)
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "L'API de maintenance prédictive est opérationnelle."}

# Endpoint 2 : Faire une prédiction (Predict)
@app.post("/predict")
def predict_failure(data: MachineData):
    try:
        # Transformation des données reçues en DataFrame (format attendu par scikit-learn)
        df = pd.DataFrame([data.dict()])
        
        # Application du preprocessor
        X_prepared = preprocessor.transform(df)
        
        # Prédiction avec XGBoost
        prediction = model.predict(X_prepared)[0]
        probability = model.predict_proba(X_prepared)[0][1]
        
        # Formatage de la réponse
        result = {
            "prediction_class": int(prediction),
            "failure_probability_percent": round(float(probability) * 100, 2),
            "risk_level": "HIGH" if prediction == 1 else "LOW"
        }
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
