import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests # <-- La nouveauté indispensable !

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Maintenance Prédictive IA", page_icon="⚙️", layout="wide")

# --- CHARGEMENT DES MODÈLES (Pour l'explicabilité du graphique) ---
@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.joblib')
    model = joblib.load('xgboost_model.joblib')
    return preprocessor, model

preprocessor, model = load_models()

# --- TITRE ET DESCRIPTION ---
st.title("⚙️ Dashboard de Maintenance Prédictive Industrielle")
st.markdown("""
Bienvenue sur l'outil d'aide à la décision. Saisissez les données des capteurs de la machine en temps réel 
pour estimer la probabilité d'une panne dans les 24 prochaines heures.
*Architecture Cloud : Interface Streamlit connectée à une API REST FastAPI (XGBoost)*
""")

st.divider()

# --- INTERFACE UTILISATEUR (Sidebar pour le scénario) ---
st.sidebar.header("📊 Saisir un scénario")

def user_input_features():
    vibration = st.sidebar.slider("Vibration (RMS)", 0.0, 10.0, 2.5, 0.1)
    temp = st.sidebar.slider("Température Moteur (°C)", 20.0, 150.0, 75.0, 1.0)
    rpm = st.sidebar.slider("Vitesse de rotation (RPM)", 0, 3000, 1500, 10)
    pressure = st.sidebar.slider("Pression", 0.0, 200.0, 100.0, 1.0)
    mode = st.sidebar.selectbox("Mode de fonctionnement", ["normal", "peak", "idle"])
    
    data = {
        'vibration_rms': vibration,
        'temperature_motor': temp,
        'rpm': rpm,
        'pressure_level': pressure,
        'operating_mode': mode
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Affichage des données saisies
st.subheader("Données machine actuelles")
st.write(input_df)

# --- PRÉDICTION VIA L'API REST ---
st.divider()
st.subheader("🚨 Résultat de l'analyse IA (via API)")

# L'URL exacte de ton API sur Render
API_URL = "https://api-maintenance-predictive.onrender.com/predict"

# On prépare le "colis" (payload) à envoyer à l'API
payload = {
    "vibration_rms": float(input_df["vibration_rms"].iloc[0]),
    "temperature_motor": float(input_df["temperature_motor"].iloc[0]),
    "rpm": int(input_df["rpm"].iloc[0]),
    "pressure_level": float(input_df["pressure_level"].iloc[0]),
    "operating_mode": str(input_df["operating_mode"].iloc[0])
}

try:
    # On frappe à la porte de l'API avec nos données
    response = requests.post(API_URL, json=payload)
    
    # Si l'API répond avec succès (Code 200)
    if response.status_code == 200:
        resultat_api = response.json()
        
        # Récupération des résultats
        prediction_class = resultat_api["prediction_class"]
        proba_pourcentage = resultat_api["failure_probability_percent"]
        proba_decimale = proba_pourcentage / 100.0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(label="Probabilité de panne (24h)", value=f"{proba_pourcentage:.1f} %")
            if prediction_class == 1:
                st.error("⚠️ ALERTE : Risque élevé de panne détecté. Intervention recommandée.")
            else:
                st.success("✅ Machine saine. Aucun risque immédiat détecté.")

        with col2:
            st.progress(float(proba_decimale))
            st.caption("Jauge de risque (0% = Sain, 100% = Panne imminente)")
    else:
        st.error(f"Erreur de l'API : Code {response.status_code}")

except Exception as e:
    st.error(f"Impossible de joindre l'API sur Render. Détail de l'erreur : {e}")

# --- EXPLICABILITÉ (Feature Importance) ---
st.divider()
st.subheader("🧠 Pourquoi le modèle prend-il cette décision ?")
st.markdown("Voici l'importance globale des capteurs apprise par le modèle XGBoost :")

# Récupération des noms de colonnes et de l'importance
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_features = preprocessor.transformers_[1][2]
num_features = preprocessor.transformers_[0][2]
all_features = num_features + list(cat_encoder.get_feature_names_out(cat_features))

importances = model.feature_importances_
importance_df = pd.DataFrame({'Capteur': all_features, 'Impact': importances}).sort_values(by='Impact', ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Impact', y='Capteur', data=importance_df, palette='Reds_r', ax=ax)
ax.set_title("Poids des capteurs dans la décision (XGBoost)")
st.pyplot(fig)
