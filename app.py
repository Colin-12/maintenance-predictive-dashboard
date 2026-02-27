import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Maintenance Prédictive IA", page_icon="⚙️", layout="wide")

# --- CHARGEMENT DES MODÈLES (Mis en cache pour la rapidité) ---
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
*Modèle propulsé par XGBoost (Recall : 0.95)*
""")

st.divider()

# --- INTERFACE UTILISATEUR (Sidebar pour le scénario) ---
st.sidebar.header("📊 Saisir un scénario (Capteurs)")

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

# --- PRÉDICTION ---
# Application du preprocessor
input_prepared = preprocessor.transform(input_df)

# Prédiction
prediction = model.predict(input_prepared)
prediction_proba = model.predict_proba(input_prepared)[0][1] # Probabilité de la classe 1 (Panne)

st.divider()
st.subheader("🚨 Résultat de l'analyse IA")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Probabilité de panne (24h)", value=f"{prediction_proba * 100:.1f} %")
    if prediction[0] == 1:
        st.error("⚠️ ALERTE : Risque élevé de panne détecté. Intervention recommandée.")
    else:
        st.success("✅ Machine saine. Aucun risque immédiat détecté.")

with col2:
    st.progress(float(prediction_proba))
    st.caption("Jauge de risque (0% = Sain, 100% = Panne imminente)")

# --- EXPLICABILITÉ (Feature Importance globale) ---
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
