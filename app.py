import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="TJM Predictor", layout="wide")

st.title("💼 Pricing & Prédiction du TJM")

# -----------------------------
# DATASET SYNTHETIQUE METIER
# -----------------------------
@st.cache_data
def generate_data():

    np.random.seed(42)
    n = 1000

    data = pd.DataFrame({
        "experience": np.random.randint(1, 20, n),
        "seniorite": np.random.randint(1, 5, n),
        "secteur": np.random.randint(1, 6, n),
        "mission": np.random.randint(1, 4, n),
        "localisation": np.random.randint(1, 4, n),
        "rarete": np.random.randint(1, 5, n)
    })

    data["TJM"] = (
        data["experience"] * 12
        + data["seniorite"] * 90
        + data["secteur"] * 40
        + data["mission"] * 60
        + data["localisation"] * 70
        + data["rarete"] * 80
        + np.random.normal(0, 40, n)
    )

    return data


df = generate_data()

# -----------------------------
# MODEL
# -----------------------------
X = df.drop("TJM", axis=1)
y = df["TJM"]

model = RandomForestRegressor()
model.fit(X, y)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🧾 Profil consultant")

experience = st.sidebar.slider("Expérience (années)", 0, 20, 5)

seniorite = st.sidebar.selectbox(
    "Séniorité",
    {"Junior": 1, "Confirmé": 2, "Senior": 3, "Expert": 4}
)

secteur = st.sidebar.selectbox(
    "Secteur",
    {"Banque": 1, "Assurance": 2, "Industrie": 3, "Retail": 4, "IT": 5}
)

mission = st.sidebar.selectbox(
    "Type de mission",
    {"RUN": 1, "Projet": 2, "Transformation": 3}
)

localisation = st.sidebar.selectbox(
    "Localisation",
    {"Province": 1, "Paris": 2, "Remote": 3}
)

rarete = st.sidebar.slider("Rareté du profil", 1, 4, 2)

# -----------------------------
# PREDICTION
# -----------------------------
input_data = pd.DataFrame({
    "experience": [experience],
    "seniorite": [seniorite],
    "secteur": [secteur],
    "mission": [mission],
    "localisation": [localisation],
    "rarete": [rarete],
})

prediction = model.predict(input_data)[0]

st.metric("💰 TJM estimé", f"{prediction:.0f} €")

# -----------------------------
# SIMULATEUR REVENU
# -----------------------------
st.subheader("📈 Simulation de revenu")

nb_jours = st.slider("Nombre de jours facturés / an", 150, 220, 200)

ca = prediction * nb_jours

st.metric("Chiffre d'affaires annuel estimé", f"{ca:,.0f} €")

# -----------------------------
# POSITIONNEMENT MARCHE
# -----------------------------
st.subheader("📊 Positionnement vs marché")

fig = px.box(df, y="TJM")
fig.add_hline(y=prediction, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)
