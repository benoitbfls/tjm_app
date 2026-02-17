import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="TJM & Marge Simulator", layout="wide")

st.title("💼 Simulateur TJM & Marge ESN")

# -----------------------------
# DATA
# -----------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "experience": np.random.randint(1, 20, n),
        "seniorite": np.random.randint(1, 5, n),
        "secteur": np.random.randint(1, 6, n),
        "mission": np.random.randint(1, 4, n),
        "localisation": np.random.randint(1, 4, n),
        "rarete": np.random.randint(1, 5, n)
    })

    df["TJM"] = (
        df["experience"] * 12
        + df["seniorite"] * 90
        + df["secteur"] * 40
        + df["mission"] * 60
        + df["localisation"] * 70
        + df["rarete"] * 80
        + np.random.normal(0, 40, n)
    )

    return df

df = generate_data()

# -----------------------------
# MODEL
# -----------------------------
@st.cache_resource
def train_model():
    X = df.drop("TJM", axis=1)
    y = df["TJM"]
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

model = train_model()

# -----------------------------
# SIDEBAR PROFIL
# -----------------------------
st.sidebar.header("👤 Profil consultant")

experience = st.sidebar.slider("Expérience", 0, 20, 5)

seniorite_map = {"Junior": 1, "Confirmé": 2, "Senior": 3, "Expert": 4}
seniorite = seniorite_map[st.sidebar.selectbox("Séniorité", list(seniorite_map))]

secteur_map = {"Banque": 1, "Assurance": 2, "Industrie": 3, "Retail": 4, "IT": 5}
secteur = secteur_map[st.sidebar.selectbox("Secteur", list(secteur_map))]

mission_map = {"RUN": 1, "Projet": 2, "Transformation": 3}
mission = mission_map[st.sidebar.selectbox("Mission", list(mission_map))]

localisation_map = {"Province": 1, "Paris": 2, "Remote": 3}
localisation = localisation_map[st.sidebar.selectbox("Localisation", list(localisation_map))]

rarete = st.sidebar.slider("Rareté du profil", 1, 4, 2)

# -----------------------------
# INPUT MODEL
# -----------------------------
input_data = pd.DataFrame([{
    "experience": experience,
    "seniorite": seniorite,
    "secteur": secteur,
    "mission": mission,
    "localisation": localisation,
    "rarete": rarete
}])

prediction = model.predict(input_data)[0]

# -----------------------------
# TJM
# -----------------------------
st.subheader("💰 TJM estimé")
st.metric("TJM", f"{prediction:.0f} €")

# -----------------------------
# PARAMETRES FINANCIERS
# -----------------------------
st.sidebar.header("💼 Paramètres financiers")

salaire_brut = st.sidebar.number_input("Salaire brut annuel (€)", 30000, 120000, 45000)
charges = st.sidebar.slider("Coefficient charges", 1.2, 2.0, 1.45)
staffing = st.sidebar.slider("Taux de staffing", 0.5, 1.0, 0.8)
jours_factures = st.sidebar.slider("Jours facturés", 150, 220, 200)

# -----------------------------
# CALCULS
# -----------------------------
ca = prediction * jours_factures * staffing
cout = salaire_brut * charges
marge = ca - cout
taux_marge = (marge / ca) * 100 if ca > 0 else 0

# -----------------------------
# KPI
# -----------------------------
st.subheader("📊 KPI Business")

col1, col2, col3 = st.columns(3)

col1.metric("CA annuel", f"{ca:,.0f} €")
col2.metric("Coût chargé", f"{cout:,.0f} €")
col3.metric("Marge brute", f"{marge:,.0f} €")

st.metric("Taux de marge", f"{taux_marge:.1f} %")

# -----------------------------
# POSITIONNEMENT
# -----------------------------
st.subheader("📈 Positionnement marché")

fig = px.box(df, y="TJM")
fig.add_hline(y=prediction, line_dash="dash", line_color="red")

st.plotly_chart(fig, use_container_width=True)
