import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

st.set_page_config(page_title="Prédicteur de TJM", layout="wide")

st.title("💼 Prédiction du TJM d’un consultant")

# -----------------------------
# DATASET SYNTHETIQUE
# -----------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500

    data = pd.DataFrame({
        "experience": np.random.randint(1, 20, n),
        "seniorite": np.random.randint(1, 4, n),
        "certifications": np.random.randint(0, 10, n),
        "domaine": np.random.randint(1, 5, n),
    })

    data["TJM"] = (
        data["experience"] * 15 +
        data["seniorite"] * 100 +
        data["certifications"] * 20 +
        data["domaine"] * 50 +
        np.random.normal(0, 50, n)
    )

    return data

df = generate_data()

# -----------------------------
# ENTRAINEMENT MODELE
# -----------------------------
X = df.drop("TJM", axis=1)
y = df["TJM"]

model = RandomForestRegressor()
model.fit(X, y)

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("🧾 Paramètres du consultant")

experience = st.sidebar.slider("Années d'expérience", 0, 20, 5)
seniorite = st.sidebar.selectbox("Séniorité", [1, 2, 3])
certifications = st.sidebar.slider("Nb certifications", 0, 10, 2)
domaine = st.sidebar.selectbox("Domaine", [1, 2, 3, 4])

# -----------------------------
# PREDICTION
# -----------------------------
input_data = pd.DataFrame({
    "experience": [experience],
    "seniorite": [seniorite],
    "certifications": [certifications],
    "domaine": [domaine],
})

prediction = model.predict(input_data)[0]

st.metric("💰 TJM estimé", f"{prediction:.0f} €")

# -----------------------------
# VISUALISATION
# -----------------------------
st.subheader("Distribution des TJM")

fig = px.histogram(df, x="TJM", nbins=30)
st.plotly_chart(fig, use_container_width=True)
