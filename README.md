import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Modelo simples treinado com dados simulados
def carregar_modelo():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X = pd.DataFrame({
        'idade': np.random.randint(20, 60, size=500),
        'sexo': np.random.choice([0, 1], size=500),
        'dor_lombar_noturna': np.random.choice([0, 1], size=500),
        'rigidez_matinal': np.random.choice([0, 1], size=500),
        'hla_b27': np.random.choice([0, 1], size=500),
        'vhs': np.random.normal(15, 5, size=500).clip(min=0),
        'pcr': np.random.normal(5, 2, size=500).clip(min=0),
        'hist_familiar': np.random.choice([0, 1], size=500),
        'resposta_aine': np.random.choice([0, 1], size=500),
        'rm_sacroiliaca': np.random.choice([0, 1], size=500),
    })
    y = (
        (X['hla_b27'] == 1) &
        (X['dor_lombar_noturna'] == 1) &
        (X['rm_sacroiliaca'] == 1)
    ).astype(int)
    model.fit(X, y)
    return model

model = carregar_modelo()

st.title("Análise de Risco de Espondilite Anquilosante")

# Interface
idade = st.slider("Idade", 15, 80, 35)
sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
dor_lombar_noturna = st.checkbox("Dor lombar noturna")
rigidez_matinal = st.checkbox("Rigidez matinal > 30 min")
hla_b27 = st.checkbox("HLA-B27 positivo")
vhs = st.number_input("VHS (mm/h)", 0.0, 100.0, 15.0)
pcr = st.number_input("PCR (mg/L)", 0.0, 50.0, 5.0)
hist_familiar = st.checkbox("Histórico familiar de EA")
resposta_aine = st.checkbox("Melhora com AINEs")
rm_sacroiliaca = st.checkbox("Alterações na RM sacroilíaca")

# Codificação
sexo_bin = 1 if sexo == "Masculino" else 0
input_data = pd.DataFrame([[
    idade, sexo_bin, int(dor_lombar_noturna), int(rigidez_matinal),
    int(hla_b27), vhs, pcr, int(hist_familiar),
    int(resposta_aine), int(rm_sacroiliaca)
]], columns=[
    'idade', 'sexo', 'dor_lombar_noturna', 'rigidez_matinal',
    'hla_b27', 'vhs', 'pcr', 'hist_familiar',
    'resposta_aine', 'rm_sacroiliaca'
])

# Previsão
if st.button("Analisar risco"):
    prob = model.predict_proba(input_data)[0][1]
    st.subheader(f"Risco estimado de Espondilite Anquilosante: {prob:.2%}")
    if prob > 0.5:
        st.error("Alto risco — considerar avaliação especializada.")
    else:
        st.success("Baixo risco — manter acompanhamento clínico.")

