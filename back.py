import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import streamlit as st

# Dados fornecidos
data = pd.DataFrame({
    'Tamanho do Projeto': [3.00, 2.50, 3.00, 3.00, 1.50, 1.50, 1.50, 4.00, 4.00, 1.50],
    'Tamanho da Equipe': [3.00, 2.00, 3.00, 3.00, 2.00, 1.50, 1.50, 4.00, 1.50, 1.50],
    'Estabilidade dos Requisitos': [2.00, 1.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Relacionamento da Equipe': [1.00, 2.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00],
    'Compromisso com o Cliente': [1.00, 2.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00],
    'Clareza do Escopo': [2.00, 1.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Clareza o Risco': [2.00, 1.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Estabilidade Ambiental': [2.00, 1.00, 2.00, 2.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Flexibilidade dos Stakeholders': [1.00, 2.00, 1.00, 1.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00],
    'teste': ['Water Fall', 'Lean', 'V-Model', 'Six Sigma', 'Scrum', 'XP', 'Crystal', 'Agile', 'DevOps', 'Kanban']
})

# Separe as features e o target
X = data.drop("teste", axis=1)
y = data["teste"]

# Normalize as features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Escolha o número de clusters
n_clusters = len(np.unique(y))  # Escolher o número de clusters baseado no número de classes no target

# Inicialize e ajuste o modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Adicione os rótulos de cluster ao DataFrame original
data['cluster'] = kmeans.labels_

# Analise a qualidade da clusterização (opcional)
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
st.write(f'Silhouette Score: {silhouette_avg}')

# Visualize os clusters e os targets
cluster_target_mapping = data.groupby('cluster')['teste'].agg(lambda x: x.mode().iloc[0])
st.write(cluster_target_mapping)

# Função para prever o cluster de uma nova amostra
def predict_cluster(new_sample, model, scaler):
    new_sample_scaled = scaler.transform([new_sample])
    cluster_label = model.predict(new_sample_scaled)[0]
    predicted_target = cluster_target_mapping.loc[cluster_label]
    return predicted_target

st.title("Previsão de Cluster")

# Interface de entrada do usuário
tam_proj = st.number_input('Tamanho do Projeto', min_value=0.0, max_value=10.0, value=3.0)
tam_equipe = st.number_input('Tamanho da Equipe', min_value=0.0, max_value=10.0, value=3.0)
estab_req = st.number_input('Estabilidade dos Requisitos', min_value=0.0, max_value=10.0, value=2.0)
rel_equipe = st.number_input('Relacionamento da Equipe', min_value=0.0, max_value=10.0, value=1.0)
comp_cliente = st.number_input('Compromisso com o Cliente', min_value=0.0, max_value=10.0, value=1.0)
clar_escopo = st.number_input('Clareza do Escopo', min_value=0.0, max_value=10.0, value=2.0)
clar_risco = st.number_input('Clareza o Risco', min_value=0.0, max_value=10.0, value=2.0)
estab_ambiental = st.number_input('Estabilidade Ambiental', min_value=0.0, max_value=10.0, value=2.0)
flex_stake = st.number_input('Flexibilidade dos Stakeholders', min_value=0.0, max_value=10.0, value=1.0)

# Nova amostra baseada nas entradas do usuário
new_sample = [tam_proj, tam_equipe, estab_req, rel_equipe, comp_cliente, clar_escopo, clar_risco, estab_ambiental, flex_stake]

# Botão para previsão
if st.button('Prever Cluster'):
    predicted_target = predict_cluster(new_sample, kmeans, scaler)
    st.write(f'Predicted Target: {predicted_target}')