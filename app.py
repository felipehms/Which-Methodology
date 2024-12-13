from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

app = Flask(__name__)

data = pd.DataFrame({
    'Tamanho do Projeto': [5.00, 5.00, 2.00, 2.00, 2.00, 3.00, 2.00],
    'Tamanho da Equipe': [5.00, 5.00, 3.00, 2.00, 2.00, 3.00, 2.00],
    'Estabilidade dos Requisitos': [5.00, 5.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Relacionamento da Equipe': [1.00, 1.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'Compromisso com o Cliente': [3.00, 3.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'Clareza do Escopo': [5.00, 5.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Clareza o Risco': [5.00, 5.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Estabilidade Ambiental': [5.00, 5.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Flexibilidade dos Stakeholders': [1.00, 1.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'Complexidade do Projeto': [2.00, 5.00, 4.00, 4.00, 4.00, 4.00, 4.00],
    'Riscos do Projeto': [2.00, 5.00, 3.00, 4.00, 4.00, 3.00, 3.00],
    'Tempo de Ciclo de Desenvolvimento': [5.00, 4.00, 2.00, 1.00, 1.00, 2.00, 2.00],
    'Flexibilidade para Mudanças': [1.00, 1.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'Custo do Projeto': [3.00, 4.00, 2.00, 3.00, 2.00, 2.00, 2.00],
    'Escopo de Requisitos': [5.00, 5.00, 4.00, 4.00, 4.00, 4.00, 4.00],
    'Gestão de Equipe': [3.00, 3.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'Engajamento do Cliente': [2.00, 4.00, 5.00, 5.00, 5.00, 5.00, 5.00],
    'teste': ['Water Fall', 'V-Model', 'Scrum', 'XP', 'Crystal', 'Agile', 'Kanban']
})

# Separação das features e o target
X = data.drop("teste", axis=1)
y = data["teste"]

# Normalização das features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determinação do número de clusters únicos nos dados
n_clusters = len(data['teste'].unique())

# Inicialização e ajuste do modelo KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)

# Adição dos rótulos de cluster ao DataFrame original
data['cluster'] = kmeans.labels_

# Mapeamento dos clusters
cluster_target_mapping = data.groupby('cluster')['teste'].agg(lambda x: x.mode().iloc[0]).to_dict()

# Função para prever os 3 clusters mais próximos de uma nova amostra
def predict_clusters(new_sample, model, scaler, mapping):
    new_sample_scaled = scaler.transform([new_sample])
    distances = model.transform(new_sample_scaled)
    closest_clusters = np.argsort(distances[0])[:3]
    predicted_targets = [(mapping.get(cluster, "Unknown"), distances[0][cluster]) for cluster in closest_clusters]
    return predicted_targets

# Dicionário de explicações das metodologias
methodology_explanations = {
    "Water Fall": "Waterfall é um modelo sequencial de desenvolvimento de software que segue uma abordagem linear e faseada.",
    "V-Model": "V-Model é um modelo de desenvolvimento em que as fases de verificação e validação são executadas em paralelo.",
    "Scrum": "Scrum é uma estrutura ágil que promove a entrega incremental e iterativa de projetos complexos.",
    "XP": "Extreme Programming (XP) é uma metodologia ágil que se concentra em melhorar a qualidade do software e a capacidade de resposta às mudanças.",
    "Crystal": "Crystal é uma família de metodologias ágeis que se adaptam ao tamanho e criticidade do projeto.",
    "Agile": "Agile é uma abordagem de desenvolvimento que enfatiza a colaboração, flexibilidade e entregas frequentes.",
    "Kanban": "Kanban é um método ágil de gerenciamento de trabalho que usa um sistema visual para controlar a produção."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forms', methods=['GET', 'POST'])
def forms():
    predicted_targets = None
    explanations = []
    if request.method == 'POST':
        try:
            tam_proj = float(request.form['tam_proj'])
            tam_equipe = float(request.form['tam_equipe'])
            estab_req = float(request.form['estab_req'])
            rel_equipe = float(request.form['rel_equipe'])
            comp_cliente = float(request.form['comp_cliente'])
            clar_escopo = float(request.form['clar_escopo'])
            clar_risco = float(request.form['clar_risco'])
            estab_ambiental = float(request.form['estab_ambiental'])
            flex_stake = float(request.form['flex_stake'])
            compl_proj = float(request.form['compl_proj'])
            risco_proj = float(request.form['risco_proj'])
            ciclo_desenvolvimento = float(request.form['ciclo_desenvolvimento'])
            flex_mudancas = float(request.form['flex_mudancas'])
            custo_proj = float(request.form['custo_proj'])
            escopo_requisitos = float(request.form['escopo_requisitos'])
            gestao_equipe = float(request.form['gestao_equipe'])
            engajamento_cliente = float(request.form['engajamento_cliente'])

            new_sample = [
                tam_proj, tam_equipe, estab_req, rel_equipe, comp_cliente, clar_escopo, clar_risco, estab_ambiental, flex_stake,
                compl_proj, risco_proj, ciclo_desenvolvimento, flex_mudancas, custo_proj,
                escopo_requisitos, gestao_equipe, engajamento_cliente
            ]
            predicted_targets = predict_clusters(new_sample, kmeans, scaler, cluster_target_mapping)
            explanations = [methodology_explanations[target[0]] for target in predicted_targets]
        except ValueError as e:
            print(f"Error: {e}")

    return render_template('forms.html', predicted_targets=predicted_targets, explanations=explanations)

if __name__ == '__main__':
    app.run(debug=True)
