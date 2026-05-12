# Projeto Final - Alzheimer's Disease Dataset

Análise exploratória, clusterização e classificação com K-NN aplicados ao [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset), desenvolvido como projeto final da disciplina.

---

## Sumário

- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Etapas Desenvolvidas](#etapas-desenvolvidas)
  - [1. Pré-Processamento dos Dados](#1-pré-processamento-dos-dados)
  - [2. Normalização dos Dados](#2-normalização-dos-dados)
  - [3. Redução de Dimensionalidade (PCA)](#3-redução-de-dimensionalidade-pca)
  - [4. Distribuição de Frequência](#4-distribuição-de-frequência)
  - [5. Medidas de Resumo](#5-medidas-de-resumo)
  - [6. Clusterização](#6-clusterização)
  - [7. Exemplo KNN Python (Implementação Manual)](#7-exemplo-knn-python-implementação-manual)
  - [8. Classificador K-NN (sklearn)](#8-classificador-k-nn-sklearn)
  - [9. Validação Cruzada](#9-validação-cruzada)
  - [10. Métricas de Classificação](#10-métricas-de-classificação)
  - [11. Exemplo KNN Python (Novo Paciente)](#11-exemplo-knn-python-novo-paciente)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Como Executar](#como-executar)

---

## Dataset

| Atributo | Detalhe |
|---|---|
| Fonte | Kaggle - Rabie El Kharoua |
| Instâncias | 2.149 pacientes |
| Features originais | 35 colunas |
| Target | `Diagnosis` (0 = Sem Alzheimer, 1 = Com Alzheimer) |

O dataset contém dados demográficos, hábitos de vida, histórico médico, exames laboratoriais e avaliações cognitivas/funcionais de pacientes.

---

## Estrutura do Projeto

```
ProjetoFinal.ipynb           # Notebook principal
alzheimers_disease_data.csv  # Dataset
README.md
```

---

## Etapas Desenvolvidas

### 1. Pré-Processamento dos Dados
- Leitura do CSV com `pandas`
- Remoção de valores nulos (`dropna`) e duplicatas (`drop_duplicates`)
- Renomeação de todas as colunas para português
- Remoção de colunas irrelevantes para análise (`ID_Paciente`, `Medico_Responsavel`)

### 2. Normalização dos Dados
- **Normalização Min-Max** aplicada nas features contínuas, preservando as variáveis categóricas
- **Matriz de Correlação** (heatmap) para identificar relações entre as variáveis numéricas

### 3. Redução de Dimensionalidade (PCA)
- **PCA 2D** - scatter plot colorido por diagnóstico com variância explicada
- **PCA 3D** - visualização interativa com `plotly`

### 4. Distribuição de Frequência
- Histogramas com KDE para 10 variáveis clínicas: Idade, IMC, Pressão Sistólica/Diastólica, Colesteróis, Triglicerídeos, MMSE e Avaliação Funcional
- Gráfico de barras comparando a prevalência de sintomas e fatores de risco entre pacientes com e sem Alzheimer

### 5. Medidas de Resumo

#### Tendência Central
Tabela com **Intervalo**, **Média**, **Mediana**, **Ponto Médio** e **Moda** para as 10 variáveis clínicas.

#### Dispersão
Tabela com **Amplitude**, **Desvio Padrão**, **Variância** e **Coeficiente de Variação**.

#### Posição Relativa
- Cálculo do **Z-Score** para todas as variáveis contínuas
- **Boxplot padronizado** com limites de outlier em ±3σ
- **KDE comparativo** de todas as variáveis na escala Z

#### Associação
- **Matriz de Covariância**
- **Diagrama de Dispersão** MMSE × Avaliação Funcional com linha de regressão e coeficiente de Pearson

### 6. Clusterização

#### K-Means
- `n_clusters=4`, `n_init=10`, `random_state=42`
- Clusterização sobre `X_scaled` (espaço completo de features)
- Visualização dos clusters projetados no PCA 2D
- Tabela de contingência clusters × diagnóstico real

#### GMM (Gaussian Mixture Model)
- `n_components=4`, `random_state=42`
- Mesma estrutura de visualização e análise do K-Means

### 7. Exemplo KNN Python (Implementação Manual)
Implementação do algoritmo K-NN do zero, sem bibliotecas de ML:
- `cal_dist_euclideana` - distância euclidiana excluindo o rótulo
- `retorna_vizinhos` - ordena por distância e retorna os k mais próximos
- `classifica` - votação majoritária entre os vizinhos
- Teste com dataset sintético 2D de 10 pontos (2 classes)

### 8. Classificador K-NN (sklearn)
- Divisão **Holdout 70% treino / 30% teste** com `train_test_split`
- `KNeighborsClassifier(n_neighbors=5)` treinado sobre `X_scaled`
- Predição e acurácia reportadas no conjunto de teste

### 9. Validação Cruzada
- **Stratified K-Fold com k=10** (`StratifiedKFold`, `cross_val_score`)
- Gráfico de barras com acurácia por fold e linha da média
- Relatório de média e desvio padrão das acurácias

### 10. Métricas de Classificação

| Métrica | Implementação |
|---|---|
| Matriz de Confusão | `ConfusionMatrixDisplay` com rótulos em português |
| Acurácia | `accuracy_score` |
| Precisão e Revocação | `precision_score`, `recall_score` + gráfico de barras |
| F1 Score | `f1_score` + `classification_report` completo |

### 11. Exemplo KNN Python (Novo Paciente)
Classificação de um paciente hipotético usando o modelo treinado:
- Valores inseridos na escala min-max do dataset
- Transformação via `StandardScaler` antes da predição
- Output: diagnóstico previsto + probabilidade de cada classe

---

## Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação de dados |
| `numpy` | Operações numéricas |
| `matplotlib` / `seaborn` | Visualizações estáticas |
| `plotly` | Visualização 3D interativa |
| `scipy` | Distribuição normal (comparação de grupos) |
| `scikit-learn` | PCA, StandardScaler, KNN, KMeans, GMM, métricas |

---

## Como Executar

1. Instale as dependências:
```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

2. Coloque o arquivo `alzheimers_disease_data.csv` na mesma pasta do notebook.

3. Execute o `ProjetoFinal.ipynb` célula a célula, de cima para baixo.
