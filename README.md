# Projeto Final - Alzheimer's Disease Dataset

Análise exploratória, clusterização e classificação (KNN, SVM, Árvore de Decisão, Random Forest, Regressão Logística e Rede Neural) aplicadas ao [Alzheimer's Disease Dataset](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset), desenvolvido como projeto final da disciplina.

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
  - [8. Classificação](#8-classificação)
  - [9. Teste: Base Balanceada (50/50)](#9-teste-base-balanceada-5050)
  - [10. Conclusão](#10-conclusão)
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
- `n_components=2`, `random_state=42`
- Mesma estrutura de visualização e análise do K-Means

### 7. Exemplo KNN Python (Implementação Manual)
Implementação do algoritmo K-NN do zero, sem bibliotecas de ML:
- `cal_dist_euclideana` - distância euclidiana excluindo o rótulo
- `retorna_vizinhos` - ordena por distância e retorna os k mais próximos
- `classifica` - votação majoritária entre os vizinhos
- Teste com dataset sintético 2D de 10 pontos (2 classes)

### 8. Classificação
Previsão do diagnóstico usando **todas as variáveis** do dataset — as contínuas (em escala min-max) e as **categóricas** (gênero, etnia, histórico médico e os sintomas, como *Queixas de Memória* e *Problemas Comportamentais*). Cada classificador é apresentado **individualmente**, com Holdout (70/30), Validação Cruzada estratificada (k=10), Matriz de Confusão, Acurácia, F1-Score e `classification_report`:

- **KNN** (`KNeighborsClassifier`, k=5)
- **SVM** (`SVC`, kernel RBF)
- **Árvore de Decisão** (`DecisionTreeClassifier`)
- **Random Forest** (`RandomForestClassifier`)
- **Regressão Logística** (`LogisticRegression`, `max_iter=1000`)
- **Rede Neural** (`MLPClassifier`) — arquitetura **32 → 64 (ReLU) → 32 (ReLU) → 1 (sigmoide)**, otimizador Adam, regularização L2 `alpha=1e-4`, `early_stopping`

Ao final, uma **Comparação Geral** (tabela + gráfico de barras):

| Classificador | Acurácia (Holdout) | F1 (Holdout) | Acurácia Média (CV) | Desvio (CV) |
|---|---|---|---|---|
| **Random Forest** | **0.9054** | **0.8635** | **0.9423** | 0.0104 |
| Árvore de Decisão | 0.8775 | 0.8351 | 0.8939 | 0.0084 |
| SVM (RBF) | 0.8109 | 0.7265 | 0.8455 | 0.0179 |
| Regressão Logística | 0.8155 | 0.7419 | 0.8348 | 0.0234 |
| Rede Neural (MLP) | 0.7907 | 0.6993 | 0.8283 | 0.0157 |
| KNN (k=5) | 0.7132 | 0.5387 | 0.7175 | 0.0169 |

**Resultado:** o **Random Forest** lidera em todas as métricas (94,2% de acurácia na validação cruzada), no patamar dos trabalhos publicados com este dataset (~94–95%). Os preditores mais fortes são os indicadores funcionais e cognitivos (Avaliação Funcional, ADL, MMSE) e os sintomas relatados (Queixas de Memória, Problemas Comportamentais).

### 9. Teste: Base Balanceada (50/50)
Teste com a base **balanceada em 50/50** por *undersampling* da classe majoritária, treinando o **Random Forest** com todas as variáveis (Holdout, Validação Cruzada, Matriz de Confusão e métricas):

| Cenário (Random Forest) | Acurácia | F1 | Revocação (Com) | CV |
|---|---|---|---|---|
| Base original | 0.905 | 0.86 | 0.79 | 0.942 |
| **Balanceada (50/50)** | **0.943** | **0.94** | **0.95** | 0.938 |

**Resultado:** o balanceamento eleva a **revocação dos casos positivos de 0,79 para 0,95**, reduzindo os falsos negativos e equilibrando o desempenho entre as classes.

### 10. Conclusão
Célula final do notebook com a **síntese acadêmica** do trabalho: o Random Forest, usando o conjunto completo de variáveis, atinge **~94% de acurácia (validação cruzada)** e é recomendado como modelo preferencial; o balanceamento de classes, avaliado experimentalmente, eleva a revocação dos positivos a **94,7%**, apontando o caminho para reduzir os falsos negativos.

---

## Tecnologias Utilizadas

| Biblioteca | Uso |
|---|---|
| `pandas` | Manipulação de dados |
| `numpy` | Operações numéricas |
| `matplotlib` / `seaborn` | Visualizações estáticas |
| `plotly` | Visualização 3D interativa |
| `scipy` | Distribuição normal (comparação de grupos) |
| `scikit-learn` | PCA, StandardScaler, KMeans, GMM, KNN, SVM, Árvore de Decisão, Random Forest, Regressão Logística, Rede Neural (MLP), validação cruzada e métricas |

---

## Como Executar

1. Instale as dependências:
```bash
pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

2. Coloque o arquivo `alzheimers_disease_data.csv` na mesma pasta do notebook.

3. Execute o `ProjetoFinal.ipynb` célula a célula, de cima para baixo.
