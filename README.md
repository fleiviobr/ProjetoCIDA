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
  - [8. Classificador K-NN (sklearn)](#8-classificador-k-nn-sklearn)
  - [9. Validação Cruzada](#9-validação-cruzada)
  - [10. Métricas de Classificação](#10-métricas-de-classificação)
  - [11. Exemplo KNN Python (Novo Paciente)](#11-exemplo-knn-python-novo-paciente)
  - [12. Classificador SVM](#12-classificador-svm)
  - [13. Outros Classificadores (Árvore, Random Forest, Regressão Logística)](#13-outros-classificadores)
  - [14. Comparação dos Classificadores](#14-comparação-dos-classificadores)
  - [15. Rede Neural (MLP)](#15-rede-neural-mlp)
  - [16. Comparação Final (Rede Neural vs demais)](#16-comparação-final-rede-neural-vs-demais)
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

#### K-Means (versão ajustada)
Cópia da célula anterior que agrupa nas **2 componentes do PCA** (`df_pca[['PC1', 'PC2']]`) em vez de `X_scaled`. Como a clusterização e a visualização passam a ocorrer no mesmo espaço 2D, os grupos aparecem **mais fechados e separados** (a versão original agrupa em 15 dimensões, das quais o gráfico mostra apenas ~15% da variância).

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

### 12. Classificador SVM
Classificação com **Support Vector Machine** (`SVC`, kernel RBF) sobre `X_scaled`, reproduzindo as duas estratégias de divisão usadas no K-NN:
- **Holdout 70% treino / 30% teste** (`train_test_split`, mesmo `random_state=42`)
- **Validação Cruzada Stratified K-Fold com k=10** (`cross_val_score`), com gráfico de acurácia por fold

Métricas calculadas sobre o conjunto de teste do holdout:

| Métrica | Implementação |
|---|---|
| Matriz de Confusão | `ConfusionMatrixDisplay` com rótulos em português |
| Acurácia | `accuracy_score` |
| F1 Score | `f1_score` + `classification_report` completo |

### 13. Outros Classificadores
Mais três classificadores implementados exatamente no mesmo formato do SVM (Holdout 70/30, Validação Cruzada k=10, Matriz de Confusão, Acurácia e F1 Score), todos sobre `X_scaled` com `random_state=42`:
- **Árvore de Decisão** (`DecisionTreeClassifier`)
- **Random Forest** (`RandomForestClassifier`)
- **Regressão Logística** (`LogisticRegression`, `max_iter=1000`)

### 14. Comparação dos Classificadores
Avaliação lado a lado dos cinco classificadores (KNN, SVM, Árvore de Decisão, Random Forest e Regressão Logística):
- Tabela comparativa com Acurácia (Holdout), F1 (Holdout), Acurácia Média (CV 10-Fold) e Desvio Padrão
- Gráfico de barras agrupado comparando as métricas
- Conclusão com a recomendação do modelo mais adequado

| Classificador | Acurácia (Holdout) | F1 (Holdout) | Acurácia Média (CV) | Desvio (CV) |
|---|---|---|---|---|
| **Random Forest** | **0.8062** | **0.6803** | **0.8353** | **0.0150** |
| SVM (RBF) | 0.7504 | 0.6025 | 0.7878 | 0.0153 |
| Regressão Logística | 0.7395 | 0.6147 | 0.7785 | 0.0222 |
| Árvore de Decisão | 0.7256 | 0.6258 | 0.7571 | 0.0295 |
| KNN (k=5) | 0.7178 | 0.5748 | 0.7315 | 0.0225 |

**Resultado:** o **Random Forest** apresentou o melhor desempenho em todas as métricas e a maior estabilidade entre os folds, sendo o classificador recomendado para a análise.

### 15. Rede Neural (MLP)
Classificação com **rede neural feedforward** (`MLPClassifier`), no mesmo formato dos demais (Holdout 70/30, Validação Cruzada k=10, Matriz de Confusão, Acurácia e F1 Score), com a arquitetura detalhada.

**Arquitetura da rede:**

| Camada | Neurônios | Ativação |
|---|---|---|
| Entrada | 15 | — |
| Oculta 1 | 64 | ReLU |
| Oculta 2 | 32 | ReLU |
| Saída | 1 | Sigmoide |

- **Otimizador:** Adam (`learning_rate_init=1e-3`)
- **Regularização L2:** `alpha=1e-4` · **Lote:** `batch_size=32`
- **Parada antecipada** (`early_stopping`, 15 épocas sem melhora; máx. 500), `random_state=42`
- ≈ 3.137 parâmetros treináveis

### 16. Comparação Final (Rede Neural vs demais)
Tabela e gráfico comparando a Rede Neural com Random Forest, SVM, Árvore de Decisão e KNN, sob a mesma divisão Holdout 70/30 e Validação Cruzada (10-Fold):

| Classificador | Acurácia (Holdout) | F1 (Holdout) | Acurácia Média (CV) | Desvio (CV) |
|---|---|---|---|---|
| **Random Forest** | **0.8062** | **0.6803** | **0.8353** | 0.0150 |
| SVM (RBF) | 0.7504 | 0.6025 | 0.7878 | 0.0153 |
| Rede Neural (MLP) | 0.7535 | 0.6131 | 0.7850 | **0.0142** |
| Árvore de Decisão | 0.7256 | 0.6258 | 0.7571 | 0.0295 |
| KNN (k=5) | 0.7178 | 0.5748 | 0.7315 | 0.0225 |

**Resultado:** o **Random Forest** mantém o melhor desempenho geral. A **Rede Neural** é competitiva e a **mais estável** do segundo grupo (menor desvio entre os folds), empatada com o SVM — comportamento esperado em dados tabulares, onde modelos baseados em árvores costumam liderar.

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
