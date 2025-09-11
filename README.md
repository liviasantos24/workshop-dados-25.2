#  Projeto de Machine Learning – Previsão de Sobrevivência em One Piece (Fictício)

##  Objetivo
Criar um modelo de **classificação binária** para prever se um personagem de **One Piece** sobreviverá a uma batalha com base em características como força, inteligência, velocidade, tipo de fruta do diabo e habilidades especiais.

---

##  Estrutura do Projeto

### 1. Abordar o problema e analisar
- **Problema:** Prever se um personagem sobrevive a uma batalha.  
- **Tipo:** Classificação binária (0 = não sobrevive, 1 = sobrevive).  
- **Desafios:**
  - Dataset fictício e pequeno;
  - Variáveis categóricas (tipo de fruta, habilidades especiais);
  - Diferentes níveis de força, inteligência e velocidade.

---

### 2. Obter os dados
- **Fonte:** dataset fictício criado diretamente em Python com `pandas`.  
- **Colunas:**  
  - `Nome`  
  - `Forca`  
  - `Inteligencia`  
  - `Velocidade`  
  - `Tipo_Fruta`  
  - `Habilidade`  
  - `Sobrevive` (variável alvo: 1 = sobrevive, 0 = não sobrevive)  

Exemplo inicial do dataset:

| Nome   | Forca | Inteligencia | Velocidade | Tipo_Fruta | Habilidade | Sobrevive |
|--------|-------|--------------|------------|------------|------------|-----------|
| Luffy  | 95    | 70           | 90         | GomuGomu   | Haki       | 1         |
| Zoro   | 90    | 65           | 85         | Nenhuma    | Haki       | 1         |
| Usopp  | 50    | 60           | 60         | Nenhuma    | Armadilha  | 0         |

---

### 3. Explorar os dados
- Verificação do tamanho (`df.shape`) e tipos (`df.info()`)  
- Estatísticas descritivas (`df.describe()`)  
- Frequência de tipos de fruta e habilidades (`value_counts()`)  
- Distribuição da variável alvo (`df['Sobrevive'].value_counts()`)  

---

### 4. Tratamento dos dados
- Separação de variáveis numéricas (`Forca, Inteligencia, Velocidade`) e categóricas (`Tipo_Fruta, Habilidade`).  
- Pré-processamento:
  - Numéricos → imputação com mediana + padronização (`StandardScaler`).  
  - Categóricos → imputação com valor mais frequente + one-hot encoding.  
- Implementado com `ColumnTransformer` e `Pipeline`.

---

### 5. Separar Base em Arrays
```python
X = df[["Forca", "Inteligencia", "Velocidade", "Tipo_Fruta", "Habilidade"]]
y = df["Sobrevive"]
```

---

### 6. Divisão Treino/Teste
-Separação em treino (70%) e teste (30%) estratificando pelo target:
from sklearn.model_selection import train_test_split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
```
---

### 7. Definição de Modelos
- Foram treinados 3 modelos do scikit-learn:
  - Logistic Regression (LogisticRegression(max_iter=500))
  - Random Forest (RandomForestClassifier(n_estimators=200, max_depth=5))
  - KNN (KNeighborsClassifier(n_neighbors=3))
-Cada modelo foi integrado a um pipeline com pré-processamento.

---

### 8. Definição de Modelos
Foram treinados 3 modelos do scikit-learn:

- **Logistic Regression**
- **Random Forest**
- **KNN**

Cada modelo foi integrado a um pipeline com pré-processamento.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}
```

---

### 9. Validação dos Modelos
Métricas utilizadas: **Acurácia, Precision, Recall e F1-score**.  
Foi feita **validação cruzada com 5 folds estratificados (cross_val_score)**.  
Os resultados de treino/teste e da validação cruzada foram comparados.

```python
from sklearn.model_selection import cross_val_score

for nome, modelo in modelos.items():
    scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="accuracy")
    print(f"{nome} - Acurácia Média (CV): {scores.mean():.2f}")
```

---

### 10. Resultados e Uso
- O **RandomForestClassifier** apresentou o melhor desempenho (~83% de acurácia média).  
- O modelo final foi salvo como **onepiece_survival_model.pkl**.  

```python
import joblib

best_model = RandomForestClassifier(n_estimators=200, max_depth=5)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "onepiece_survival_model.pkl")
```

Como usar o modelo salvo:

```python
# Carregar modelo treinado
modelo_carregado = joblib.load("onepiece_survival_model.pkl")

# Fazer previsões em novos personagens
previsoes = modelo_carregado.predict(X_test)
```
---

###  Resumo
Este projeto mostra o ciclo completo de **Machine Learning supervisionado**: desde a análise e limpeza dos dados até o treinamento, avaliação e salvamento do modelo final.  

O dataset fictício foi inspirado no universo de **One Piece**, com atributos como força, inteligência, velocidade, tipo de fruta e habilidades especiais.  
O objetivo foi prever se um personagem **sobrevive** ou não em batalhas, com base nessas características.  

- O problema foi tratado como uma **classificação binária** (0 = não sobrevive, 1 = sobrevive).  
- Foram utilizados diferentes modelos de classificação: **Logistic Regression, KNN e Random Forest**.  
- O modelo que apresentou melhor desempenho foi o **Random Forest**, alcançando cerca de **83% de acurácia média**.  
- O modelo final foi salvo em `onepiece_survival_model.pkl` para uso posterior.  

Este projeto fictício ilustra como aplicar **técnicas de pré-processamento, treino, validação e salvamento de modelos** em um caso divertido baseado em One Piece. ⚔️🏴‍☠️



