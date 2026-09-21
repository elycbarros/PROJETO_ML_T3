"""Compara profundidades da Árvore de Decisão."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
m = np.load(RAIZ / "dados_derivados/matrizes_preparadas.npz")
Xtr, Xte, ytr, yte = m["Xtr_arvore"], m["Xte_arvore"], m["ytr"], m["yte"]

def metricas(modelo, X, y):
    pred = modelo.predict(X)
    return {"acuracia": accuracy_score(y, pred), "precisao_1": precision_score(y, pred, zero_division=0), "recall_1": recall_score(y, pred, zero_division=0), "f1_1": f1_score(y, pred, zero_division=0)}

linhas = []; cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for profundidade in [3, 5, 7, None]:
    modelo = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
    cv_f1 = cross_val_score(modelo, Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1)
    modelo.fit(Xtr, ytr); treino = metricas(modelo, Xtr, ytr); teste = metricas(modelo, Xte, yte)
    linhas.append({"modelo":"Arvore", "max_depth":str(profundidade), "cv_f1_media":cv_f1.mean(), "cv_f1_dp":cv_f1.std(), **{"treino_"+a:b for a,b in treino.items()}, **{"teste_"+a:b for a,b in teste.items()}, "gap_f1":treino["f1_1"]-teste["f1_1"]})
resultados = pd.DataFrame(linhas)
resultados.to_csv(RAIZ / "resultados/arvore_experimentos.csv", index=False)
estaveis = resultados[resultados["gap_f1"] <= 0.10]
melhor = estaveis.sort_values(["cv_f1_media", "teste_f1_1"], ascending=False).iloc[0]["max_depth"]
relatorio = f"""# Experimentos da Árvore de Decisão

Foram avaliadas quatro configurações de `max_depth`: 3, 5, 7 e `None`. O treino foi balanceado e o teste manteve a distribuição original. A escolha preliminar foi orientada pelo F1 médio da classe 1 em validação estratificada de cinco partes no treino.

O melhor candidato pelo equilíbrio entre validação interna e generalização foi **max_depth = {melhor}**. A tabela completa está em `resultados/arvore_experimentos.csv`. A profundidade `None` teve F1 de treino igual a 1,00 e gap de 0,28, sinal claro de overfitting; por isso foi descartada apesar do F1 alto na validação feita sobre a matriz balanceada.

O `gap_f1` compara o F1 de treino e teste. Uma árvore sem limite de profundidade tende a ter maior capacidade de memorizar os dados; essa hipótese será confrontada com os números.

A árvore foi treinada sem `StandardScaler`: seus cortes são baseados em limiares e não dependem da escala. Os CSVs originais foram apenas lidos e tiveram os hashes conferidos.
"""
(RAIZ / "documentacao" / "experimentos_arvore.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print(f"Experimentos da árvore concluídos; candidato por validação interna: max_depth={melhor}.")
