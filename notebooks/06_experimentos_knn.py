"""Compara configurações do KNN e registra sinais de overfitting."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
m = np.load(RAIZ / "dados_derivados/matrizes_preparadas.npz")
Xtr, Xte, ytr, yte = m["Xtr_knn"], m["Xte_knn"], m["ytr"], m["yte"]

def metricas(modelo, X, y):
    pred = modelo.predict(X)
    return {"acuracia": accuracy_score(y, pred), "precisao_1": precision_score(y, pred, zero_division=0), "recall_1": recall_score(y, pred, zero_division=0), "f1_1": f1_score(y, pred, zero_division=0)}

linhas = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k in [3, 5, 7, 9]:
    modelo = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    cv_f1 = cross_val_score(modelo, Xtr, ytr, cv=cv, scoring="f1", n_jobs=-1)
    modelo.fit(Xtr, ytr)
    treino = metricas(modelo, Xtr, ytr); teste = metricas(modelo, Xte, yte)
    linhas.append({"modelo":"KNN", "k":k, "cv_f1_media":cv_f1.mean(), "cv_f1_dp":cv_f1.std(), **{"treino_"+a:b for a,b in treino.items()}, **{"teste_"+a:b for a,b in teste.items()}, "gap_f1":treino["f1_1"]-teste["f1_1"]})
resultados = pd.DataFrame(linhas)
resultados.to_csv(RAIZ / "resultados/knn_experimentos.csv", index=False)
melhor_k = int(resultados.sort_values(["cv_f1_media", "teste_f1_1"], ascending=False).iloc[0].k)
relatorio = f"""# Experimentos do KNN

Foram avaliados quatro valores de `n_neighbors`: 3, 5, 7 e 9. O treino foi balanceado; o teste manteve a distribuição original. A seleção foi orientada pela média do F1 da classe 1 em validação estratificada de cinco partes no treino. O teste ficou reservado para comparação final.

O melhor candidato pela validação interna foi **K = {melhor_k}**. A tabela completa está em `resultados/knn_experimentos.csv`.

O `gap_f1` é a diferença entre o F1 da classe 1 no treino e no teste. Gap alto indica ajuste maior ao treino do que aos dados não vistos. A decisão final também considerará recall, precisão, F1 e matriz de confusão.

Os quatro modelos usaram as mesmas transformações; somente `n_neighbors` mudou. Os CSVs originais foram apenas lidos e tiveram os hashes conferidos.
"""
(RAIZ / "documentacao/experimentos_knn.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print(f"Experimentos KNN concluídos; candidato por validação interna: K={melhor_k}.")
