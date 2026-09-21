"""Separa dados e prepara matrizes sem aprender parâmetros no teste."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
dados = pd.read_csv(RAIZ / "dados_derivados/credito_com_feature.csv")
alvo = dados.pop("loan_status")
dados = dados.drop(columns=["loan_percent_income"])
X_treino, X_teste, y_treino, y_teste = train_test_split(dados, alvo, test_size=0.20, stratify=alvo, random_state=42)
numericas = X_treino.select_dtypes(include="number").columns.tolist()
categoricas = X_treino.select_dtypes(exclude="number").columns.tolist()
prep_knn = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numericas), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas)])
prep_arvore = ColumnTransformer([("num", SimpleImputer(strategy="median"), numericas), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categoricas)])
Xtr_knn, Xte_knn = prep_knn.fit_transform(X_treino), prep_knn.transform(X_teste)
Xtr_arvore, Xte_arvore = prep_arvore.fit_transform(X_treino), prep_arvore.transform(X_teste)
rng = np.random.default_rng(42); indices = np.arange(len(y_treino)); classes, contagens = np.unique(y_treino.to_numpy(), return_counts=True); alvo_contagem = contagens.max(); selecionados = []
for classe in classes:
    base = indices[y_treino.to_numpy() == classe]; selecionados.append(rng.choice(base, size=alvo_contagem, replace=True))
selecionados = np.concatenate(selecionados); rng.shuffle(selecionados)
Xtr_knn_bal, Xtr_arvore_bal = Xtr_knn[selecionados], Xtr_arvore[selecionados]; ytr_bal = y_treino.to_numpy()[selecionados]
out = RAIZ / "resultados"; out.mkdir(exist_ok=True)
np.savez_compressed(RAIZ / "dados_derivados/matrizes_preparadas.npz", Xtr_knn=Xtr_knn_bal, Xte_knn=Xte_knn, Xtr_arvore=Xtr_arvore_bal, Xte_arvore=Xte_arvore, ytr=ytr_bal, yte=y_teste.to_numpy())
resumo = {"random_state":42,"test_size":0.20,"features_originais":int(dados.shape[1]),"treino_antes_balanceamento":int(len(y_treino)),"teste":int(len(y_teste)),"treino_classes_antes":y_treino.value_counts().sort_index().to_dict(),"treino_classes_depois":pd.Series(ytr_bal).value_counts().sort_index().to_dict(),"teste_classes_mantidas":y_teste.value_counts().sort_index().to_dict(),"dimensao_knn":[int(Xtr_knn_bal.shape[1]),int(Xte_knn.shape[1])],"dimensao_arvore":[int(Xtr_arvore_bal.shape[1]),int(Xte_arvore.shape[1])]}
(out / "separacao_preparacao.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n")
relatorio = f"""# Separação e preparação sem vazamento

Foi aplicado `train_test_split(test_size=0.20, stratify=y, random_state=42)`. O treino ficou com {len(y_treino):,} linhas e o teste com {len(y_teste):,}. O teste preservou as classes: {resumo['teste_classes_mantidas']}.

Imputadores, codificadores e escalonadores foram ajustados somente no treino; o teste usa apenas `transform`. O KNN recebeu mediana para numéricas, moda e one-hot para categóricas e `StandardScaler` nas numéricas. A árvore recebeu as mesmas imputações e codificação, sem escalonamento.

O oversampling aleatório foi aplicado somente ao treino, depois da transformação. As classes passaram de {resumo['treino_classes_antes']} para {resumo['treino_classes_depois']}; o teste não foi reamostrado.

`loan_percent_income` foi retirado deste primeiro conjunto de preditores por representar informação quase duplicada da feature definido. Essa escolha poderá ser revisitada em análise de sensibilidade.
"""
(RAIZ / "documentacao" / "separacao_preparacao.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print("Separação e preparação concluídas; CSVs originais preservados.")
