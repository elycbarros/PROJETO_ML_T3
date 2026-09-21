from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
m = np.load(RAIZ / "dados_derivados/matrizes_preparadas.npz")
ytr, yte = m["ytr"], m["yte"]
modelos = {"knn_k3": (KNeighborsClassifier(n_neighbors=3), m["Xtr_knn"], m["Xte_knn"]), "arvore_depth7": (DecisionTreeClassifier(max_depth=7, random_state=42), m["Xtr_arvore"], m["Xte_arvore"])}
saida = RAIZ / "resultados/avaliacao_final"; saida.mkdir(parents=True, exist_ok=True); resumo=[]
for nome, (modelo, Xtr, Xte) in modelos.items():
    modelo.fit(Xtr,ytr); pred=modelo.predict(Xte); cm=confusion_matrix(yte,pred)
    (saida / f"{nome}_classification_report.txt").write_text(classification_report(yte,pred,target_names=["Em dia (0)","Inadimplente (1)"],digits=4))
    pd.DataFrame(cm,index=["Real 0","Real 1"],columns=["Previsto 0","Previsto 1"]).to_csv(saida / f"{nome}_matriz_confusao.csv")
    fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, ax=ax)
    ax.set_xticks([0,1], ["Previsto 0", "Previsto 1"]); ax.set_yticks([0,1], ["Real 0", "Real 1"]); ax.set_title(f"Matriz de confusão — {nome}")
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i,j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(saida / f"{nome}_matriz_confusao.svg"); plt.close(fig)
    tn,fp,fn,tp=cm.ravel(); resumo.append({"modelo":nome,"acuracia":accuracy_score(yte,pred),"precisao_1":precision_score(yte,pred,zero_division=0),"recall_1":recall_score(yte,pred,zero_division=0),"f1_1":f1_score(yte,pred,zero_division=0),"falsos_positivos":int(fp),"falsos_negativos":int(fn)})
df=pd.DataFrame(resumo); df.to_csv(saida / "comparacao_final.csv",index=False)
knn=df[df.modelo=="knn_k3"].iloc[0]; arv=df[df.modelo=="arvore_depth7"].iloc[0]
relatorio=f"""# Avaliação final e veredito de negócio

Foram comparados KNN (`K=3`) e Árvore de Decisão (`max_depth=7`) no teste original, sem reamostragem.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K=3 | {knn.acuracia:.3f} | {knn.precisao_1:.3f} | {knn.recall_1:.3f} | {knn.f1_1:.3f} | {int(knn.falsos_positivos)} | {int(knn.falsos_negativos)} |
| Árvore depth=7 | {arv.acuracia:.3f} | {arv.precisao_1:.3f} | {arv.recall_1:.3f} | {arv.f1_1:.3f} | {int(arv.falsos_positivos)} | {int(arv.falsos_negativos)} |

Um falso positivo trata como inadimplente quem pagaria em dia, podendo gerar recusa ou condição pior para um bom cliente. Um falso negativo libera crédito a quem inadimplirá. Assumo que o segundo erro tem maior perda financeira direta, embora custos reais não tenham sido fornecidos.

## Veredito

Recomendo a **Árvore de Decisão com `max_depth=7` para um piloto controlado**. Ela teve F1 e precisão superiores aos do KNN no teste e manteve gap de generalização menor que a árvore sem limite. Antes de produção, o banco deve calibrar o limiar com custos reais, validar em uma amostra temporal e acompanhar desempenho por segmento.

Relatórios e matrizes estão em `resultados/avaliacao_final/`. Os CSVs originais foram apenas lidos e tiveram hashes conferidos.
"""
(RAIZ / "documentacao/avaliacao_final.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest()==esperado
print("Avaliação final concluída; recomendação: árvore com profundidade 7.")
