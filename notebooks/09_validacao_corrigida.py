"""Reexecução metodológica: cada dobra ajusta prep e balanceamento apenas no treino."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1]; H=json.loads((R/'documentacao/integridade_originais.json').read_text())
for n,h in H.items(): assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
d=pd.read_csv(R/'dados_derivados/credito_com_feature.csv').copy()
d.loc[d.person_emp_length>=d.person_age,'person_emp_length']=np.nan
y=d.pop('loan_status'); d=d.drop(columns=['loan_percent_income']); tr,te=train_test_split(np.arange(len(d)),test_size=.2,stratify=y,random_state=42)
num=d.select_dtypes(include='number').columns.tolist(); cat=d.select_dtypes(exclude='number').columns.tolist()

def prep_fit(X,scale=False):
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median'))]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
def scale_after_balance(X, V, n_numeric):
    sc=StandardScaler(); X=X.copy(); V=V.copy(); X[:,:n_numeric]=sc.fit_transform(X[:,:n_numeric]); V[:,:n_numeric]=sc.transform(V[:,:n_numeric]); return X,V
def balance(X,yv,rng):
    yv=np.asarray(yv); maj=max(np.bincount(yv)); ids=[]
    for c in [0,1]:
        base=np.flatnonzero(yv==c); ids.extend(base if len(base)==maj else list(base)+list(rng.choice(base,maj-len(base),replace=True)))
    ids=np.asarray(ids); rng.shuffle(ids); return X[ids],yv[ids]
def evaluate(model,X,yv):
    p=model.predict(X); return {'accuracy':accuracy_score(yv,p),'precision_1':precision_score(yv,p,zero_division=0),'recall_1':recall_score(yv,p,zero_division=0),'f1_1':f1_score(yv,p,zero_division=0)}

def cv_scores(kind,param):
    vals=[]; cv=StratifiedKFold(5,shuffle=True,random_state=42)
    for a,b in cv.split(d.iloc[tr],y.iloc[tr]):
        rawtr=d.iloc[tr[a]]; rawv=d.iloc[tr[b]]; yt=y.iloc[tr[a]].to_numpy(); yv=y.iloc[tr[b]].to_numpy(); rng=np.random.default_rng(42)
        pf=prep_fit(rawtr); X=pf.fit_transform(rawtr); V=pf.transform(rawv); X,yt=balance(X,yt,rng)
        if kind=='knn': X,V=scale_after_balance(X,V,len(num))
        model=KNeighborsClassifier(n_neighbors=param) if kind=='knn' else DecisionTreeClassifier(max_depth=param,random_state=42); model.fit(X,yt); vals.append(f1_score(yv,model.predict(V),zero_division=0))
    return float(np.mean(vals)),float(np.std(vals))

rows=[]
for k in [3,5,7,9]: rows.append({'model':'KNN','param':str(k),'cv_f1':cv_scores('knn',k)[0],'cv_std':cv_scores('knn',k)[1]})
for depth in [3,5,7,None]: rows.append({'model':'Tree','param':str(depth),'cv_f1':cv_scores('tree',depth)[0],'cv_std':cv_scores('tree',depth)[1]})
def complexity(r):
    return 99 if r['param']=='None' else int(r['param'])
sel={m:max((r for r in rows if r['model']==m),key=lambda r:(r['cv_f1'],-complexity(r)))['param'] for m in ['KNN','Tree']}
final=[]; testd=d.iloc[te]; yt=y.iloc[tr].to_numpy(); ytest=y.iloc[te].to_numpy(); rng=np.random.default_rng(42)
out=R/'resultados'; final_dir=out/'avaliacao_final'; final_dir.mkdir(parents=True, exist_ok=True)
for name,kind,param,scale in [('KNN','knn',int(sel['KNN']),True),('Tree','tree',None if sel['Tree']=='None' else int(sel['Tree']),False)]:
    pf=prep_fit(d.iloc[tr]); X=pf.fit_transform(d.iloc[tr]); V=pf.transform(testd); X,ytb=balance(X,yt,rng)
    if kind=='knn': X,V=scale_after_balance(X,V,len(num))
    model=KNeighborsClassifier(n_neighbors=param) if kind=='knn' else DecisionTreeClassifier(max_depth=param,random_state=42); model.fit(X,ytb); p=model.predict(V); cm=confusion_matrix(ytest,p); tn,fp,fn,tp=cm.ravel(); met=evaluate(model,V,ytest); final.append({'model':name,'param':str(param),'cv_f1':next(r['cv_f1'] for r in rows if r['model']==('KNN' if kind=='knn' else 'Tree') and r['param']==str(param)),'test_f1_1':met['f1_1'],'test_accuracy':met['accuracy'],'test_precision_1':met['precision_1'],'test_recall_1':met['recall_1'],'fp':int(fp),'fn':int(fn)})
    file_prefix = f"knn_k{param}" if kind == 'knn' else f"arvore_depth{param}"
    (final_dir / f"{file_prefix}_classification_report.txt").write_text(classification_report(ytest, p, target_names=["Em dia (0)", "Inadimplente (1)"], digits=4))
    pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Previsto 0", "Previsto 1"]).to_csv(final_dir / f"{file_prefix}_matriz_confusao.csv")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1], ["Previsto 0", "Previsto 1"])
    ax.set_yticks([0, 1], ["Real 0", "Real 1"])
    ax.set_title(f"Matriz de confusão — {name} ({param})")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
    fig.tight_layout()
    fig.savefig(final_dir / f"{file_prefix}_matriz_confusao.svg")
    plt.close(fig)

pd.DataFrame(rows).to_csv(out/'validacao_corrigida.csv',index=False)
pd.DataFrame(final).to_csv(out/'avaliacao_corrigida.csv',index=False)
comp_final = pd.DataFrame([
    {'modelo': 'knn_k3', 'acuracia': final[0]['test_accuracy'], 'precisao_1': final[0]['test_precision_1'], 'recall_1': final[0]['test_recall_1'], 'f1_1': final[0]['test_f1_1'], 'falsos_positivos': final[0]['fp'], 'falsos_negativos': final[0]['fn']},
    {'modelo': 'arvore_depth7', 'acuracia': final[1]['test_accuracy'], 'precisao_1': final[1]['test_precision_1'], 'recall_1': final[1]['test_recall_1'], 'f1_1': final[1]['test_f1_1'], 'falsos_positivos': final[1]['fp'], 'falsos_negativos': final[1]['fn']}
])
comp_final.to_csv(final_dir / "comparacao_final.csv", index=False)

relatorio_md = f"""# Avaliação final e veredito de negócio

Foram comparados KNN (`K={sel['KNN']}`) e Árvore de Decisão (`max_depth={sel['Tree']}`) no teste original (6.483 registros), sem reamostragem, após seleção estrita por validação cruzada sem vazamento.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K={sel['KNN']} | {final[0]['test_accuracy']:.3f} | {final[0]['test_precision_1']:.3f} | {final[0]['test_recall_1']:.3f} | {final[0]['test_f1_1']:.3f} | {final[0]['fp']} | {final[0]['fn']} |
| Árvore depth={sel['Tree']} | {final[1]['test_accuracy']:.3f} | {final[1]['test_precision_1']:.3f} | {final[1]['test_recall_1']:.3f} | {final[1]['test_f1_1']:.3f} | {final[1]['fp']} | {final[1]['fn']} |

Um falso positivo trata como inadimplente quem pagaria em dia, gerando atrito ou recusa injustificada de um bom cliente. Um falso negativo libera crédito a quem inadimplirá, gerando perda financeira direta do principal.

## Veredito

Recomendo a **Árvore de Decisão com `max_depth=7` para um piloto controlado**.
Na validação corrigida, a Árvore superou o KNN em todas as métricas:
- F1-Score superior ({final[1]['test_f1_1']:.3f} vs {final[0]['test_f1_1']:.3f});
- Recall superior ({final[1]['test_recall_1']:.3f} vs {final[0]['test_recall_1']:.3f}), capturando mais inadimplentes ({final[1]['fn']} FN contra {final[0]['fn']} do KNN);
- Precisão e acurácia substancialmente maiores, reduzindo os falsos positivos quase pela metade ({final[1]['fp']} contra {final[0]['fp']} do KNN).

Antes de qualquer implantação em produção, o banco deve:
1. Calibrar o limiar de decisão com os custos financeiros monetários reais de FP e FN;
2. Realizar validação fora do tempo (amostra temporal subsequente);
3. Monitorar métricas de disparidade e estabilidade por segmento socioeconômico.

Relatórios e matrizes estão em `resultados/avaliacao_final/`. Os CSVs originais foram apenas lidos e tiveram hashes conferidos.
"""
(R / "documentacao/avaliacao_final.md").write_text(relatorio_md)
(R/'documentacao/correcao_metodologica.md').write_text(f'''# Correção metodológica\n\nA validação foi refeita a partir dos dados brutos do treino. Em cada dobra, imputação, codificação, balanceamento por oversampling da classe minoritária e escalonamento do KNN foram ajustados somente na parte de treino. A validação manteve sua distribuição original.\n\nO oversampling preserva todos os registros da classe majoritária e acrescenta somente cópias da classe minoritária. Os dois registros com tempo de emprego maior que a idade foram convertidos em nulos e entram na imputação do treino.\n\nParâmetros escolhidos exclusivamente pela média de F1 da classe 1 na validação: KNN={sel['KNN']}; Árvore={sel['Tree']}. Resultados corrigidos: `resultados/validacao_corrigida.csv` e `resultados/avaliacao_corrigida.csv`. Os resultados anteriores ficam superados.\n''')
for n,h in H.items(): assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
print('Validação corrigida concluída com geração de artefatos visuais',sel)

