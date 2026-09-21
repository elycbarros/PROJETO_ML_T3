"""Gera gráficos complementares de otimização, overfitting, feature importance e simulação financeira."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / 'documentacao/integridade_originais.json').read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado

d = pd.read_csv(RAIZ / 'dados_derivados/credito_com_feature.csv').copy()
d.loc[d.person_emp_length >= d.person_age, 'person_emp_length'] = np.nan
y = d.pop('loan_status')
d = d.drop(columns=['loan_percent_income'])
tr, te = train_test_split(np.arange(len(d)), test_size=0.2, stratify=y, random_state=42)
num = d.select_dtypes(include='number').columns.tolist()
cat = d.select_dtypes(exclude='number').columns.tolist()

def prep_fit(X):
    return ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median'))]), num),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), ('oh', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ])

def balance(X, yv, rng):
    yv = np.asarray(yv); maj = max(np.bincount(yv)); ids = []
    for c in [0, 1]:
        base = np.flatnonzero(yv == c)
        ids.extend(base if len(base) == maj else list(base) + list(rng.choice(base, maj - len(base), replace=True)))
    ids = np.asarray(ids); rng.shuffle(ids)
    return X[ids], yv[ids]

saida = RAIZ / 'resultados/avaliacao_final'
saida.mkdir(parents=True, exist_ok=True)

# 1. Gráfico de Overfitting da Árvore de Decisão
print("Gerando curva de overfitting da Árvore...")
profundidades = [3, 5, 7, None]
labels = ['3', '5', '7', 'Ilimitada (None)']
tr_means, val_means, val_stds = [], [], []

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for depth in profundidades:
    t_scores, v_scores = [], []
    for a, b in cv.split(d.iloc[tr], y.iloc[tr]):
        rawtr, rawv = d.iloc[tr[a]], d.iloc[tr[b]]
        yt, yv = y.iloc[tr[a]].to_numpy(), y.iloc[tr[b]].to_numpy()
        rng = np.random.default_rng(42)
        pf = prep_fit(rawtr)
        X = pf.fit_transform(rawtr); V = pf.transform(rawv)
        X_bal, yt_bal = balance(X, yt, rng)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_bal, yt_bal)
        t_scores.append(f1_score(yt, model.predict(X), zero_division=0))
        v_scores.append(f1_score(yv, model.predict(V), zero_division=0))
    tr_means.append(np.mean(t_scores))
    val_means.append(np.mean(v_scores))
    val_stds.append(np.std(v_scores))

fig, ax = plt.subplots(figsize=(7, 4.5))
x_pos = np.arange(len(labels))
ax.plot(x_pos, tr_means, marker='o', linewidth=2.2, color='#e74c3c', label='F1 Treino (com memorização)')
ax.errorbar(x_pos, val_means, yerr=val_stds, marker='s', linewidth=2.2, color='#2980b9', capsize=4, label='F1 Validação Cruzada (5-fold)')
ax.axvline(x=2, color='#27ae60', linestyle='--', alpha=0.8, label='Ponto Ótimo Selecionado (depth=7)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_xlabel('Profundidade Máxima (max_depth)', fontsize=11)
ax.set_ylabel('F1-Score (Classe Inadimplente)', fontsize=11)
ax.set_title('Diagnóstico de Overfitting — Árvore de Decisão', fontsize=12, fontweight='bold')
ax.set_ylim(0.65, 1.05)
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True)
fig.tight_layout()
fig.savefig(saida / 'curva_overfitting_arvore.svg')
plt.close(fig)

# 2. Gráfico de Validação do KNN
print("Gerando curva de validação do KNN...")
knn_df = pd.read_csv(RAIZ / 'resultados/validacao_corrigida.csv')
knn_data = knn_df[knn_df['model'] == 'KNN']
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(knn_data['param'].astype(int), knn_data['cv_f1'], yerr=knn_data['cv_std'], marker='o', linewidth=2, color='#8e44ad', capsize=4, label='F1 Validação Cruzada')
ax.scatter([3], [knn_data.loc[knn_data['param'] == 3, 'cv_f1'].values[0]], color='#27ae60', s=100, zorder=5, label='K Selecionado (K=3)')
ax.set_xlabel('Número de Vizinhos (K)', fontsize=11)
ax.set_ylabel('F1-Score Médio (Classe Inadimplente)', fontsize=11)
ax.set_title('Otimização de Hiperparâmetro — KNN', fontsize=12, fontweight='bold')
ax.set_xticks([3, 5, 7, 9])
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig(saida / 'curva_validacao_knn.svg')
plt.close(fig)

# 3. Feature Importance da Árvore de Decisão (depth=7)
print("Gerando Feature Importance da Árvore...")
pf_full = prep_fit(d.iloc[tr])
Xtr_full = pf_full.fit_transform(d.iloc[tr])
ytr_full = y.iloc[tr].to_numpy()
rng = np.random.default_rng(42)
Xtr_b, ytr_b = balance(Xtr_full, ytr_full, rng)
tree_opt = DecisionTreeClassifier(max_depth=7, random_state=42)
tree_opt.fit(Xtr_b, ytr_b)

cat_features = pf_full.named_transformers_['cat'].named_steps['oh'].get_feature_names_out(cat)
feature_names = num + list(cat_features)
importances = pd.Series(tree_opt.feature_importances_, index=feature_names).sort_values(ascending=True)
importances_df = importances.reset_index()
importances_df.columns = ['feature', 'importance']
importances_df.to_csv(saida / 'feature_importance_arvore.csv', index=False)

top_imp = importances.tail(10)
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#27ae60' if f == 'comprometimento_renda' else '#3498db' for f in top_imp.index]
ax.barh(top_imp.index, top_imp.values, color=colors, edgecolor='none', height=0.65)
for i, v in enumerate(top_imp.values):
    ax.text(v + 0.005, i, f"{v:.1%}", va='center', fontsize=9.5, fontweight='bold' if top_imp.index[i] == 'comprometimento_renda' else 'normal')
ax.set_xlabel('Importância Relativa (Gini Importance)', fontsize=11)
ax.set_title('Top 10 Variáveis Mais Determinantes — Árvore (depth=7)', fontsize=12, fontweight='bold')
ax.set_xlim(0, max(top_imp.values) * 1.15)
ax.grid(True, axis='x', linestyle=':', alpha=0.6)
fig.tight_layout()
fig.savefig(saida / 'feature_importance_arvore.svg')
plt.close(fig)

# 4. Simulação Financeira Comparativa (P&L do Banco)
print("Gerando simulação financeira de P&L...")
# Hipótese ilustrativa para o negócio:
# FN (inadimplente não detectado): Perda média de principal = R$ 5.000
# FP (bom pagador recusado): Custo de oportunidade de juros perdidos = R$ 1.000
custo_fn = 5000
custo_fp = 1000

# KNN: 756 FP, 379 FN
# Árvore: 421 FP, 331 FN
custos = pd.DataFrame([
    {
        'Modelo': 'KNN (K=3)',
        'Falsos Positivos': 756,
        'Falsos Negativos': 379,
        'Custo FP (R$)': 756 * custo_fp,
        'Custo FN (R$)': 379 * custo_fn,
        'Custo Total (R$)': (756 * custo_fp) + (379 * custo_fn)
    },
    {
        'Modelo': 'Árvore (depth=7)',
        'Falsos Positivos': 421,
        'Falsos Negativos': 331,
        'Custo FP (R$)': 421 * custo_fp,
        'Custo FN (R$)': 331 * custo_fn,
        'Custo Total (R$)': (421 * custo_fp) + (331 * custo_fn)
    }
])
custos.to_csv(saida / 'simulacao_financeira_custos.csv', index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
bar_width = 0.45
x_idx = np.arange(2)
ax.bar(x_idx, custos['Custo FN (R$)'] / 1e3, bar_width, label='Custo de Falsos Negativos (Perda de Empréstimo)', color='#e74c3c')
ax.bar(x_idx, custos['Custo FP (R$)'] / 1e3, bar_width, bottom=custos['Custo FN (R$)'] / 1e3, label='Custo de Falsos Positivos (Lucro Cessante)', color='#f39c12')

for i in range(2):
    tot = custos['Custo Total (R$)'].iloc[i] / 1e3
    ax.text(i, tot + 40, f"Total: R$ {tot:,.0f} mil".replace(',', '.'), ha='center', fontweight='bold', fontsize=10.5)

economia = (custos['Custo Total (R$)'].iloc[0] - custos['Custo Total (R$)'].iloc[1]) / 1e3
ax.set_xticks(x_idx)
ax.set_xticklabels(custos['Modelo'], fontsize=11, fontweight='bold')
ax.set_ylabel('Perda Financeira Estimada (Milhares de R$)', fontsize=11)
ax.set_title(f'Impacto Financeiro Estimado no Teste — Economia de R$ {economia:,.0f} mil'.replace(',', '.'), fontsize=12, fontweight='bold')
ax.set_ylim(0, 3200)
ax.grid(True, axis='y', linestyle=':', alpha=0.6)
ax.legend(loc='upper right', frameon=True)
fig.tight_layout()
fig.savefig(saida / 'simulacao_financeira_custos.svg')
plt.close(fig)

for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print("Gráficos complementares gerados com sucesso!")
