"""Gera os gráficos da EDA sem modificar os CSVs de entrada."""
from pathlib import Path
import hashlib, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
saida = RAIZ / "resultados" / "graficos_eda"
saida.mkdir(parents=True, exist_ok=True)
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado

dados = pd.read_csv(RAIZ / "credit_risk_dataset.csv")
classes = dados["loan_status"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Em dia (0)", "Inadimplente (1)"], classes.values, color=["#4472C4", "#C00000"])
ax.set_title("Distribuição da variável-alvo")
ax.set_ylabel("Quantidade de contratos")
for bar, n in zip(bars, classes.values):
    ax.text(bar.get_x() + bar.get_width()/2, n, f"{n:,}\n{n/len(dados):.1%}", ha="center", va="bottom")
fig.tight_layout(); fig.savefig(saida / "01_distribuicao_alvo.svg"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for status, cor in [(0, "#4472C4"), (1, "#C00000")]:
    ax.hist(dados.loc[dados.loan_status == status, "person_income"], bins=45, density=True,
            histtype="step", linewidth=1.8, color=cor, label=f"Status {status}")
ax.legend()
ax.set_title("Distribuição da renda anual por status do empréstimo")
ax.set_xlabel("Renda anual"); ax.set_ylabel("Densidade")
fig.tight_layout(); fig.savefig(saida / "02_histograma_renda_status.svg"); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
dados.boxplot(column="loan_percent_income", by="loan_status", ax=ax)
fig.suptitle("")
ax.set_title("Comprometimento percentual da renda por status")
ax.set_xlabel("Status (0 = em dia; 1 = inadimplente)"); ax.set_ylabel("Percentual da renda")
fig.tight_layout(); fig.savefig(saida / "03_boxplot_comprometimento.svg"); plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 8))
corr = dados.select_dtypes(include="number").corr(method="pearson")
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(np.arange(len(corr.columns)), labels=corr.columns, rotation=90)
ax.set_yticks(np.arange(len(corr.columns)), labels=corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=.8)
ax.set_title("Correlação de Pearson entre variáveis numéricas")
fig.tight_layout(); fig.savefig(saida / "04_correlacao_pearson.svg"); plt.close(fig)

resumo = {
    "percentual_inadimplencia": round(dados.loan_status.mean() * 100, 2),
    "renda_mediana_em_dia": round(dados.loc[dados.loan_status == 0, "person_income"].median(), 2),
    "renda_mediana_inadimplente": round(dados.loc[dados.loan_status == 1, "person_income"].median(), 2),
    "comprometimento_mediano_em_dia": round(dados.loc[dados.loan_status == 0, "loan_percent_income"].median(), 3),
    "comprometimento_mediano_inadimplente": round(dados.loc[dados.loan_status == 1, "loan_percent_income"].median(), 3),
    "correlacao_status_comprometimento": round(dados[["loan_status", "loan_percent_income"]].corr().iloc[0, 1], 3),
    "correlacao_status_taxa_juros": round(dados[["loan_status", "loan_int_rate"]].corr().iloc[0, 1], 3),
}
(saida / "resumo_numerico.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n")
relatorio = f"""# EDA: gráficos e interpretação

## Distribuição do alvo

A inadimplência representa {resumo['percentual_inadimplencia']:.2f}% dos registros. Há diferença relevante entre as classes; o split deverá preservar a proporção e a avaliação usará recall, precisão e F1 além da acurácia. O balanceamento, quando aplicado, ficará restrito ao treino.

## Renda anual

A mediana da renda é {resumo['renda_mediana_em_dia']:,.0f} para contratos em dia e {resumo['renda_mediana_inadimplente']:,.0f} para contratos inadimplentes. O histograma deve ser lido com cautela porque a renda é assimétrica e possui valores muito altos; essa variável será examinada junto dos outliers.

## Comprometimento da renda

As medianas de `loan_percent_income` são {resumo['comprometimento_mediano_em_dia']:.3f} para a classe 0 e {resumo['comprometimento_mediano_inadimplente']:.3f} para a classe 1. A correlação de Pearson com o alvo é {resumo['correlacao_status_comprometimento']:.3f}; isso é associação descritiva, não causalidade.

## Correlações e próximos passos

A correlação entre `loan_status` e `loan_int_rate` é {resumo['correlacao_status_taxa_juros']:.3f}. Correlação não será justificativa única para excluir colunas. Próximas decisões: investigar caudas de renda, idade e tempo de emprego; comparar `comprometimento_renda` com `loan_percent_income`; e definir imputação após examinar outliers.

As figuras ficam em `resultados/graficos_eda/`. Os CSVs foram apenas lidos e tiveram seus hashes conferidos antes e depois da execução.
"""
(RAIZ / "documentacao" / "eda_graficos.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print("EDA gerada; integridade dos CSVs confirmada.")
