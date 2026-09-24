"""Gera os gráficos da EDA e a Seção 1 de documentacao/01_eda_e_preparacao.md.

Chamado por notebooks/03_executar_pipeline.py (que roda este arquivo primeiro e depois
pipeline_credito.run_all()). Pode também ser rodado sozinho: nesse caso, as seções 2 a 4
do arquivo consolidado (escritas por relatorios_credito.write_reports) ficam como estavam
antes — este script nunca apaga o que vem depois da Seção 1.
Não modifica os CSVs de entrada.
"""
from pathlib import Path
import hashlib, json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
saida = RAIZ / "resultados" / "graficos_eda"
saida.mkdir(parents=True, exist_ok=True)
matplotlib.rcParams["svg.hashsalt"] = "credito-eda"
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado

dados = pd.read_csv(RAIZ / "credit_risk_dataset.csv")
dados.describe().T.to_csv(saida / "estatisticas_originais.csv")
pd.DataFrame({"tipo": dados.dtypes.astype(str), "nulos": dados.isna().sum()}).to_csv(saida / "inventario.csv")
nulos = dados.isna().sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(nulos.index, nulos.values, color="#7F6000")
ax.set_title("Valores ausentes por coluna — base original")
ax.set_ylabel("Quantidade de valores ausentes")
ax.tick_params(axis="x", rotation=70)
for bar, n in zip(bars, nulos.values):
    if n:
        ax.text(bar.get_x() + bar.get_width()/2, n, f"{n:,}", ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, max(nulos.max() * 1.18, 1))
fig.tight_layout(); fig.savefig(saida / "05_valores_nulos.svg", metadata={"Date": None}); fig.savefig(saida / "05_valores_nulos.png", dpi=140); plt.close(fig)
classes = dados["loan_status"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(["Em dia (0)", "Inadimplente (1)"], classes.values, color=["#4472C4", "#C00000"])
ax.set_title("Distribuição da variável-alvo")
ax.set_ylabel("Quantidade de contratos")
for bar, n in zip(bars, classes.values):
    ax.text(bar.get_x() + bar.get_width()/2, n, f"{n:,}\n{n/len(dados):.1%}", ha="center", va="bottom")
ax.set_ylim(0, classes.max()*1.18)
fig.tight_layout(); fig.savefig(saida / "01_distribuicao_alvo.svg", metadata={"Date": None}); fig.savefig(saida / "01_distribuicao_alvo.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.geomspace(dados.person_income.min(), dados.person_income.max(), 46)
for status, cor in [(0, "#4472C4"), (1, "#C00000")]:
    valores = dados.loc[dados.loan_status == status, "person_income"]
    ax.hist(valores, bins=bins, weights=np.ones(len(valores))/len(valores),
            histtype="step", linewidth=1.8, color=cor, label=f"Status {status}")
ax.set_xscale("log")
ax.legend()
ax.set_title("Distribuição da renda anual por status do empréstimo")
ax.set_xlabel("Renda anual (escala logarítmica; todos os registros)"); ax.set_ylabel("Fração da classe em cada intervalo")
fig.tight_layout(); fig.savefig(saida / "02_histograma_renda_status.svg", metadata={"Date": None}); fig.savefig(saida / "02_histograma_renda_status.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
dados.boxplot(column="loan_percent_income", by="loan_status", ax=ax)
fig.suptitle("")
ax.set_title("Comprometimento percentual da renda por status")
ax.set_xlabel("Status (0 = em dia; 1 = inadimplente)"); ax.set_ylabel("Razão empréstimo/renda (proporção: 0,10 = 10%)")
fig.tight_layout(); fig.savefig(saida / "03_boxplot_comprometimento.svg", metadata={"Date": None}); fig.savefig(saida / "03_boxplot_comprometimento.png", dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
for ax, coluna, titulo in [
    (axes[0], "person_income", "Renda anual (person_income)"),
    (axes[1], "loan_amnt", "Valor do empréstimo (loan_amnt)"),
]:
    dados.boxplot(column=coluna, by="loan_status", ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("Status (0 = em dia; 1 = inadimplente)")
    ax.set_yscale("log")
fig.suptitle("")
fig.suptitle("Identificação de valores discrepantes por status", y=1.02)
fig.tight_layout(); fig.savefig(saida / "06_boxplot_outliers_renda_valor.svg", metadata={"Date": None}); fig.savefig(saida / "06_boxplot_outliers_renda_valor.png", dpi=140, bbox_inches="tight"); plt.close(fig)

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
fig.tight_layout(); fig.savefig(saida / "04_correlacao_pearson.svg", metadata={"Date": None}); fig.savefig(saida / "04_correlacao_pearson.png", dpi=140); plt.close(fig)

def contagem_outliers_iqr(serie):
    q1, q3 = serie.quantile([0.25, 0.75])
    iqr = q3 - q1
    limite = q3 + 1.5 * iqr
    return int((serie > limite).sum()), float(limite)

outliers_renda, limite_renda = contagem_outliers_iqr(dados.person_income)
outliers_valor, limite_valor = contagem_outliers_iqr(dados.loan_amnt)

resumo = {
    "percentual_inadimplencia": round(dados.loan_status.mean() * 100, 2),
    "renda_mediana_em_dia": round(dados.loc[dados.loan_status == 0, "person_income"].median(), 2),
    "renda_mediana_inadimplente": round(dados.loc[dados.loan_status == 1, "person_income"].median(), 2),
    "comprometimento_mediano_em_dia": round(dados.loc[dados.loan_status == 0, "loan_percent_income"].median(), 3),
    "comprometimento_mediano_inadimplente": round(dados.loc[dados.loan_status == 1, "loan_percent_income"].median(), 3),
    "correlacao_status_comprometimento": round(dados[["loan_status", "loan_percent_income"]].corr().iloc[0, 1], 3),
    "correlacao_status_taxa_juros": round(dados[["loan_status", "loan_int_rate"]].corr().iloc[0, 1], 3),
    "outliers_renda_iqr": outliers_renda,
    "limite_renda_iqr": round(limite_renda, 2),
    "outliers_valor_iqr": outliers_valor,
    "limite_valor_iqr": round(limite_valor, 2),
}
(saida / "resumo_numerico.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n")

SECAO_1 = f"""## 1. Análise exploratória (EDA)

### Distribuição do alvo

A inadimplência representa {resumo['percentual_inadimplencia']:.2f}% dos registros. Há diferença relevante entre as classes; o split deverá preservar a proporção e a avaliação usará recall, precisão e F1 além da acurácia. O balanceamento, quando aplicado, ficará restrito ao treino.

### Renda anual

A mediana da renda é {resumo['renda_mediana_em_dia']:,.0f} para contratos em dia e {resumo['renda_mediana_inadimplente']:,.0f} para contratos inadimplentes. O histograma usa intervalos comuns e eixo logarítmico, sem remover registros. Cada classe é normalizada separadamente; a altura mostra sua fração no intervalo. Valores altos não demonstram erro por si sós.

### Comprometimento da renda

As medianas de `loan_percent_income` são {resumo['comprometimento_mediano_em_dia']:.3f} para a classe 0 e {resumo['comprometimento_mediano_inadimplente']:.3f} para a classe 1. A correlação de Pearson com o alvo é {resumo['correlacao_status_comprometimento']:.3f}; isso é associação descritiva, não causalidade.

### Outliers: renda e valor do empréstimo

O critério de referência é o limite superior do boxplot (Q3 + 1,5×IQR), usado só para
visualizar a cauda, não como regra de exclusão. Por esse critério, {resumo['outliers_renda_iqr']:,}
registros de `person_income` ficam acima de {resumo['limite_renda_iqr']:,.0f} e
{resumo['outliers_valor_iqr']:,} registros de `loan_amnt` ficam acima de
{resumo['limite_valor_iqr']:,.0f}. O gráfico 06 mostra essas caudas por status, em escala
logarítmica. A decisão (seção 2 abaixo) é manter esses valores: são extremos raros e
plausíveis (rendas e empréstimos altos existem), não erros de digitação como as idades de
123/144 anos. Essa manutenção pesa mais no KNN, sensível a distâncias euclidianas mesmo após
o StandardScaler, do que na árvore, que corta por limiar e é robusta à magnitude dos valores.

### Valores ausentes

`person_emp_length` tem {int(dados['person_emp_length'].isna().sum()):,} valores ausentes e `loan_int_rate` tem {int(dados['loan_int_rate'].isna().sum()):,}. A mediana será usada para o tempo de emprego porque a distribuição é assimétrica; a média será usada para a taxa porque média e mediana são próximas. Esses valores são aprendidos somente no treino de cada dobra, depois do split.

### Correlações e próximos passos

A correlação entre `loan_status` e `loan_int_rate` é {resumo['correlacao_status_taxa_juros']:.3f}. Correlação não determina exclusão automática nem causalidade. A preparação (seções 2 a 4) remove repetições exatas e idades de 123/144 anos, invalida dois tempos de emprego impossíveis, mantém rendas extremas plausíveis, substitui a razão redundante pela coluna exigida e aprende imputadores somente no treino. Esta EDA descreve toda a base; não é uma análise cega de holdout.

As figuras ficam em `resultados/graficos_eda/`. O gráfico 05 resume os nulos observados antes da imputação e o gráfico 06 mostra os outliers de renda e valor do empréstimo por status. Os CSVs foram apenas lidos e tiveram seus hashes conferidos antes e depois da execução.
"""

CABECALHO = (
    "# EDA e preparação dos dados\n\n"
    "Este arquivo é escrito em duas etapas por dois scripts diferentes: a Seção 1 por "
    "`notebooks/02_eda_graficos.py` (lendo o CSV original) e as Seções 2 a 4 por "
    "`notebooks/pipeline_credito.py`, via `relatorios_credito.write_reports` (lendo a base "
    "já limpa). Rodar `notebooks/03_executar_pipeline.py` executa as duas etapas em "
    "sequência e produz o arquivo completo; rodar só este script deixa as Seções 2 a 4 como "
    "estavam.\n\n"
)

destino = RAIZ / "documentacao" / "01_eda_e_preparacao.md"
if destino.exists():
    existente = destino.read_text()
    marcador = "## 2. Limpeza e imputação"
    resto = existente.split(marcador, 1)
    cauda = ("\n" + marcador + resto[1]) if len(resto) == 2 else ""
else:
    cauda = (
        "\n## 2. Limpeza e imputação\n\n"
        "Seção não gerada: rode `notebooks/pipeline_credito.py` (ou "
        "`notebooks/03_executar_pipeline.py`, que já inclui essa etapa) para produzi-la, "
        "junto com as Seções 3 e 4.\n"
    )
destino.write_text(CABECALHO + SECAO_1 + cauda)

for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print("EDA gerada; integridade dos CSVs confirmada.")
