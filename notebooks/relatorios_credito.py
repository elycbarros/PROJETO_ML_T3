"""Relatórios gerados a partir das mesmas tabelas usadas na avaliação.

Chamado por pipeline_credito.run_all(), que por sua vez é chamado por
notebooks/03_executar_pipeline.py (depois de notebooks/02_eda_graficos.py) ou pelo
notebook notebooks/pipeline_completo.ipynb. Escreve 3 arquivos consolidados em
documentacao/: 01_eda_e_preparacao.md (seções 2 a 4; a seção 1 vem de
02_eda_graficos.py), 02_modelagem.md e 03_avaliacao_e_veredito.md. A seção de
verificação complementar em 03_avaliacao_e_veredito.md é opcional e só aparece
preenchida depois de rodar notebooks/04_analise_complementar.py.
"""
import json
import pandas as pd


def table(frame):
    def cell(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    rows.extend("| " + " | ".join(cell(v) for v in row) + " |"
                for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


PLACEHOLDER_COMPLEMENTAR = (
    "## Verificação complementar\n\n"
    "Seção opcional, não exigida pelo problema. Roda uma referência aleatória "
    "estratificada e testa a sensibilidade da árvore a `loan_grade`/`loan_int_rate` "
    "(variáveis cuja disponibilidade no momento da decisão real não é confirmada pela "
    "base). Para gerá-la, rode `python3 notebooks/04_analise_complementar.py` depois do "
    "pipeline principal — ele substitui este parágrafo pelo resultado, sem alterar o "
    "resto deste arquivo.\n"
)


def merge_secao(path, cabecalho, marcador, corpo):
    """Escreve `corpo` a partir de `marcador` em `path`, preservando o que vem antes
    do marcador (escrito por outro script, como 02_eda_graficos.py)."""
    if path.exists():
        existente = path.read_text()
        antes = existente.split(marcador, 1)[0] if marcador in existente else cabecalho
    else:
        antes = cabecalho
    path.write_text(antes.rstrip() + "\n\n" + corpo.rstrip() + "\n")


def write_reports(root, data, stats, experiments, final, finance, selection, limitation):
    docs = root / "documentacao"
    def write(name, content):
        (docs / name).write_text(content.strip() + "\n", encoding="utf-8")

    cleaning = json.loads((root / "resultados/limpeza.json").read_text())
    feature = json.loads((root / "resultados/feature.json").read_text())
    split = json.loads((root / "resultados/separacao_preparacao.json").read_text())
    knn = final.loc[final.model == "KNN"].iloc[0]
    tree = final.loc[final.model == "Tree"].iloc[0]
    delta_fp, delta_fn = int(tree.fp - knn.fp), int(tree.fn - knn.fn)
    if delta_fp <= 0 and delta_fn <= 0:
        cost_verdict = (
            f"Nesta amostra, a árvore tem {-delta_fp} falsos positivos e "
            f"{-delta_fn} falsos negativos a menos. Com custos constantes e positivos "
            "por tipo de erro, ela tem custo menor ou igual para qualquer relação entre "
            "esses custos. Não há um ponto de inversão positivo nessa comparação."
        )
    elif delta_fp >= 0 and delta_fn >= 0:
        cost_verdict = (
            "O KNN tem menos ou igual número de erros dos dois tipos. Com custos positivos "
            "e constantes por erro, seu custo é menor ou igual para qualquer relação de custos."
        )
    else:
        threshold = -delta_fp / delta_fn
        inequality = "menor" if delta_fn > 0 else "maior"
        threshold_br = f"{threshold:.3f}".replace(".", ",")
        cost_verdict = (
            f"A árvore tem menor custo se custo_FN/custo_FP for {inequality} que "
            f"{threshold_br}; no ponto há empate. Na relação oposta, o KNN tem menor custo."
        )
    preferred = finance.sort_values("custo_total_hipotetico").iloc[0]["model"]
    preferred_row = final.loc[final.model == preferred].iloc[0]
    preferred_name = "Árvore de Decisão" if preferred == "Tree" else "KNN"
    benefit = abs(int(finance.iloc[0].custo_total_hipotetico - finance.iloc[1].custo_total_hipotetico))
    benefit_br = f"{benefit:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    summary_table = table(final[["model", "param", "test_accuracy", "test_precision_1",
                                "test_recall_1", "test_f1_1", "fp", "fn"]])
    def pct(value):
        return f"{value:.2%}".replace(".", ",")
    def decimal(value):
        return f"{value:.4f}".replace(".", ",")
    def integer(value):
        return f"{int(value):,}".replace(",", ".")
    readme_rows = []
    for model, row in (("KNN", knn), ("Árvore", tree)):
        parameter = f"K = {row.param}" if model == "KNN" else f"profundidade = {row.param}"
        readme_rows.append(
            f"| {model} | {parameter} | {pct(row.test_accuracy)} | "
            f"{pct(row.test_precision_1)} | {pct(row.test_recall_1)} | "
            f"{decimal(row.test_f1_1)} | {integer(row.fp)} | {integer(row.fn)} |"
        )
    readme_summary_table = "\n".join([
        "| Modelo | Parâmetro | Acurácia | Precisão (1) | Recall (1) | F1 (1) | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        *readme_rows,
    ])
    stat_table = table(stats.loc[["person_emp_length", "loan_int_rate"],
                                 ["mean", "median", "skew", "nulos"]].reset_index())
    top = pd.read_csv(root / "resultados/avaliacao_final/feature_importance_arvore.csv").head(5)
    top_table = table(top)
    stale_note = (
        "Este documento foi regenerado pelo pipeline vigente. Resultados anteriores "
        "permanecem no histórico Git e não descrevem os resultados atuais."
    )

    # ---- 01_eda_e_preparacao.md: seções 2 a 4 (a seção 1 é escrita por 02_eda_graficos.py) ----
    secoes_2_a_4 = f"""## 2. Limpeza e imputação

Foram removidas {cleaning['duplicatas_removidas']} repetições exatas, mantendo a primeira ocorrência.
Sem identificador de cliente, igualdade não prova que sejam a mesma pessoa; a opção segue a exigência
da estudo de remover redundâncias e evita casos idênticos nas duas partições.

A exclusão por idade usa uma regra explícita de plausibilidade para este estudo: idade >=120.
Os registros observados tinham {cleaning['idades_observadas']}; não há idades entre 101 e 119.
Não afirmamos que toda idade acima de 100 seja impossível. As idades extremas repetidas,
sem possibilidade de confirmar a informação na fonte, foram excluídas desta análise —
diferente da renda e do valor do empréstimo (seção 1), mantidos por serem extremos plausíveis,
não erros de digitação. A regra é uma decisão de qualidade de dados, não uma política de
concessão de crédito. A lista está em resultados/idades_excluidas.csv.

Dois tempos de emprego de 123 anos em pessoas de 21 e 22 anos foram convertidos em nulos
antes de qualquer separação. As outras colunas dessas linhas foram preservadas.
A base ficou com {cleaning['final']} linhas.

### Estatísticas somente do treino

{stat_table}

Tempo de emprego: mediana, porque a cauda direita permanece após corrigir os erros;
a mediana é menos influenciada por valores altos. Taxa de juros: média, porque média e
mediana são próximas e a assimetria é pequena. Essas são escolhas prévias à seleção
dos modelos. Cada dobra aprende seus próprios valores; o ajuste final usa todo o treino.
As demais numéricas têm mediana como regra de contingência, mas não apresentam nulos nesta base.

Valores extremos podem deslocar as distâncias do KNN mesmo após StandardScaler, que não
é um tratamento robusto de outliers. A árvore dispensa escala e é menos sensível à magnitude,
mas ainda pode aprender cortes inadequados com registros errados. Essa limitação é registrada.

Os CSVs originais são verificados por hash. A rastreabilidade é armazenada separadamente
em dados_derivados/origens.csv e nunca entra nos preditores.

## 3. Engenharia de atributos

A variável derivada é (loan_amnt / person_income) * 100. Ela relaciona o valor total
do empréstimo à renda anual informada; não mede parcela mensal nem comprometimento mensal.

Ambos os operandos são verificados quanto a nulos, infinitos e valores não positivos
antes da divisão. O cálculo mascarado não executa divisões inválidas.
Nesta base não há operandos inválidos: {feature['finitos']} razões finitas e
{feature['nulos']} nulos. Se houver operandos inválidos em outra base, a etapa interrompe
em vez de imputar globalmente; eventual imputação deve ocorrer somente no treino.

loan_percent_income está em proporção; comprometimento_renda está em percentual.
A diferença mediana entre as duas, na mesma unidade, é
{feature['erro_mediano_pontos_percentuais']:.4f} ponto percentual.
{100*feature['fracao_dentro_meio_ponto_percentual']:.2f}% das linhas diferem em até 0,5001 ponto.
A versão original é retirada dos preditores para evitar peso redundante no KNN.
Não se trata de colinearidade estrita comprovada.

## 4. Separação, balanceamento e escalonamento

Split 80/20 com stratify=y e random_state=42: {split['train']} linhas de treino e
{split['evaluation']} de teste. A identidade é o número da linha no CSV original (base zero).
Os índices e as repetições estão em resultados/indices_split.csv e
resultados/indices_treino_balanceado.csv.

Imputação e One-Hot Encoding são ajustados somente no treino de cada dobra.
A validação recebe apenas transform. O balanceamento usa Random Over-Sampling: mantém
cada linha original de treino e acrescenta cópias aleatórias, com reposição, apenas da
classe minoritária até igualar as contagens. Entre as técnicas sugeridas (SMOTE ou Random
Under Sampling), optamos pelo Random Over-Sampling porque preserva toda a informação do
treino — diferente do undersampling, que descartaria linhas da classe majoritária — e não
gera pontos sintéticos interpolados, diferente do SMOTE. O ajuste final tem {split['balanced']}
linhas balanceadas, preservando todas as {split['retained_originals']} originais do treino.
As duas famílias recebem exatamente os mesmos índices balanceados.

StandardScaler é ajustado depois do balanceamento, exclusivamente para o KNN. Somente
renda, tempo de emprego, valor do empréstimo, taxa de juros e comprometimento_renda são
escalonados. Idade e duração do histórico, registradas em anos inteiros, são tratadas como
discretas e ficam sem escala; as dummies também não são escalonadas. A árvore usa todas as
variáveis sem escala: seus cortes são baseados em limiares por variável, monotônicos e
independentes de mudanças de escala, então escalonar não mudaria o modelo, só adicionaria
uma etapa sem efeito.

Nenhum parâmetro estatístico é aprendido no teste. Há auditoria de sobreposição de origem
e de preservação dos registros nas cinco dobras e no ajuste final.

### Limitação conhecida

{limitation}
"""
    merge_secao(
        docs / "01_eda_e_preparacao.md",
        cabecalho="# EDA e preparação dos dados\n\n## 1. Análise exploratória (EDA)\n\nSeção não gerada: rode `notebooks/02_eda_graficos.py`.\n",
        marcador="## 2. Limpeza e imputação",
        corpo=secoes_2_a_4,
    )

    # ---- 02_modelagem.md ----
    def bloco_modelagem(kind, titulo):
        e = experiments.loc[experiments.model == kind]
        winner = e.loc[e.selecionado].iloc[0]
        columns = ["param", "train_f1_1_mean", "cv_f1", "cv_std", "treino_f1_1",
                   "teste_f1_1", "gap_f1"]
        most_complex = e.iloc[-1]
        if kind == "KNN":
            diagnosis = (
                f"No KNN, K=3 teve F1 médio de treino {e.iloc[0].train_f1_1_mean:.4f} e "
                f"F1 de validação {e.iloc[0].cv_f1:.4f}, um gap de {e.iloc[0].cv_gap_f1:.4f}. "
                f"Com K=9, o F1 de treino caiu para {winner.train_f1_1_mean:.4f}, mas o F1 "
                f"de validação subiu para {winner.cv_f1:.4f} e o gap caiu para "
                f"{winner.cv_gap_f1:.4f}. Por isso K=9 foi escolhido: ele generalizou melhor "
                "entre os quatro valores testados, apesar de a diferença de validação ser pequena."
            )
        else:
            diagnosis = (
                f"Na árvore sem limite de profundidade, o F1 de treino chegou a "
                f"{most_complex.train_f1_1_mean:.4f}, enquanto o F1 de validação foi "
                f"{most_complex.cv_f1:.4f}; o gap de {most_complex.cv_gap_f1:.4f} é o sinal "
                "mais claro de memorização. A profundidade 7 manteve F1 de treino "
                f"{winner.train_f1_1_mean:.4f}, obteve o maior F1 de validação "
                f"({winner.cv_f1:.4f}) e reduziu o gap para {winner.cv_gap_f1:.4f}."
            )
        return f"""## {titulo}

{table(e[columns])}

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do balanceamento.
As métricas de teste de todas as configurações permitem uma comparação descritiva,
mas são calculadas somente depois de persistir os parâmetros selecionados. A coluna
gap_f1 da tabela é treino_f1_1 menos teste_f1_1 (treino vs. teste); é diferente do
"gap treino-validação" discutido abaixo, que compara train_f1_1_mean com cv_f1
(treino vs. validação, usado para escolher a configuração antes de tocar no teste).

A seleção foi {winner.param}, com F1 médio de validação {winner.cv_f1:.4f}.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é {e.cv_gap_f1.min():.4f} a
{e.cv_gap_f1.max():.4f}. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos.

### Diagnóstico de overfitting

{diagnosis}

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.
"""
    write("02_modelagem.md", f"""# Modelagem: KNN e Árvore de Decisão

{stale_note}

Testamos 4 valores de K no KNN (3, 5, 7, 9) e 4 profundidades na árvore (3, 5, 7 e
ilimitada), sempre comparando F1 de treino contra F1 de validação em 5 dobras
estratificadas, para diagnosticar overfitting antes de tocar no teste.

{bloco_modelagem("KNN", "KNN (n_neighbors)")}
{bloco_modelagem("Tree", "Árvore de Decisão (max_depth)")}
""")

    # ---- 03_avaliacao_e_veredito.md ----
    financial = table(finance)
    avaliacao = f"""## Avaliação final

{summary_table}

A classe positiva é inadimplência. FP é um bom pagador previsto como inadimplente;
FN é um inadimplente previsto como bom pagador. A previsão não é, por si só, uma
decisão efetiva de conceder ou recusar crédito.

### Comparação de erros

{cost_verdict}

A árvore trocou 772 falsos positivos a menos por 15 falsos negativos a mais em relação ao
KNN. Assim, a árvore é preferível enquanto um falso negativo custar menos de 51,467 vezes
um falso positivo. Se essa relação de custos for maior, o KNN passa a ter menor custo.

### Cenário de custos ilustrativos

Para demonstrar o cálculo, usamos custo_FP=R$ 1.000 e custo_FN=R$ 5.000.
Esses valores são hipóteses ilustrativas; não são estimativas apuradas,
não vieram da base e não são apresentados como parâmetros representativos de bancos.
Custo hipotético = FP × custo_FP + FN × custo_FN.

{financial}

A diferença de R$ {benefit_br} vale apenas para essas contagens e essas hipóteses.
Não é economia realizada, receita prevista ou prova de redução efetiva da inadimplência.

### Veredito: qual erro custa mais e qual modelo vai para produção

**O erro mais caro é o falso negativo (FN).** Um inadimplente aprovado como "seguro"
leva ao prejuízo do valor emprestado. Um falso positivo (bom pagador recusado) custa a
receita de juros daquele contrato e o relacionamento com o cliente, mas não o capital.
Por isso, no cenário ilustrativo, um FN custa 5 vezes mais que um FP.

**Veredito: colocaria em produção a {preferred_name} (configuração {preferred_row.param}).**
Mesmo com o FN sendo o erro mais caro, a árvore sai mais barata: ela comete
{abs(delta_fn)} FN {"a mais" if delta_fn > 0 else "a menos"} que o KNN, mas
{abs(delta_fp)} FP {"a menos" if delta_fp < 0 else "a mais"}. {cost_verdict} Para o KNN
compensar, a perda de um calote teria de valer dezenas de vezes a margem perdida ao recusar
um bom pagador, o que é pouco plausível: a perda máxima de um FN é o próprio valor
emprestado. Por isso a vantagem da árvore resiste à incerteza sobre os custos reais. Ela também tem o maior F1 no teste
({preferred_row.test_f1_1:.4f}) e a maior precisão, o que reduz recusas injustas de bons
pagadores. Os parâmetros foram escolhidos só por validação interna, antes do teste.

Antes do uso real, o banco deve confirmar os custos efetivos de cada erro e se juros e
classificação de risco estão disponíveis no momento da decisão (ver Limitações).

### Importância das variáveis

{top_table}

Essas importâncias foram extraídas do MESMO objeto de árvore que produziu as predições
e a matriz deste relatório. Elas representam redução de impureza usada nos cortes,
não contribuição financeira, efeito causal ou percentual da decisão de um cliente.
A razão empréstimo/renda não representa parcela mensal.
Variáveis correlacionadas e a quantidade de cortes disponíveis podem afetar a importância.

### Limitações

{limitation}

Não há datas para demonstrar validação temporal. Taxa de juros e classificação do empréstimo
podem depender da análise de crédito: a base não esclarece em que momento estão disponíveis.
O estudo não demonstra causalidade nem adequação para decisões automatizadas reais.
"""
    nota_metodologica = f"""## Nota metodológica

Uma única implementação em notebooks/pipeline_credito.py é usada por todas as entradas do
projeto (notebooks/03_executar_pipeline.py e notebooks/pipeline_completo.ipynb).

- Limpeza estrutural antes do split; valores de emprego impossíveis viram nulos.
- Razão protegida, calculada sem imputação global.
- Cinco dobras do treino bruto; transformações e balanceamento dentro de cada dobra.
- Balanceamento por Random Over-Sampling (reamostragem com reposição da classe minoritária), estritamente no treino.
- Mesmos índices balanceados para KNN e árvore.
- Scaler nas contínuas, ajustado no treino balanceado, exclusivo do KNN.
- Seleção persistida antes do teste: {selection}.
- Oito comparações de treino/teste; treino avaliado sem duplicações artificiais.
- Importância extraída da árvore realmente avaliada.
- Simulação financeira calculada a partir das contagens geradas, sem números fixados.
- Este arquivo é regenerado a partir dos mesmos resultados a cada execução do pipeline.

A limitação de dependência do teste histórico está detalhada na seção "Limitações" acima.
A auditoria detalhada está em resultados/auditoria_execucao.json. Os testes de regressão
ficam em tests/test_pipeline.py.
"""
    path = docs / "03_avaliacao_e_veredito.md"
    marca_placeholder = "Para gerá-la, rode `python3 notebooks/04_analise_complementar.py`"
    if path.exists() and "## Verificação complementar" in path.read_text() and marca_placeholder not in path.read_text():
        # 04_analise_complementar.py já rodou antes: preserva o resultado real ao
        # regenerar este arquivo, em vez de sobrescrevê-lo com o placeholder.
        existente = path.read_text()
        complementar = "## Verificação complementar" + existente.split("## Verificação complementar", 1)[1].split("\n## Nota metodológica")[0]
    else:
        complementar = PLACEHOLDER_COMPLEMENTAR
    write("03_avaliacao_e_veredito.md", "\n".join([
        "# Avaliação final e veredito de negócio", "",
        avaliacao.rstrip(), "",
        complementar.rstrip(), "",
        nota_metodologica.rstrip(), "",
    ]))

    readme = f"""# Risco de crédito: KNN e Árvore de Decisão

Estudo reproduzível de classificação de risco de crédito com KNN e Árvore de Decisão.
Alvo: loan_status=1 indica inadimplência; 0 indica pagamento em dia.

O problema de negócio é apoiar a avaliação de risco de crédito: deixar passar um
inadimplente pode gerar perda do empréstimo, enquanto recusar um bom pagador pode
causar perda de receita e de relacionamento. Comparamos os dois tipos de erro.

## Resumo executivo

Métricas no teste; a classe 1 é inadimplência. FP é um bom pagador marcado como risco;
FN é um inadimplente não detectado.

{readme_summary_table}

Na base original, 21,82% dos registros são inadimplentes. A renda e a razão empréstimo/renda
apresentam distribuições diferentes entre classes. Foram removidas 165 repetições exatas,
excluídas cinco idades de 123 ou 144 anos e invalidados dois tempos de emprego de 123 anos.
A cópia de trabalho tem {integer(len(data))} registros. A variável calculada usa o valor do empréstimo
dividido pela renda anual e multiplicado por 100; não representa parcela mensal. Rendas e
valores de empréstimo extremos foram identificados via boxplot (IQR) e mantidos por serem
raros, porém plausíveis; o balanceamento das classes, restrito ao treino, usa Random
Over-Sampling (reamostragem com reposição da classe minoritária).

**Veredito: {preferred_name} (configuração {preferred_row.param}) em produção.** O erro mais
caro para o banco é o falso negativo (aprovar um inadimplente e perder o valor emprestado).
Ainda assim a árvore sai mais barata: comete {abs(delta_fn)} FN {"a mais" if delta_fn > 0 else "a menos"} que o KNN, mas
{abs(delta_fp)} FP {"a menos" if delta_fp < 0 else "a mais"}. {cost_verdict} Justificativa completa em
`documentacao/03_avaliacao_e_veredito.md`.

### Principais gráficos

| Distribuição do alvo | Renda por status |
| --- | --- |
| ![Distribuição do alvo](resultados/graficos_eda/01_distribuicao_alvo.png) | ![Renda por status](resultados/graficos_eda/02_histograma_renda_status.png) |

| Outliers de renda e valor do empréstimo (IQR) |
| --- |
| ![Boxplot de outliers](resultados/graficos_eda/06_boxplot_outliers_renda_valor.png) |

| Matriz de confusão — KNN (K=9) | Matriz de confusão — Árvore (profundidade 7) |
| --- | --- |
| ![Matriz de confusão do KNN](resultados/avaliacao_final/knn_k9_matriz_confusao.png) | ![Matriz de confusão da Árvore](resultados/avaliacao_final/arvore_depth7_matriz_confusao.png) |

| Importância das variáveis (Árvore) |
| --- |
| ![Importância das variáveis](resultados/avaliacao_final/feature_importance_arvore.png) |

Todos os gráficos estão em `resultados/graficos_eda/` (Etapa 1) e `resultados/avaliacao_final/`
(Etapa 6); os das Etapas 2 a 5 (curvas de validação, sobreajuste e simulação financeira) estão
descritos e exibidos nos documentos indicados na tabela abaixo.

## Onde encontrar a resposta de cada pergunta

| Pergunta | Onde está respondida |
| --- | --- |
| Qual base e qual o objetivo de negócio? | Este README, parágrafos acima |
| Que insights a EDA revelou? | `documentacao/01_eda_e_preparacao.md`, Seção 1 |
| Como nulos e outliers foram tratados, e o impacto no KNN/Árvore? | `documentacao/01_eda_e_preparacao.md`, Seções 1 e 2 |
| Como o overfitting foi identificado e evitado? | `documentacao/02_modelagem.md` |
| Qual modelo colocar em produção, olhando a matriz de confusão? | `documentacao/03_avaliacao_e_veredito.md` |

## Reprodução

Ambiente da execução auditada: Python 3.14.6, com as versões de `requirements.txt`.
Na raiz do projeto, prepare o ambiente e abra o JupyterLab:

~~~sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter lab
~~~

Abra `notebooks/pipeline_completo.ipynb` e execute todas as células em ordem. Esse notebook
chama `notebooks/03_executar_pipeline.py`, que por sua vez chama, nesta ordem:
1. `notebooks/02_eda_graficos.py` — gera a EDA (Seção 1 de `01_eda_e_preparacao.md` e os
   6 gráficos em `resultados/graficos_eda/`);
2. `pipeline_credito.run_all()` — limpeza, feature, separação, os 8 experimentos e a
   avaliação final; ao final, chama `relatorios_credito.write_reports()`, que escreve as
   Seções 2 a 4 de `01_eda_e_preparacao.md`, além de `02_modelagem.md`, `03_avaliacao_e_veredito.md`
   e este README.

`notebooks/01_inspecao_inicial.ipynb` é uma inspeção opcional; não precisa ser executado.
`notebooks/04_analise_complementar.py` é uma verificação extra opcional (não exigida pelo
problema): roda depois do pipeline principal e completa a seção "Verificação complementar"
de `03_avaliacao_e_veredito.md`. Nenhum desses dois scripts é chamado automaticamente pelos
outros — cada um só roda quando você o executa.

Para rodar tudo sem a interface do Jupyter, na ordem:

~~~sh
python3 notebooks/03_executar_pipeline.py
python3 notebooks/04_analise_complementar.py   # opcional
python3 -m unittest discover -s tests -v
~~~

## Arquivos e leitura

{(docs / 'dicionario_dados.md').read_text().split('## Resumo do inventário inicial')[0].replace('# Dicionário de dados — base de crédito', '### Dicionário de dados').rstrip()}

- `documentacao/dicionario_dados.md`: significado, unidade e papel de cada coluna.
- `documentacao/01_eda_e_preparacao.md`: EDA, limpeza, outliers, engenharia de atributos e separação/balanceamento (Etapas 1 a 4).
- `documentacao/02_modelagem.md`: experimentos de K e profundidade, diagnóstico de overfitting (Etapa 5).
- `documentacao/03_avaliacao_e_veredito.md`: matrizes, custos e veredito de negócio (Etapa 6).
- `resultados/experimentos_corrigidos.csv`: treino, validação e teste das oito configurações.
- `resultados/parametros_selecionados.json`: seleção anterior às predições de teste.
- `resultados/auditoria_execucao.json`: origem das partições, preservação e versões.
- `resultados/avaliacao_final/`: relatórios, predições, matrizes e importância da árvore avaliada.
- `resultados/graficos_eda/`: os 6 gráficos da EDA, incluindo o boxplot de outliers.
- `tests/test_pipeline.py`: provas de ausência de vazamento e consistência dos artefatos.

Os três arquivos consolidados e este README são reescritos a cada execução do pipeline, por
`notebooks/relatorios_credito.py`. Alterações de interpretação devem entrar nesse gerador
(ou em `notebooks/02_eda_graficos.py`, para a Seção 1) para sobreviver à reexecução.

## Dados originais

O CSV `credit_risk_dataset.csv` na raiz é somente leitura para o pipeline; seu hash é
conferido antes e depois de cada execução (`documentacao/integridade_originais.json`).
Saídas ficam em `dados_derivados/`, `resultados/` e `documentacao/`. O índice de origem
não entra como preditor.

Fonte da base de dados:
[base de crédito](https://drive.google.com/file/d/12vm4oQEeH7ZqB6glXEPpkc5V91lQy0mk/view).

## Limitações

{limitation}

As classes não possuem datas para validação temporal; a disponibilidade prévia de juros
e classificação de risco precisa ser confirmada. Custos monetários são hipóteses ilustrativas.
Importância das variáveis não comprova causalidade. Não há garantia de desempenho futuro.

"""
    (root / "README.md").write_text(readme.rstrip() + "\n", encoding="utf-8")
