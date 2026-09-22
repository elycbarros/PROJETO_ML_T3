"""Relatórios gerados a partir das mesmas tabelas usadas na avaliação."""
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
        cost_verdict = (
            f"A árvore tem menor custo se custo_FN/custo_FP for {inequality} que "
            f"{threshold:.3f}; no ponto há empate. Na relação oposta, o KNN tem menor custo."
        )
    preferred = finance.sort_values("custo_total_hipotetico").iloc[0]["model"]
    preferred_row = final.loc[final.model == preferred].iloc[0]
    preferred_name = "Árvore de Decisão" if preferred == "Tree" else "KNN"
    benefit = abs(int(finance.iloc[0].custo_total_hipotetico - finance.iloc[1].custo_total_hipotetico))
    benefit_br = f"{benefit:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    summary_table = table(final[["model", "param", "test_accuracy", "test_precision_1",
                                "test_recall_1", "test_f1_1", "fp", "fn"]])
    stat_table = table(stats.loc[["person_emp_length", "loan_int_rate"],
                                 ["mean", "median", "skew", "nulos"]].reset_index())
    top = pd.read_csv(root / "resultados/avaliacao_final/feature_importance_arvore.csv").head(5)
    top_table = table(top)
    stale_note = (
        "Este documento foi regenerado pelo pipeline vigente. Resultados anteriores "
        "permanecem no histórico Git e não devem ser usados na análise."
    )
    write("data_prep.md", f"""# Limpeza e imputação

Foram removidas {cleaning['duplicatas_removidas']} repetições exatas, mantendo a primeira ocorrência.
Sem identificador de cliente, igualdade não prova que sejam a mesma pessoa; a opção segue a exigência
da estudo de remover redundâncias e evita casos idênticos nas duas partições.

A exclusão por idade usa uma regra explícita de plausibilidade para este estudo: idade >=120.
Os registros observados tinham {cleaning['idades_observadas']}; não há idades entre 101 e 119.
Não afirmamos que toda idade acima de 100 seja impossível. As idades extremas repetidas,
sem possibilidade de confirmar a informação na fonte, foram excluídas desta análise.
A regra é uma decisão de qualidade de dados, não uma política de concessão de crédito.
A lista está em resultados/idades_excluidas.csv.

Dois tempos de emprego de 123 anos em pessoas de 21 e 22 anos foram convertidos em nulos
antes de qualquer separação. As outras colunas dessas linhas foram preservadas.
A base ficou com {cleaning['final']} linhas.

## Estatísticas somente do treino

{stat_table}

Tempo de emprego: mediana, porque a cauda direita permanece após corrigir os erros;
a mediana é menos influenciada por valores altos. Taxa de juros: média, porque média e
mediana são próximas e a assimetria é pequena. Essas são escolhas prévias à seleção
dos modelos. Cada dobra aprende seus próprios valores; o ajuste final usa todo o treino.
As demais numéricas têm mediana como regra de contingência, mas não apresentam nulos nesta base.

Rendas e empréstimos positivos extremos são mantidos: raridade não demonstra erro.
Valores extremos podem deslocar as distâncias do KNN mesmo após StandardScaler, que não
é um tratamento robusto de outliers. A árvore dispensa escala e é menos sensível à magnitude,
mas ainda pode aprender cortes inadequados com registros errados. Essa limitação é registrada.

Os CSVs originais são verificados por hash. A rastreabilidade é armazenada separadamente
em dados_derivados/origens.csv e nunca entra nos preditores.
""")
    write("feature_engineering.md", f"""# Engenharia de atributos

A fórmula definido é (loan_amnt / person_income) * 100. Ela relaciona o valor total
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
""")
    write("separacao_preparacao.md", f"""# Separação e preparação

Split 80/20 com stratify=y e random_state=42: {split['train']} linhas de treino e
{split['evaluation']} de teste. A identidade é o número da linha no CSV original (base zero).
Os índices e as repetições estão em resultados/indices_split.csv e
resultados/indices_treino_balanceado.csv.

Imputação e One-Hot Encoding são ajustados somente no treino de cada dobra.
A validação recebe apenas transform. O oversampling mantém cada linha de treino
e acrescenta cópias da classe minoritária. O ajuste final tem {split['balanced']} linhas
balanceadas, preservando todas as {split['retained_originals']} originais do treino.
As duas famílias recebem exatamente os mesmos índices balanceados.

StandardScaler é ajustado depois do balanceamento. Somente renda, tempo de emprego,
valor do empréstimo, taxa de juros e comprometimento_renda são escalonados.
Idade e duração do histórico, registradas em anos inteiros, são tratadas como discretas
e ficam sem escala; as dummies também não são escalonadas. Essa decisão atende à
separação pedida entre contínuas e demais atributos, mas deixa as durações em sua
unidade original no cálculo de distância. A árvore usa todas as variáveis sem escala.

Nenhum parâmetro estatístico é aprendido no teste. Há auditoria de sobreposição de origem
e de preservação dos registros nas cinco dobras e no ajuste final.

## Limitação conhecida

{limitation}
""")

    for kind, filename in [("KNN", "experimentos_knn.md"), ("Tree", "experimentos_arvore.md")]:
        e = experiments.loc[experiments.model == kind]
        winner = e.loc[e.selecionado].iloc[0]
        columns = ["param", "train_f1_1_mean", "cv_f1", "cv_std", "treino_f1_1",
                   "teste_f1_1", "gap_f1"]
        write(filename, f"""# Experimentos: {kind}

{stale_note}

{table(e[columns])}

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do oversampling.
As métricas de teste de todas as configurações atendem à comparação pedida pelo problema,
mas são calculadas somente depois de persistir os parâmetros selecionados.

A seleção foi {winner.param}, com F1 médio de validação {winner.cv_f1:.4f}.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é {e.cv_gap_f1.min():.4f} a
{e.cv_gap_f1.max():.4f}. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos. Médias próximas com desempenho
baixo podem indicar subajuste, sem provar isso isoladamente. A pequena diferença entre
candidatos precisa ser lida junto com os desvios das dobras, não como superioridade universal.

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.
""")
    financial = table(finance)
    write("avaliacao_final.md", f"""# Avaliação final e veredito

{summary_table}

A classe positiva é inadimplência. FP é um bom pagador previsto como inadimplente;
FN é um inadimplente previsto como bom pagador. A previsão não é, por si só, uma
decisão efetiva de conceder ou recusar crédito.

## Comparação de erros

{cost_verdict}

## Cenário de custos ilustrativos

Para demonstrar o cálculo, usamos custo_FP=R$ 1.000 e custo_FN=R$ 5.000.
Esses valores foram escolhidos para o exercício; não são estimativas apuradas,
não vieram da base e não são apresentados como parâmetros representativos de bancos.
Custo hipotético = FP × custo_FP + FN × custo_FN.

{financial}

A diferença de R$ {benefit_br} vale apenas para essas contagens e essas hipóteses.
Não é economia realizada, receita prevista ou prova de redução efetiva da inadimplência.

## Recomendação

Recomendaria {preferred_name}, configuração {preferred_row.param}, como candidato a um
piloto de apoio à análise neste cenário. Seu F1 de teste é {preferred_row.test_f1_1:.4f}.
Os parâmetros de cada família foram selecionados somente por validação interna.
O custo real, a estabilidade temporal e a disponibilidade das variáveis no momento
da previsão precisam ser definidos antes de uso operacional.

## Importância das variáveis

{top_table}

Essas importâncias foram extraídas do MESMO objeto de árvore que produziu as predições
e a matriz deste relatório. Elas representam redução de impureza usada nos cortes,
não contribuição financeira, efeito causal ou percentual da decisão de um cliente.
A razão empréstimo/renda não representa parcela mensal.
Variáveis correlacionadas e a quantidade de cortes disponíveis podem afetar a importância.

## Limitações

{limitation}

Não há datas para demonstrar validação temporal. Taxa de juros e classificação do empréstimo
podem depender da análise de crédito: a base não esclarece em que momento estão disponíveis.
O estudo não demonstra causalidade nem adequação para decisões automatizadas reais.
""")
    write("correcao_metodologica.md", f"""# Correção metodológica

Uma única implementação em notebooks/pipeline_credito.py é usada pelas entradas do projeto.
Os resultados históricos foram substituídos nos arquivos vigentes e permanecem no Git.

- Limpeza estrutural antes do split; valores de emprego impossíveis viram nulos.
- Razão protegida, calculada sem imputação global.
- Cinco dobras do treino bruto; transformações e balanceamento dentro de cada dobra.
- Mesmos índices balanceados para KNN e árvore.
- Scaler nas contínuas, ajustado no treino balanceado.
- Seleção persistida antes do teste: {selection}.
- Oito comparações de treino/teste; treino avaliado sem duplicações artificiais.
- Importância extraída da árvore realmente avaliada.
- Simulação financeira calculada a partir das contagens geradas, sem números fixados.
- Relatórios e figuras são regenerados a partir dos mesmos resultados.

{limitation}

A auditoria detalhada está em resultados/auditoria_execucao.json. Os testes de regressão
ficam em tests/test_pipeline.py.
""")

    requirement_table = """| Critério técnico | Evidência |
|---|---|
| EDA | Inventário, estatísticas e quatro gráficos interpretados |
| Limpeza | Limpeza registrada e estatísticas do treino para imputação |
| Engenharia | Razão definido, proteção numérica e unidade definida |
| Separação | Split estratificado e auditoria de origem por dobra |
| Balanceamento e escala | Treino preservado; scaler após balanceamento só no KNN |
| Experimentos | Oito linhas em experimentos_corrigidos.csv com treino e teste |
| Overfitting | Treino natural, validação e teste para ambas as famílias |
| Avaliação | Relatórios completos e matrizes dos candidatos selecionados |
| Veredito | Erros, custos hipotéticos e limitações explicitados |
| Documentação | README, dicionário, versões e notebook executável |"""
    for name in ["revisao_final.md", "auditoria_nota10.md"]:
        write(name, f"""# Conferência técnica

{requirement_table}

A tabela aponta evidências; não é garantia de nota. A validação automatizada deve passar
com python3 -m unittest discover -s tests -v. A prova da última execução está em
resultados/auditoria_execucao.json e as versões estão em requirements.txt.

O histórico preserva as correções reais. Não simulamos branches ou etapas retroativamente.

{limitation}
""")
    write("plano.md", """# Estado técnico do projeto

A execução vigente é notebooks/09_validacao_corrigida.py (ou o notebook pipeline_completo.ipynb).
Ela cobre limpeza, engenharia, separação, validação, oito experimentos e avaliação.
O inventário e a EDA são executados pelo lançador antes da modelagem.

As evidências e limitações estão em revisao_final.md, auditoria_nota10.md e avaliacao_final.md.
permanecem fora desta etapa.
""")
    write("roteiro_video.md", f"""# guia de apoio — resultados vigentes

Use como tópicos para ensaio, com duração alvo de 6min30s. Não precisa ler palavra por palavra.
Confira os números após qualquer nova execução.

## 0:00–0:40 — Base e objetivo

Escolhi a base de risco de crédito: 32.581 registros. O alvo 1 indica inadimplência.
O objetivo é comparar KNN e árvore como apoio à análise de risco.
Os CSVs originais foram preservados e conferidos por hash.

## 0:40–1:40 — Descobertas da EDA

21,82% dos registros são da classe inadimplente, então acurácia sozinha é insuficiente.
A mediana da renda é 60.000 na classe em dia e 41.498 na inadimplente.
A razão empréstimo/renda tem medianas de aproximadamente 13% e 24%.
Esses números são associações nesta base, não causas demonstradas.
Mostrar os gráficos de classes, renda e razão.

## 1:40–2:50 — Limpeza e preparação

Removi 165 repetições exatas e cinco idades de 123 ou 144 anos na cópia de trabalho.
Não tratei toda idade acima de 100 como impossível.
Dois tempos de emprego de 123 anos viraram nulos.
Usei mediana para emprego e média para juros, com justificativa nas estatísticas do treino.
A razão definido usa o valor total do empréstimo dividido pela renda anual, vezes 100.
Rendas extremas plausíveis foram mantidas; elas afetam mais as distâncias do KNN.
O scaler ajuda na escala, mas não elimina a influência dos outliers.

## 2:50–4:25 — Parâmetros e overfitting

Testei K=3,5,7,9 e profundidade=3,5,7,None.
Cada dobra aprende seus próprios imputadores e codificação, balanceia apenas seu treino
e ajusta a escala do KNN depois do balanceamento. A árvore não recebe escala.
O treino natural é usado para medir desempenho, sem repetições artificiais.
Mostrar as duas curvas e comentar os gaps realmente observados nas tabelas.
O melhor K foi {selection['KNN']} e a profundidade selecionada foi {selection['Tree']}.
As escolhas foram persistidas antes das comparações de teste.
A diferença pequena entre médias não significa que um modelo sempre vencerá em outras amostras.

## 4:25–6:10 — Erros e veredito

{summary_table}

Mostrar as matrizes. FP significa bom pagador classificado como risco; FN significa
inadimplente não detectado. Esses erros têm consequências distintas para o banco.
{cost_verdict}

Com os custos puramente ilustrativos de R$ 1.000 por FP e R$ 5.000 por FN, a diferença
é R$ {benefit_br}. É uma comparação hipotética, não dinheiro efetivamente economizado.
Minha recomendação neste cenário é {preferred_name}, configuração {preferred_row.param}.

## 6:10–6:30 — Limitações

O teste foi consultado durante o desenvolvimento anterior. A validação atual corrige
o vazamento, mas precisamos de outra base independente para uma confirmação externa.
Também precisamos esclarecer quando juros e classificação estariam disponíveis.
""")
    readme = f"""# Risco de crédito: KNN e Árvore de Decisão

estudo de risco de crédito de Machine Learning e Visão Computacional — projeto.
Alvo: loan_status=1 indica inadimplência; 0 indica pagamento em dia, conforme o problema.

O problema de negócio é apoiar a avaliação de risco de crédito: deixar passar um
inadimplente pode gerar perda do empréstimo, enquanto recusar um bom pagador pode
causar perda de receita e de relacionamento. Comparamos os dois tipos de erro.

## Resumo executivo

{summary_table}

Na base original, 21,82% dos registros são inadimplentes. A renda e a razão empréstimo/renda
apresentam distribuições diferentes entre classes. Foram removidas 165 repetições exatas,
excluídas cinco idades de 123 ou 144 anos e invalidados dois tempos de emprego de 123 anos.
A cópia de trabalho tem {len(data)} registros. A coluna definido usa valor do empréstimo
dividido pela renda anual e multiplicado por 100; não representa parcela mensal.

O candidato recomendado no cenário ilustrativo é {preferred_name}, configuração
{preferred_row.param}. {cost_verdict}

## Reprodução

Ambiente verificado: Python 3.14.6. As versões usadas estão em requirements.txt.
Na raiz do projeto:

~~~sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 notebooks/09_validacao_corrigida.py
python3 -m unittest discover -s tests -v
~~~

Alternativa: iniciar JupyterLab e executar todas as células de
notebooks/pipeline_completo.ipynb. Ele realmente chama o pipeline e regenera os resultados;
não depende de matrizes ou figuras previamente calculadas.

A entrada 09 executa EDA e o pipeline canônico em notebooks/pipeline_credito.py.
As entradas 03 a 08 e 10 foram atualizadas para a implementação vigente;
não é preciso executá-las em sequência. A avaliação não usa os scripts antigos do histórico.

## Arquivos e leitura

{(docs / 'dicionario_dados.md').read_text().split('## Resumo do inventário inicial')[0].replace('# Dicionário de dados — base de crédito', '### Dicionário de dados')}

- documentacao/dicionario_dados.md: significado, unidade e papel de cada coluna.
- documentacao/data_prep.md: decisões e estatísticas do treino.
- documentacao/experimentos_knn.md e experimentos_arvore.md: comparação de complexidade.
- documentacao/avaliacao_final.md: erros, custos hipotéticos e interpretação.
- resultados/experimentos_corrigidos.csv: treino, validação e teste das oito configurações.
- resultados/parametros_selecionados.json: seleção anterior às predições de teste.
- resultados/auditoria_execucao.json: origem das partições, preservação e versões.
- resultados/avaliacao_final/: relatórios, predições, matrizes e importância da árvore avaliada.
- tests/test_pipeline.py: provas de ausência de vazamento e consistência dos artefatos.

Os relatórios identificados como gerados são mantidos por notebooks/relatorios_credito.py.
Alterações de interpretação devem entrar nesse gerador para sobreviver à reexecução.
O dicionário e este README apresentam os resultados vigentes; resultados anteriores permanecem no Git.

## Dados originais

Os dois CSVs da raiz são somente leitura para o pipeline. Os hashes são conferidos antes
e depois. Apenas a base de crédito entra na modelagem. Saídas ficam em dados_derivados/,
resultados/ e documentacao/. O índice de origem não entra como preditor.

Fonte disponibilizada no problema:
[base de crédito](https://drive.google.com/file/d/12vm4oQEeH7ZqB6glXEPpkc5V91lQy0mk/view).

## Limitações

{limitation}

As classes não possuem datas para validação temporal; a disponibilidade prévia de juros
e classificação de risco precisa ser confirmada. Custos monetários são hipóteses do exercício.
Importância das variáveis não comprova causalidade. Não há garantia de desempenho futuro.

"""
    (root / "README.md").write_text(readme, encoding="utf-8")
