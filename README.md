# Risco de crédito: KNN e Árvore de Decisão

estudo de risco de crédito de Machine Learning e Visão Computacional — projeto.
Alvo: loan_status=1 indica inadimplência; 0 indica pagamento em dia, conforme o problema.

O problema de negócio é apoiar a avaliação de risco de crédito: deixar passar um
inadimplente pode gerar perda do empréstimo, enquanto recusar um bom pagador pode
causar perda de receita e de relacionamento. Comparamos os dois tipos de erro.

## Resumo executivo

| model | param | test_accuracy | test_precision_1 | test_recall_1 | test_f1_1 | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | 9 | 0.7861 | 0.5075 | 0.7433 | 0.6031 | 1023 | 364 |
| Tree | 7 | 0.9028 | 0.8054 | 0.7327 | 0.7674 | 251 | 379 |

Na base original, 21,82% dos registros são inadimplentes. A renda e a razão empréstimo/renda
apresentam distribuições diferentes entre classes. Foram removidas 165 repetições exatas,
excluídas cinco idades de 123 ou 144 anos e invalidados dois tempos de emprego de 123 anos.
A cópia de trabalho tem 32411 registros. A coluna definido usa valor do empréstimo
dividido pela renda anual e multiplicado por 100; não representa parcela mensal.

O candidato recomendado no cenário ilustrativo é Árvore de Decisão, configuração
7. A árvore tem menor custo se custo_FN/custo_FP for menor que 51.467; no ponto há empate. Na relação oposta, o KNN tem menor custo.

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

### Dicionário de dados

| Coluna | Tipo | Papel | Descrição operacional |
|---|---|---|---|
| `person_age` | inteiro | preditora | Idade informada da pessoa solicitante. |
| `person_income` | inteiro | preditora | Renda anual informada. |
| `person_home_ownership` | categórica | preditora | Situação de moradia informada. |
| `person_emp_length` | decimal | preditora | Tempo de emprego informado, em anos. |
| `loan_intent` | categórica | preditora | Finalidade declarada do empréstimo. |
| `loan_grade` | categórica | preditora | Classificação de risco do empréstimo na base. |
| `loan_amnt` | inteiro | preditora | Valor solicitado para o empréstimo. |
| `loan_int_rate` | decimal | preditora | Taxa de juros do empréstimo. |
| `loan_status` | inteiro | alvo | 0: pagamento em dia; 1: inadimplência. |
| `loan_percent_income` | decimal | excluída dos preditores | Razão empréstimo/renda em proporção: 0,13 significa 13%; redundante com a feature calculada. |
| `cb_person_default_on_file` | categórica | preditora | Indicador de inadimplência anterior. |
| `cb_person_cred_hist_length` | inteiro | preditora | Comprimento do histórico de crédito, em anos. |
| `comprometimento_renda` | decimal | preditora calculada | `(loan_amnt / person_income) * 100`, em percentual; não representa parcela mensal. |



- documentacao/dicionario_dados.md: significado, unidade e papel de cada coluna.
- documentacao/data_prep.md: decisões e estatísticas do treino.
- documentacao/experimentos_knn.md e experimentos_arvore.md: comparação de complexidade.
- documentacao/avaliacao_final.md: erros, custos hipotéticos e interpretação.
- documentacao/rastreabilidade_requisitos.md: exigência do problema, evidência e fala de apoio.
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

O teste e a base completa já foram consultados durante o desenvolvimento anterior. A correção mantém a semente e a divisão e não usa o teste na seleção atual, mas não recupera a independência de um conjunto externo intocado.

As classes não possuem datas para validação temporal; a disponibilidade prévia de juros
e classificação de risco precisa ser confirmada. Custos monetários são hipóteses do exercício.
Importância das variáveis não comprova causalidade. Não há garantia de desempenho futuro.

