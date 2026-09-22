# Dicionário de dados — base de crédito

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

## Resumo do inventário inicial

- 32.581 linhas e 12 colunas originais.
- 165 linhas duplicadas.
- Nulos em `person_emp_length` (895) e `loan_int_rate` (3.116).
- `loan_status = 0`: 25.473 linhas (78,18%).
- `loan_status = 1`: 7.108 linhas (21,82%).
- A classe positiva é minoritária; recall, precisão e F1 serão analisados além da acurácia.

No inventário, idade máxima de 144 e tempo de emprego máximo de 123 motivaram investigação. Na limpeza vigente, os cinco registros de idade 123/144 são excluídos da cópia e os dois tempos de emprego de 123 anos viram nulos. Valores monetários são apresentados na unidade original: a base não confirma uma moeda. A simulação em reais é somente ilustrativa.

Idade e duração do histórico são tratadas como discretas em anos inteiros e ficam sem escala. As demais numéricas contínuas recebem StandardScaler apenas no KNN. O índice de origem, salvo separadamente para rastreabilidade, nunca é preditor. Tipos e nulos efetivos são gerados em resultados/graficos_eda/inventario.csv.
