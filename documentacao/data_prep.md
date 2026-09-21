# Data prep: duplicatas, nulos e consistência

## Duplicatas

Foram encontradas 165 linhas inteiramente duplicadas em 32,581 registros. Elas foram removidas somente na cópia derivada; o CSV original permanece intacto.

## Nulos

Continuam ausentes valores em `person_emp_length` e `loan_int_rate`. A política é imputar a mediana dentro do pipeline de treino: são variáveis numéricas com assimetria e valores extremos. O valor será aprendido apenas no treino e aplicado ao teste sem novo ajuste.

## Consistência

| Regra | Quantidade |
|---|---:|
| idade maior que 100 | 5 |
| tempo de emprego maior que 80 anos | 2 |
| tempo de emprego maior ou igual à idade | 2 |
| renda não positiva | 0 |
| empréstimo não positivo | 0 |
| histórico de crédito maior que a idade | 0 |

As 5 linhas com idade acima de 100 foram excluídas da cópia derivada porque são inconsistentes com a população de solicitantes. Não foram aplicados limites genéricos a renda ou empréstimo: não há valores não positivos e valores altos ainda podem representar casos reais.

## Arquivo derivado

`dados_derivados/credito_sem_duplicatas_e_idades_invalidas.csv` contém a limpeza estrutural e mantém os nulos. A imputação será ajustada somente dentro do treino. A coluna `comprometimento_renda` será criada na próxima etapa.
