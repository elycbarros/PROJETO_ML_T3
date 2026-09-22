# Limpeza e imputação

Foram removidas 165 repetições exatas, mantendo a primeira ocorrência.
Sem identificador de cliente, igualdade não prova que sejam a mesma pessoa; a opção segue a exigência
da estudo de remover redundâncias e evita casos idênticos nas duas partições.

A exclusão por idade usa uma regra explícita de plausibilidade para este estudo: idade >=120.
Os registros observados tinham [144, 144, 123, 123, 144]; não há idades entre 101 e 119.
Não afirmamos que toda idade acima de 100 seja impossível. As idades extremas repetidas,
sem possibilidade de confirmar a informação na fonte, foram excluídas desta análise.
A regra é uma decisão de qualidade de dados, não uma política de concessão de crédito.
A lista está em resultados/idades_excluidas.csv.

Dois tempos de emprego de 123 anos em pessoas de 21 e 22 anos foram convertidos em nulos
antes de qualquer separação. As outras colunas dessas linhas foram preservadas.
A base ficou com 32411 linhas.

## Estatísticas somente do treino

| index | mean | median | skew | nulos |
| --- | --- | --- | --- | --- |
| person_emp_length | 4.7784 | 4.0000 | 1.2259 | 717 |
| loan_int_rate | 11.0133 | 10.9900 | 0.2014 | 2482 |

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
