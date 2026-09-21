# EDA: gráficos e interpretação

## Distribuição do alvo

A inadimplência representa 21.82% dos registros. Há diferença relevante entre as classes; o split deverá preservar a proporção e a avaliação usará recall, precisão e F1 além da acurácia. O balanceamento, quando aplicado, ficará restrito ao treino.

## Renda anual

A mediana da renda é 60,000 para contratos em dia e 41,498 para contratos inadimplentes. O histograma deve ser lido com cautela porque a renda é assimétrica e possui valores muito altos; essa variável será examinada junto dos outliers.

## Comprometimento da renda

As medianas de `loan_percent_income` são 0.130 para a classe 0 e 0.240 para a classe 1. A correlação de Pearson com o alvo é 0.379; isso é associação descritiva, não causalidade.

## Correlações e próximos passos

A correlação entre `loan_status` e `loan_int_rate` é 0.335. Correlação não será justificativa única para excluir colunas. Próximas decisões: investigar caudas de renda, idade e tempo de emprego; comparar `comprometimento_renda` com `loan_percent_income`; e definir imputação após examinar outliers.

As figuras ficam em `resultados/graficos_eda/`. Os CSVs foram apenas lidos e tiveram seus hashes conferidos antes e depois da execução.
