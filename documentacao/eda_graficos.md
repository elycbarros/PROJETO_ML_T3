# EDA: gráficos e interpretação

## Distribuição do alvo

A inadimplência representa 21.82% dos registros. Há diferença relevante entre as classes; o split deverá preservar a proporção e a avaliação usará recall, precisão e F1 além da acurácia. O balanceamento, quando aplicado, ficará restrito ao treino.

## Renda anual

A mediana da renda é 60,000 para contratos em dia e 41,498 para contratos inadimplentes. O histograma usa intervalos comuns e eixo logarítmico, sem remover registros. Cada classe é normalizada separadamente; a altura mostra sua fração no intervalo. Valores altos não demonstram erro por si sós.

## Comprometimento da renda

As medianas de `loan_percent_income` são 0.130 para a classe 0 e 0.240 para a classe 1. A correlação de Pearson com o alvo é 0.379; isso é associação descritiva, não causalidade.

## Valores ausentes

`person_emp_length` tem 895 valores ausentes e `loan_int_rate` tem 3,116. A mediana será usada para o tempo de emprego porque a distribuição é assimétrica; a média será usada para a taxa porque média e mediana são próximas. Esses valores são aprendidos somente no treino de cada dobra, depois do split.

## Correlações e próximos passos

A correlação entre `loan_status` e `loan_int_rate` é 0.335. Correlação não determina exclusão automática nem causalidade. A preparação remove repetições exatas e idades de 123/144 anos e invalida dois tempos de emprego impossíveis. Mantém rendas extremas plausíveis, substitui a razão redundante pela coluna exigida e aprende imputadores somente no treino. Esta EDA descreve toda a base; não é uma análise cega de holdout.

As figuras ficam em `resultados/graficos_eda/`. O gráfico 05 resume os nulos observados antes da imputação. Os CSVs foram apenas lidos e tiveram seus hashes conferidos antes e depois da execução.
