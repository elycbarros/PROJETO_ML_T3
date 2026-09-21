# Correção metodológica

A validação foi refeita a partir dos dados brutos do treino. Em cada dobra, imputação, codificação, balanceamento por oversampling da classe minoritária e escalonamento do KNN foram ajustados somente na parte de treino. A validação manteve sua distribuição original.

O oversampling preserva todos os registros da classe majoritária e acrescenta somente cópias da classe minoritária. Os dois registros com tempo de emprego maior que a idade foram convertidos em nulos e entram na imputação do treino.

Parâmetros escolhidos exclusivamente pela média de F1 da classe 1 na validação: KNN=3; Árvore=7. Resultados corrigidos: `resultados/validacao_corrigida.csv` e `resultados/avaliacao_corrigida.csv`. Os resultados anteriores ficam superados.

Na avaliação corrigida, o KNN K=3 alcançou F1 0,648, recall 0,735, 758 falsos positivos e 376 falsos negativos. A árvore depth=7 alcançou F1 0,743, recall 0,767, 421 falsos positivos e 331 falsos negativos. A árvore continua sendo a melhor candidata nos critérios agregados. Com esses números, o KNN só seria preferível sob uma hipótese de custo de falso negativo superior a aproximadamente 16,1 vezes o custo de falso positivo.
