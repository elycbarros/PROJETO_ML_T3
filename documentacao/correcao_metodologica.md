# Correção metodológica

A validação foi refeita a partir dos dados brutos do treino. Em cada dobra, imputação, codificação, balanceamento por oversampling da classe minoritária e escalonamento do KNN foram ajustados somente na parte de treino. A validação manteve sua distribuição original.

O oversampling preserva todos os registros da classe majoritária e acrescenta somente cópias da classe minoritária. Os dois registros com tempo de emprego maior que a idade foram convertidos em nulos e entram na imputação do treino.

Parâmetros escolhidos exclusivamente pela média de F1 da classe 1 na validação: KNN=3; Árvore=7. Resultados corrigidos: `resultados/validacao_corrigida.csv` e `resultados/avaliacao_corrigida.csv`. Os resultados anteriores ficam superados.
