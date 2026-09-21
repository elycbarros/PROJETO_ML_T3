# Experimentos da Árvore de Decisão

Foram avaliadas quatro configurações de `max_depth`: 3, 5, 7 e `None`. O treino foi balanceado e o teste manteve a distribuição original. A escolha preliminar foi orientada pelo F1 médio da classe 1 em validação estratificada de cinco partes no treino.

O melhor candidato pelo equilíbrio entre validação interna e generalização foi **max_depth = 7**. A tabela completa está em `resultados/arvore_experimentos.csv`. A profundidade `None` teve F1 de treino igual a 1,00 e gap de 0,28, sinal claro de overfitting; por isso foi descartada apesar do F1 alto na validação feita sobre a matriz balanceada.

O `gap_f1` compara o F1 de treino e teste. Uma árvore sem limite de profundidade tende a ter maior capacidade de memorizar os dados; essa hipótese será confrontada com os números.

A árvore foi treinada sem `StandardScaler`: seus cortes são baseados em limiares e não dependem da escala. Os CSVs originais foram apenas lidos e tiveram os hashes conferidos.
