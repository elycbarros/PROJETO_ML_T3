# Experimentos do KNN

Foram avaliados quatro valores de `n_neighbors`: 3, 5, 7 e 9. O treino foi balanceado; o teste manteve a distribuição original. A seleção foi orientada pela média do F1 da classe 1 em validação estratificada de cinco partes no treino. O teste ficou reservado para comparação final.

O melhor candidato pela validação interna foi **K = 3**. A tabela completa está em `resultados/knn_experimentos.csv`.

O `gap_f1` é a diferença entre o F1 da classe 1 no treino e no teste. Gap alto indica ajuste maior ao treino do que aos dados não vistos. A decisão final também considerará recall, precisão, F1 e matriz de confusão.

Os quatro modelos usaram as mesmas transformações; somente `n_neighbors` mudou. Os CSVs originais foram apenas lidos e tiveram os hashes conferidos.
