# Avaliação final e veredito de negócio

Este relatório registra a avaliação anterior e está **superado** pela validação corrigida em `resultados/avaliacao_corrigida.csv`. A execução vigente é `notebooks/09_validacao_corrigida.py`, que ajusta preparação e balanceamento dentro de cada dobra.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K=3 | 0.825 | 0.579 | 0.733 | 0.647 | 756 | 379 |
| Árvore depth=7 | 0.884 | 0.721 | 0.767 | 0.743 | 421 | 331 |

Um falso positivo trata como inadimplente quem pagaria em dia, podendo gerar recusa ou condição pior para um bom cliente. Um falso negativo libera crédito a quem inadimplirá. Assumo que o segundo erro tem maior perda financeira direta, embora custos reais não tenham sido fornecidos.

## Veredito

Recomendo preliminarmente a **Árvore de Decisão com `max_depth=7` para um piloto controlado**. Ela teve F1 e recall superiores no teste corrigido e reduziu os falsos negativos de 379 para 331. O KNN reduziu falsos positivos, então a escolha depende dos custos reais; com esses números, ele só seria preferível se um falso negativo custasse mais de aproximadamente 16,1 vezes um falso positivo. Antes de produção, o banco deve calibrar o limiar com custos reais, validar em uma amostra temporal e acompanhar desempenho por segmento.

Relatórios e matrizes estão em `resultados/avaliacao_final/`. Os CSVs originais foram apenas lidos e tiveram hashes conferidos.
