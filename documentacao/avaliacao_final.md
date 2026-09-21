# Avaliação final e veredito de negócio

Foram comparados KNN (`K=3`) e Árvore de Decisão (`max_depth=7`) no teste original, sem reamostragem.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K=3 | 0.812 | 0.553 | 0.731 | 0.630 | 839 | 381 |
| Árvore depth=7 | 0.909 | 0.843 | 0.717 | 0.775 | 189 | 402 |

Um falso positivo trata como inadimplente quem pagaria em dia, podendo gerar recusa ou condição pior para um bom cliente. Um falso negativo libera crédito a quem inadimplirá. Assumo que o segundo erro tem maior perda financeira direta, embora custos reais não tenham sido fornecidos.

## Veredito

Recomendo a **Árvore de Decisão com `max_depth=7` para um piloto controlado**. Ela teve F1 e precisão superiores aos do KNN no teste e manteve gap de generalização menor que a árvore sem limite. Antes de produção, o banco deve calibrar o limiar com custos reais, validar em uma amostra temporal e acompanhar desempenho por segmento.

Relatórios e matrizes estão em `resultados/avaliacao_final/`. Os CSVs originais foram apenas lidos e tiveram hashes conferidos.
