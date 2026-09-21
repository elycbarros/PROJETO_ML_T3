# Avaliação final e veredito de negócio

Foram comparados KNN (`K=3`) e Árvore de Decisão (`max_depth=7`) no teste original (6.483 registros), sem reamostragem, após seleção estrita por validação cruzada sem vazamento.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K=3 | 0.825 | 0.579 | 0.733 | 0.647 | 756 | 379 |
| Árvore depth=7 | 0.884 | 0.721 | 0.767 | 0.743 | 421 | 331 |

Um falso positivo trata como inadimplente quem pagaria em dia, gerando atrito ou recusa injustificada de um bom cliente. Um falso negativo libera crédito a quem inadimplirá, gerando perda financeira direta do principal.

## Veredito

Recomendo a **Árvore de Decisão com `max_depth=7` para um piloto controlado**.
Na validação corrigida, a Árvore superou o KNN em todas as métricas:
- F1-Score superior (0.743 vs 0.647);
- Recall superior (0.767 vs 0.733), capturando mais inadimplentes (331 FN contra 379 do KNN);
- Precisão e acurácia substancialmente maiores, reduzindo os falsos positivos quase pela metade (421 contra 756 do KNN).

Antes de qualquer implantação em produção, o banco deve:
1. Calibrar o limiar de decisão com os custos financeiros monetários reais de FP e FN;
2. Realizar validação fora do tempo (amostra temporal subsequente);
3. Monitorar métricas de disparidade e estabilidade por segmento socioeconômico.

Relatórios e matrizes estão em `resultados/avaliacao_final/`. Os CSVs originais foram apenas lidos e tiveram hashes conferidos.
