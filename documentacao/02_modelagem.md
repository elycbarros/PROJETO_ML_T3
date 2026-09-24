# Modelagem: KNN e Árvore de Decisão

Este documento foi regenerado pelo pipeline vigente. Resultados anteriores permanecem no histórico Git e não descrevem os resultados atuais.

Testamos 4 valores de K no KNN (3, 5, 7, 9) e 4 profundidades na árvore (3, 5, 7 e
ilimitada), sempre comparando F1 de treino contra F1 de validação em 5 dobras
estratificadas, para diagnosticar overfitting antes de tocar no teste.

## KNN (n_neighbors)

| param | train_f1_1_mean | cv_f1 | cv_std | treino_f1_1 | teste_f1_1 | gap_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.8444 | 0.5925 | 0.0107 | 0.8458 | 0.6018 | 0.2440 |
| 5 | 0.7586 | 0.5832 | 0.0076 | 0.7612 | 0.5931 | 0.1680 |
| 7 | 0.7197 | 0.5898 | 0.0031 | 0.7245 | 0.5985 | 0.1259 |
| 9 | 0.7048 | 0.5984 | 0.0040 | 0.7083 | 0.6031 | 0.1051 |

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do balanceamento.
As métricas de teste de todas as configurações permitem uma comparação descritiva,
mas são calculadas somente depois de persistir os parâmetros selecionados. A coluna
gap_f1 da tabela é treino_f1_1 menos teste_f1_1 (treino vs. teste); é diferente do
"gap treino-validação" discutido abaixo, que compara train_f1_1_mean com cv_f1
(treino vs. validação, usado para escolher a configuração antes de tocar no teste).

A seleção foi 9, com F1 médio de validação 0.5984.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é 0.1064 a
0.2519. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos.

### Diagnóstico de overfitting

No KNN, K=3 teve F1 médio de treino 0.8444 e F1 de validação 0.5925, um gap de 0.2519. Com K=9, o F1 de treino caiu para 0.7048, mas o F1 de validação subiu para 0.5984 e o gap caiu para 0.1064. Por isso K=9 foi escolhido: ele generalizou melhor entre os quatro valores testados, apesar de a diferença de validação ser pequena.

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.

## Árvore de Decisão (max_depth)

| param | train_f1_1_mean | cv_f1 | cv_std | treino_f1_1 | teste_f1_1 | gap_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.7079 | 0.7060 | 0.0093 | 0.7083 | 0.7179 | -0.0096 |
| 5 | 0.7368 | 0.7331 | 0.0073 | 0.7280 | 0.7220 | 0.0060 |
| 7 | 0.7715 | 0.7551 | 0.0117 | 0.7865 | 0.7674 | 0.0191 |
| None | 1.0000 | 0.7481 | 0.0039 | 1.0000 | 0.7561 | 0.2439 |

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do balanceamento.
As métricas de teste de todas as configurações permitem uma comparação descritiva,
mas são calculadas somente depois de persistir os parâmetros selecionados. A coluna
gap_f1 da tabela é treino_f1_1 menos teste_f1_1 (treino vs. teste); é diferente do
"gap treino-validação" discutido abaixo, que compara train_f1_1_mean com cv_f1
(treino vs. validação, usado para escolher a configuração antes de tocar no teste).

A seleção foi 7, com F1 médio de validação 0.7551.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é 0.0018 a
0.2519. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos.

### Diagnóstico de overfitting

Na árvore sem limite de profundidade, o F1 de treino chegou a 1.0000, enquanto o F1 de validação foi 0.7481; o gap de 0.2519 é o sinal mais claro de memorização. A profundidade 7 manteve F1 de treino 0.7715, obteve o maior F1 de validação (0.7551) e reduziu o gap para 0.0164.

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.
