# Experimentos: Tree

Este documento foi regenerado pelo pipeline vigente. Resultados anteriores permanecem no histórico Git e não devem ser usados na análise.

| param | train_f1_1_mean | cv_f1 | cv_std | treino_f1_1 | teste_f1_1 | gap_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.7079 | 0.7060 | 0.0093 | 0.7083 | 0.7179 | -0.0096 |
| 5 | 0.7368 | 0.7331 | 0.0073 | 0.7280 | 0.7220 | 0.0060 |
| 7 | 0.7715 | 0.7551 | 0.0117 | 0.7865 | 0.7674 | 0.0191 |
| None | 1.0000 | 0.7479 | 0.0057 | 1.0000 | 0.7574 | 0.2426 |

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do oversampling.
As métricas de teste de todas as configurações atendem à comparação pedida pelo problema,
mas são calculadas somente depois de persistir os parâmetros selecionados.

A seleção foi 7, com F1 médio de validação 0.7551.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é 0.0018 a
0.2521. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos. Médias próximas com desempenho
baixo podem indicar subajuste, sem provar isso isoladamente. A pequena diferença entre
candidatos precisa ser lida junto com os desvios das dobras, não como superioridade universal.

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.
