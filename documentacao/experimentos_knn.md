# Experimentos: KNN

Este documento foi regenerado pelo pipeline vigente. Resultados anteriores permanecem no histórico Git e não devem ser usados na análise.

| param | train_f1_1_mean | cv_f1 | cv_std | treino_f1_1 | teste_f1_1 | gap_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | 0.8444 | 0.5925 | 0.0107 | 0.8458 | 0.6018 | 0.2440 |
| 5 | 0.7586 | 0.5832 | 0.0076 | 0.7612 | 0.5931 | 0.1680 |
| 7 | 0.7197 | 0.5898 | 0.0031 | 0.7245 | 0.5985 | 0.1259 |
| 9 | 0.7048 | 0.5984 | 0.0040 | 0.7083 | 0.6031 | 0.1051 |

train_f1_1_mean mede o treino original de cada dobra; cv_f1 mede sua validação.
treino_f1_1 é medido no treino original completo, sem repetições do oversampling.
As métricas de teste de todas as configurações atendem à comparação pedida pelo problema,
mas são calculadas somente depois de persistir os parâmetros selecionados.

A seleção foi 9, com F1 médio de validação 0.5984.
Usamos o maior F1 médio; em empate exato, K maior no KNN e menor profundidade na árvore.

O intervalo observado do gap treino-validação é 0.1064 a
0.2519. Quanto maior a vantagem no treino, maior o indício de ajuste excessivo;
não aplicamos um limite arbitrário para eliminar modelos. Médias próximas com desempenho
baixo podem indicar subajuste, sem provar isso isoladamente. A pequena diferença entre
candidatos precisa ser lida junto com os desvios das dobras, não como superioridade universal.

## Leitura do overfitting

No KNN, K=3 teve F1 médio de treino 0.8444 e F1 de validação 0.5925, um gap de 0.2519. Com K=9, o F1 de treino caiu para 0.7048, mas o F1 de validação subiu para 0.5984 e o gap caiu para 0.1064. Por isso K=9 foi escolhido: ele generalizou melhor entre os quatro valores testados, apesar de a diferença de validação ser pequena.

O gráfico de treino, validação e teste está em resultados/avaliacao_final/.
