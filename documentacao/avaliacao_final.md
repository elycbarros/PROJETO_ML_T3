# Avaliação final e veredito

| model | param | test_accuracy | test_precision_1 | test_recall_1 | test_f1_1 | fp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | 9 | 0.7861 | 0.5075 | 0.7433 | 0.6031 | 1023 | 364 |
| Tree | 7 | 0.9028 | 0.8054 | 0.7327 | 0.7674 | 251 | 379 |

A classe positiva é inadimplência. FP é um bom pagador previsto como inadimplente;
FN é um inadimplente previsto como bom pagador. A previsão não é, por si só, uma
decisão efetiva de conceder ou recusar crédito.

## Comparação de erros

A árvore tem menor custo se custo_FN/custo_FP for menor que 51.467; no ponto há empate. Na relação oposta, o KNN tem menor custo.

A árvore trocou 772 falsos positivos a menos por 15 falsos negativos a mais em relação ao
KNN. Assim, a árvore é preferível enquanto um falso negativo custar menos de 51,467 vezes
um falso positivo. Se essa relação de custos for maior, o KNN passa a ter menor custo.

## Cenário de custos ilustrativos

Para demonstrar o cálculo, usamos custo_FP=R$ 1.000 e custo_FN=R$ 5.000.
Esses valores foram escolhidos para o exercício; não são estimativas apuradas,
não vieram da base e não são apresentados como parâmetros representativos de bancos.
Custo hipotético = FP × custo_FP + FN × custo_FN.

| model | param | fp | fn | custo_fp_hipotetico | custo_fn_hipotetico | custo_total_hipotetico |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | 9 | 1023 | 364 | 1023000 | 1820000 | 2843000 |
| Tree | 7 | 251 | 379 | 251000 | 1895000 | 2146000 |

A diferença de R$ 697.000,00 vale apenas para essas contagens e essas hipóteses.
Não é economia realizada, receita prevista ou prova de redução efetiva da inadimplência.

## Recomendação

Recomendaria Árvore de Decisão, configuração 7, como candidato a um
piloto de apoio à análise neste cenário. Seu F1 de teste é 0.7674.
Os parâmetros de cada família foram selecionados somente por validação interna.
O custo real, a estabilidade temporal e a disponibilidade das variáveis no momento
da previsão precisam ser definidos antes de uso operacional.

## Importância das variáveis

| feature | importance |
| --- | --- |
| comprometimento_renda | 0.3170 |
| loan_int_rate | 0.2691 |
| person_income | 0.1449 |
| person_home_ownership_RENT | 0.0634 |
| loan_grade_D | 0.0524 |

Essas importâncias foram extraídas do MESMO objeto de árvore que produziu as predições
e a matriz deste relatório. Elas representam redução de impureza usada nos cortes,
não contribuição financeira, efeito causal ou percentual da decisão de um cliente.
A razão empréstimo/renda não representa parcela mensal.
Variáveis correlacionadas e a quantidade de cortes disponíveis podem afetar a importância.

## Limitações

O teste e a base completa já foram consultados durante o desenvolvimento anterior. A correção mantém a semente e a divisão e não usa o teste na seleção atual, mas não recupera a independência de um conjunto externo intocado.

Não há datas para demonstrar validação temporal. Taxa de juros e classificação do empréstimo
podem depender da análise de crédito: a base não esclarece em que momento estão disponíveis.
O estudo não demonstra causalidade nem adequação para decisões automatizadas reais.
