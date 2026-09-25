# Avaliação final e veredito de negócio

## Avaliação final

| Modelo | Parâmetro | Acurácia | Precisão (1) | Recall (1) | F1 (1) | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | 9 | 0,7861 | 0,5075 | 0,7433 | 0,6031 | 1.023 | 364 |
| Árvore | 7 | 0,9028 | 0,8054 | 0,7327 | 0,7674 | 251 | 379 |

A classe positiva é inadimplência. FP é um bom pagador previsto como inadimplente;
FN é um inadimplente previsto como bom pagador. A previsão não é, por si só, uma
decisão efetiva de conceder ou recusar crédito.

### Comparação de erros

Em relação ao KNN, a árvore comete 15 FN a mais
e 772 FP a menos. A árvore tem menor custo se custo_FN/custo_FP for menor que 51,467; no ponto há empate. Na relação oposta, o KNN tem menor custo.

### Cenário de custos ilustrativos

Para demonstrar o cálculo, usamos custo_FP=R$ 1.000 e custo_FN=R$ 5.000.
Esses valores são hipóteses ilustrativas; não são estimativas apuradas,
não vieram da base e não são apresentados como parâmetros representativos de bancos.
Custo hipotético = FP × custo_FP + FN × custo_FN.

| Modelo | Parâmetro | FP | FN | Custo FP (R$) | Custo FN (R$) | Custo total (R$) |
| --- | --- | --- | --- | --- | --- | --- |
| KNN | 9 | 1.023 | 364 | 1.023.000 | 1.820.000 | 2.843.000 |
| Árvore | 7 | 251 | 379 | 251.000 | 1.895.000 | 2.146.000 |

A diferença de R$ 697.000,00 vale apenas para essas contagens e essas hipóteses.
Não é economia realizada, receita prevista ou prova de redução efetiva da inadimplência.

### Veredito: qual erro custa mais e qual modelo vai para produção

**O erro mais caro é o falso negativo (FN).** Um inadimplente aprovado como "seguro"
leva ao prejuízo do valor emprestado. Um falso positivo (bom pagador recusado) custa a
receita de juros daquele contrato e o relacionamento com o cliente, mas não o capital.
Por isso, no cenário ilustrativo, um FN custa 5 vezes mais que um FP.

**Veredito: colocaria em produção a Árvore de Decisão (profundidade máxima 7).** Mesmo com o
FN sendo o erro mais caro, ela sai mais barata pelos números já mostrados acima. Para o
KNN compensar, a perda de um calote teria de valer dezenas de vezes a margem perdida ao
recusar um bom pagador, o que é pouco plausível: a perda máxima de um FN é o próprio valor
emprestado. Por isso a vantagem da árvore resiste à incerteza sobre os custos reais. Ela
também tem o maior F1 no teste (0,7674) e a maior precisão, o que
reduz recusas injustas de bons pagadores. Os parâmetros foram escolhidos só por validação
interna, antes do teste.

Antes do uso real, o banco deve confirmar os custos efetivos de cada erro e se juros e
classificação de risco estão disponíveis no momento da decisão (ver Limitações).

### Importância das variáveis

| Variável | Importância |
| --- | --- |
| comprometimento_renda | 0,3170 |
| loan_int_rate | 0,2691 |
| person_income | 0,1449 |
| person_home_ownership_RENT | 0,0634 |
| loan_grade_D | 0,0524 |

Essas importâncias foram extraídas do MESMO objeto de árvore que produziu as predições
e a matriz deste relatório. Elas representam redução de impureza usada nos cortes,
não contribuição financeira, efeito causal ou percentual da decisão de um cliente.
A razão empréstimo/renda não representa parcela mensal.
Variáveis correlacionadas e a quantidade de cortes disponíveis podem afetar a importância.

### Limitações

O teste só é usado depois que os hiperparâmetros são escolhidos por validação cruzada (5 dobras, só no treino). Como a base completa foi examinada durante o desenvolvimento deste projeto, o teste não tem a independência de uma amostra nunca vista antes; a seleção, porém, não consulta o teste em nenhuma etapa.

Não há datas para demonstrar validação temporal. Taxa de juros e classificação do empréstimo
podem depender da análise de crédito: a base não esclarece em que momento estão disponíveis.
O estudo não demonstra causalidade nem adequação para decisões automatizadas reais.

## Verificação complementar

Esta análise usa apenas as cinco dobras do conjunto de treino. A divisão de teste, a seleção de KNN/Árvore e o veredito principal permanecem inalterados. O script `notebooks/04_analise_complementar.py` gera esta seção quando executado após o pipeline principal.

| Modelo | F1 médio (classe 1) | Desvio entre dobras | Recall médio | Precisão média |
| --- | ---: | ---: | ---: | ---: |
| Referência aleatória estratificada | 0,2219 | 0,0063 | 0,2239 | 0,2200 |
| Árvore completa (profundidade 7) | 0,7551 | 0,0131 | 0,7329 | 0,7814 |
| Árvore sem grade e juros (profundidade 7) | 0,6327 | 0,0076 | 0,6844 | 0,5888 |

A referência aleatória estratificada preserva aproximadamente a proporção das classes, mas não aprende relações entre atributos e alvo.
A árvore reduzida exclui todas as colunas codificadas de `loan_grade` e `loan_int_rate`; mantém a mesma profundidade, as mesmas dobras e os mesmos índices de balanceamento da árvore completa.
A diferença média de F1 (completa menos reduzida) foi 0,1225.
Isso mede sensibilidade nesta base, não demonstra vazamento por si só. É preciso confirmar quando grade e juros ficam disponíveis no processo real.
A profundidade 7 foi fixada a partir da análise principal; não foi otimizada novamente para a versão reduzida. Portanto, esta comparação é exploratória, e não uma nova seleção de modelo.

Resultados por dobra: `resultados/analise_complementar_dobras.csv`. Resumo: `resultados/analise_complementar_resumo.csv`.

## Nota metodológica

Uma única implementação em notebooks/pipeline_credito.py é usada por todas as entradas do
projeto (notebooks/03_executar_pipeline.py e notebooks/pipeline_completo.ipynb).

- Limpeza estrutural antes do split; valores de emprego impossíveis viram nulos.
- Razão protegida, calculada sem imputação global.
- Cinco dobras do treino bruto; transformações e balanceamento dentro de cada dobra.
- Balanceamento por Random Over-Sampling (reamostragem com reposição da classe minoritária), estritamente no treino.
- Mesmos índices balanceados para KNN e árvore.
- Scaler nas contínuas, ajustado no treino balanceado, exclusivo do KNN.
- Seleção persistida antes do teste: {'KNN': '9', 'Tree': '7'}.
- Oito comparações de treino/teste; treino avaliado sem duplicações artificiais.
- Importância extraída da árvore realmente avaliada.
- Simulação financeira calculada a partir das contagens geradas, sem números fixados.
- Este arquivo é regenerado a partir dos mesmos resultados a cada execução do pipeline.

A limitação de dependência do teste histórico está detalhada na seção "Limitações" acima.
A auditoria detalhada está em resultados/auditoria_execucao.json. Os testes de regressão
ficam em tests/test_pipeline.py.
