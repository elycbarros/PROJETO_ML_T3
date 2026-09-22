# Correção metodológica

Uma única implementação em notebooks/pipeline_credito.py é usada pelas entradas do projeto.
Os resultados históricos foram substituídos nos arquivos vigentes e permanecem no Git.

- Limpeza estrutural antes do split; valores de emprego impossíveis viram nulos.
- Razão protegida, calculada sem imputação global.
- Cinco dobras do treino bruto; transformações e balanceamento dentro de cada dobra.
- Mesmos índices balanceados para KNN e árvore.
- Scaler nas contínuas, ajustado no treino balanceado.
- Seleção persistida antes do teste: {'KNN': '9', 'Tree': '7'}.
- Oito comparações de treino/teste; treino avaliado sem duplicações artificiais.
- Importância extraída da árvore realmente avaliada.
- Simulação financeira calculada a partir das contagens geradas, sem números fixados.
- Relatórios e figuras são regenerados a partir dos mesmos resultados.

O teste e a base completa já foram consultados durante o desenvolvimento anterior. A correção mantém a semente e a divisão e não usa o teste na seleção atual, mas não recupera a independência de um conjunto externo intocado.

A auditoria detalhada está em resultados/auditoria_execucao.json. Os testes de regressão
ficam em tests/test_pipeline.py.
