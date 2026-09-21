# Risco de crédito: KNN e Árvore de Decisão

estudo de risco de crédito de Machine Learning e Visão Computacional — projeto.

O objetivo é comparar dois modelos para prever `loan_status`: 1 indica inadimplência e 0 indica pagamento em dia, conforme o problema. A recomendação final considerará os erros de classificação e suas consequências para o banco.

## Resumo executivo

A base possui 32.581 registros originais, dos quais 21,82% representam inadimplência. A preparação removeu 165 duplicatas e 5 registros com idade acima de 100 anos somente em arquivos derivados. Os nulos de tempo de emprego e taxa de juros foram imputados por mediana dentro do treino.

Após a validação corrigida, o KNN com `K=3` teve F1 de 0,648 no teste e a Árvore com `max_depth=7` teve F1 de 0,743, acurácia de 0,884 e precisão de 0,721. O modelo recomendado preliminarmente é a Árvore de Decisão. A decisão deve ser recalibrada com custos reais de falsos positivos e falsos negativos antes de qualquer uso operacional.

## Dados originais

Os dois CSVs fornecidos ficam na raiz e devem permanecer intactos. Apenas `credit_risk_dataset.csv` será usado na modelagem. Os hashes SHA-256 estão em `documentacao/integridade_originais.json`. Transformações acontecerão em memória; eventuais exportações irão para `dados_derivados/`.

Fonte da base de crédito: https://drive.google.com/file/d/12vm4oQEeH7ZqB6glXEPpkc5V91lQy0mk/view

## Organização

- `notebooks/01_inspecao_inicial.ipynb`: inventário e verificações da base original.
- `notebooks/02_eda_graficos.py`: gráficos e interpretação da EDA.
- `notebooks/03_data_prep_diagnostico.py`: duplicatas, nulos e consistência.
- `notebooks/04_feature_engineering.py`: criação da coluna definido.
- `notebooks/05_separacao_preparacao.py`: split, imputação, encoding, balanceamento e escala.
- `notebooks/06_experimentos_knn.py`: quatro valores de K.
- `notebooks/07_experimentos_arvore.py`: quatro profundidades da árvore.
- `notebooks/08_avaliacao_final.py`: relatórios, matrizes e veredito.
- `notebooks/09_validacao_corrigida.py`: reexecução sem vazamento na validação cruzada; este é o resultado vigente.
- `documentacao/dicionario_dados.md`: dicionário e resumo do inventário da base de crédito.
- `documentacao/eda_graficos.md`: interpretações da análise exploratória.
- `documentacao/data_prep.md`: política de limpeza e consistência.
- `documentacao/feature_engineering.md`: regra e validação da nova coluna.
- `documentacao/separacao_preparacao.md`: controle contra vazamento.
- `documentacao/avaliacao_final.md`: comparação final e recomendação.
- `documentacao/plano.md`: critérios de conclusão e decisões pendentes.
- `dados_derivados/`: arquivos transformados, quando necessários.
- `resultados/`: tabelas e gráficos produzidos durante o projeto.

## Reprodução

Com Python e as dependências instaladas, executar a partir da raiz, nesta ordem:

```text
# Abrir notebooks/01_inspecao_inicial.ipynb e executar as células
python notebooks/02_eda_graficos.py
python notebooks/03_data_prep_diagnostico.py
python notebooks/04_feature_engineering.py
python notebooks/05_separacao_preparacao.py
python notebooks/06_experimentos_knn.py
python notebooks/07_experimentos_arvore.py
python notebooks/08_avaliacao_final.py
python notebooks/09_validacao_corrigida.py
```

Os scripts sempre conferem os hashes dos arquivos de entrada. Os CSVs originais não são sobrescritos; resultados transformados ficam em `dados_derivados/` e tabelas, gráficos e relatórios em `resultados/` e `documentacao/`. Os resultados de `06_experimentos_knn.py` e `07_experimentos_arvore.py` são históricos; `09_validacao_corrigida.py` é a execução metodologicamente válida.

## Limitações

O teste é uma divisão aleatória, não uma validação temporal. A estudo não fornece custo financeiro por erro, portanto o veredito usa a hipótese explícita de que liberar crédito a um inadimplente tende a ser mais caro. A recomendação é um resultado acadêmico e precisa de validação adicional antes de produção.
