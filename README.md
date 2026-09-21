# Risco de crédito: KNN e Árvore de Decisão

estudo de risco de crédito de Machine Learning e Visão Computacional — projeto.

O objetivo é comparar dois modelos para prever `loan_status`: 1 indica inadimplência e 0 indica pagamento em dia, conforme o problema. A recomendação final considerará os erros de classificação e suas consequências para o banco.

## Estado do trabalho

Inspeção inicial da base. Nenhum modelo foi selecionado e ainda não há resultado de desempenho.

## Dados originais

Os dois CSVs fornecidos ficam na raiz e devem permanecer intactos. Apenas `credit_risk_dataset.csv` será usado na modelagem. Os hashes SHA-256 estão em `documentacao/integridade_originais.json`. Transformações acontecerão em memória; eventuais exportações irão para `dados_derivados/`.

Fonte da base de crédito: https://drive.google.com/file/d/12vm4oQEeH7ZqB6glXEPpkc5V91lQy0mk/view

## Organização

- `notebooks/01_inspecao_inicial.ipynb`: inventário e verificações da base original.
- `documentacao/plano.md`: critérios de conclusão e decisões pendentes.
- `dados_derivados/`: arquivos transformados, quando necessários.
- `resultados/`: tabelas e gráficos produzidos durante o projeto.

## Execução da inspeção

Abrir o notebook no Jupyter com Python e pandas instalados e executar todas as células. Ele localiza a base a partir da raiz do projeto ou da pasta `notebooks`, lê o CSV e confere sua integridade. Não salva alterações na base.

O dicionário completo, as instruções finais de reprodução e o resumo executivo serão adicionados após a preparação e os experimentos.
