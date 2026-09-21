# Feature engineering: comprometimento da renda

## Regra aplicada

Foi criada `comprometimento_renda` com `(loan_amnt / person_income) * 100`. O cálculo só ocorre quando `person_income` existe e é maior que zero; valores infinitos são convertidos em nulos. Na base derivada, foram produzidos 0 nulos e 32411 valores finitos.

## Comparação com a variável original

`loan_percent_income` parece representar a mesma relação em escala de proporção. Após convertê-la para percentual, a diferença absoluta mediana foi 0.2500 ponto percentual e 11% das linhas ficaram dentro de 0,005 ponto percentual.

Essa semelhança será considerada na preparação dos modelos para evitar peso duplicado à mesma informação. A coluna exigida permanece na base derivada e no dicionário; a seleção final de atributos será registrada depois do split.

## Arquivo gerado

`dados_derivados/credito_com_feature.csv` contém a limpeza estrutural e a coluna calculada. Os CSVs originais foram apenas lidos e tiveram seus hashes conferidos.
