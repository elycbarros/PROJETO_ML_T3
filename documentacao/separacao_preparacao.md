# Separação e preparação

Split 80/20 com stratify=y e random_state=42: 25928 linhas de treino e
6483 de teste. A identidade é o número da linha no CSV original (base zero).
Os índices e as repetições estão em resultados/indices_split.csv e
resultados/indices_treino_balanceado.csv.

Imputação e One-Hot Encoding são ajustados somente no treino de cada dobra.
A validação recebe apenas transform. O oversampling mantém cada linha de treino
e acrescenta cópias da classe minoritária. O ajuste final tem 40514 linhas
balanceadas, preservando todas as 25928 originais do treino.
As duas famílias recebem exatamente os mesmos índices balanceados.

StandardScaler é ajustado depois do balanceamento. Somente renda, tempo de emprego,
valor do empréstimo, taxa de juros e comprometimento_renda são escalonados.
Idade e duração do histórico, registradas em anos inteiros, são tratadas como discretas
e ficam sem escala; as dummies também não são escalonadas. Essa decisão atende à
separação pedida entre contínuas e demais atributos, mas deixa as durações em sua
unidade original no cálculo de distância. A árvore usa todas as variáveis sem escala.

Nenhum parâmetro estatístico é aprendido no teste. Há auditoria de sobreposição de origem
e de preservação dos registros nas cinco dobras e no ajuste final.

## Limitação conhecida

O teste e a base completa já foram consultados durante o desenvolvimento anterior. A correção mantém a semente e a divisão e não usa o teste na seleção atual, mas não recupera a independência de um conjunto externo intocado.
