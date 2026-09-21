# Separação e preparação sem vazamento

Foi aplicado `train_test_split(test_size=0.20, stratify=y, random_state=42)`. O treino ficou com 25,928 linhas e o teste com 6,483. O teste preservou as classes: {0: 5065, 1: 1418}.

Imputadores, codificadores e escalonadores foram ajustados somente no treino; o teste usa apenas `transform`. O KNN recebeu mediana para numéricas, moda e one-hot para categóricas e `StandardScaler` nas numéricas. A árvore recebeu as mesmas imputações e codificação, sem escalonamento.

O oversampling aleatório foi aplicado somente ao treino, depois da transformação. As classes passaram de {0: 20257, 1: 5671} para {0: 20257, 1: 20257}; o teste não foi reamostrado.

`loan_percent_income` foi retirado deste primeiro conjunto de preditores por representar informação quase duplicada da feature definido. Essa escolha poderá ser revisitada em análise de sensibilidade.
