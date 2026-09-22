# Engenharia de atributos

A fórmula definido é (loan_amnt / person_income) * 100. Ela relaciona o valor total
do empréstimo à renda anual informada; não mede parcela mensal nem comprometimento mensal.

Ambos os operandos são verificados quanto a nulos, infinitos e valores não positivos
antes da divisão. O cálculo mascarado não executa divisões inválidas.
Nesta base não há operandos inválidos: 32411 razões finitas e
0 nulos. Se houver operandos inválidos em outra base, a etapa interrompe
em vez de imputar globalmente; eventual imputação deve ocorrer somente no treino.

loan_percent_income está em proporção; comprometimento_renda está em percentual.
A diferença mediana entre as duas, na mesma unidade, é
0.2500 ponto percentual.
98.68% das linhas diferem em até 0,5001 ponto.
A versão original é retirada dos preditores para evitar peso redundante no KNN.
Não se trata de colinearidade estrita comprovada.
