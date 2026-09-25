# EDA e preparação dos dados

Gerado por `notebooks/03_executar_pipeline.py` (ver "Reprodução" no README para a ordem completa).

## 1. Análise exploratória (EDA)

### Distribuição do alvo

A inadimplência representa 21,82% dos registros. Há diferença relevante entre as classes; o split deverá preservar a proporção e a avaliação usará recall, precisão e F1 além da acurácia. O balanceamento, quando aplicado, ficará restrito ao treino.

### Renda anual

A mediana da renda é 60.000 para contratos em dia e 41.498 para contratos inadimplentes. O histograma usa intervalos comuns e eixo logarítmico, sem remover registros. Cada classe é normalizada separadamente; a altura mostra sua fração no intervalo. Valores altos não demonstram erro por si sós.

### Comprometimento da renda

As medianas de `loan_percent_income` são 0,130 para a classe 0 e 0,240 para a classe 1. A correlação de Pearson com o alvo é 0,379; isso é associação descritiva, não causalidade.

### Outliers: renda e valor do empréstimo

O critério de referência é o limite superior do boxplot (Q3 + 1,5×IQR), usado só para
visualizar a cauda, não como regra de exclusão. Por esse critério, 1.484
registros de `person_income` ficam acima de 140.250 e
1.689 registros de `loan_amnt` ficam acima de
23.000. O gráfico 06 mostra essas caudas por status, em escala
logarítmica. A decisão (seção 2 abaixo) é manter esses valores: são extremos raros e
plausíveis (rendas e empréstimos altos existem), não erros de digitação como as idades de
123/144 anos. Essa manutenção pesa mais no KNN, sensível a distâncias euclidianas mesmo após
o StandardScaler, do que na árvore, que corta por limiar e é robusta à magnitude dos valores.

### Valores ausentes

`person_emp_length` tem 895 valores ausentes e `loan_int_rate` tem 3.116. A mediana será usada para o tempo de emprego porque a distribuição é assimétrica; a média será usada para a taxa porque média e mediana são próximas. Esses valores são aprendidos somente no treino de cada dobra, depois do split.

### Correlações e próximos passos

A correlação entre `loan_status` e `loan_int_rate` é 0,335. Correlação não determina exclusão automática nem causalidade. A preparação (seções 2 a 4) remove repetições exatas e idades de 123/144 anos, invalida dois tempos de emprego impossíveis, mantém rendas extremas plausíveis, substitui a razão redundante pela coluna exigida e aprende imputadores somente no treino. Esta EDA descreve toda a base; não é uma análise cega de holdout.

As figuras ficam em `resultados/graficos_eda/`. O gráfico 05 resume os nulos observados antes da imputação e o gráfico 06 mostra os outliers de renda e valor do empréstimo por status. Os CSVs foram apenas lidos e tiveram seus hashes conferidos antes e depois da execução.

## 2. Limpeza e imputação

Foram removidas 165 repetições exatas, mantendo a primeira ocorrência.
Sem identificador de cliente, igualdade não prova que sejam a mesma pessoa; a opção segue a exigência
de remover redundâncias e evita casos idênticos nas duas partições.

A exclusão por idade usa uma regra explícita de plausibilidade para este estudo: idade >=120.
Os registros observados tinham [144, 144, 123, 123, 144]; não há idades entre 101 e 119.
Não afirmamos que toda idade acima de 100 seja impossível. As idades extremas repetidas,
sem possibilidade de confirmar a informação na fonte, foram excluídas desta análise —
diferente da renda e do valor do empréstimo (seção 1), mantidos por serem extremos plausíveis,
não erros de digitação. A regra é uma decisão de qualidade de dados, não uma política de
concessão de crédito. A lista está em resultados/idades_excluidas.csv.

Dois tempos de emprego de 123 anos em pessoas de 21 e 22 anos foram convertidos em nulos
antes de qualquer separação. As outras colunas dessas linhas foram preservadas.
A base ficou com 32411 linhas.

### Estatísticas somente do treino

| Variável | Média | Mediana | Assimetria | Nulos |
| --- | --- | --- | --- | --- |
| person_emp_length | 4,7784 | 4,0000 | 1,2259 | 717 |
| loan_int_rate | 11,0133 | 10,9900 | 0,2014 | 2.482 |

Tempo de emprego: mediana, porque a cauda direita permanece após corrigir os erros;
a mediana é menos influenciada por valores altos. Taxa de juros: média, porque média e
mediana são próximas e a assimetria é pequena. Essas são escolhas prévias à seleção
dos modelos. Cada dobra aprende seus próprios valores; o ajuste final usa todo o treino.
As demais numéricas têm mediana como regra de contingência, mas não apresentam nulos nesta base.

Valores extremos podem deslocar as distâncias do KNN mesmo após StandardScaler, que não
é um tratamento robusto de outliers. A árvore dispensa escala e é menos sensível à magnitude,
mas ainda pode aprender cortes inadequados com registros errados. Essa limitação é registrada.

Os CSVs originais são verificados por hash. A rastreabilidade é armazenada separadamente
em dados_derivados/origens.csv e nunca entra nos preditores.

## 3. Engenharia de atributos

A variável derivada é (loan_amnt / person_income) * 100. Ela relaciona o valor total
do empréstimo à renda anual informada; não mede parcela mensal nem comprometimento mensal.

Ambos os operandos são verificados quanto a nulos, infinitos e valores não positivos
antes da divisão. O cálculo mascarado não executa divisões inválidas.
Nesta base não há operandos inválidos: 32411 razões finitas e
0 nulos. Se houver operandos inválidos em outra base, a etapa interrompe
em vez de imputar globalmente; eventual imputação deve ocorrer somente no treino.

loan_percent_income está em proporção; comprometimento_renda está em percentual.
A diferença mediana entre as duas, na mesma unidade, é
0,2500 ponto percentual.
98,68% das linhas diferem em até 0,5001 ponto.
A versão original é retirada dos preditores para evitar peso redundante no KNN.
Não se trata de colinearidade estrita comprovada.

## 4. Separação, balanceamento e escalonamento

Split 80/20 com stratify=y e random_state=42: 25928 linhas de treino e
6483 de teste. A identidade é o número da linha no CSV original (base zero).
Os índices e as repetições estão em resultados/indices_split.csv e
resultados/indices_treino_balanceado.csv.

Imputação e One-Hot Encoding são ajustados somente no treino de cada dobra.
A validação recebe apenas transform. O balanceamento usa Random Over-Sampling: mantém
cada linha original de treino e acrescenta cópias aleatórias, com reposição, apenas da
classe minoritária até igualar as contagens.

**Escolha da técnica de balanceamento.** Comparamos Random Over-Sampling, SMOTE e
Random Under Sampling. Optamos pelo Random
Over-Sampling por dois motivos, cada um comparado à técnica sugerida correspondente:
frente ao Random Under Sampling, o Random Over-Sampling preserva 100% das linhas
originais da classe majoritária no treino — o undersampling descartaria linhas reais só
para igualar as contagens; frente ao SMOTE, o Random Over-Sampling nunca gera pontos
sintéticos interpolados no espaço de atributos — toda linha do treino balanceado,
original ou repetida, é uma observação real da base, o que facilita a auditoria de
rastreabilidade (resultados/indices_treino_balanceado.csv) e evita introduzir
combinações de atributos que não ocorreram de fato. Não é desconhecimento das técnicas
comparadas; a escolha prioriza preservação e rastreabilidade. O ajuste final tem
40514 linhas balanceadas, preservando todas as 25928
originais do treino. As duas famílias recebem exatamente os mesmos índices balanceados.

StandardScaler é ajustado depois do balanceamento, exclusivamente para o KNN. Somente
renda, tempo de emprego, valor do empréstimo, taxa de juros e comprometimento_renda são
escalonados. Idade e duração do histórico, registradas em anos inteiros, são tratadas como
discretas e ficam sem escala; as dummies também não são escalonadas. A árvore usa todas as
variáveis sem escala: seus cortes são baseados em limiares por variável, monotônicos e
independentes de mudanças de escala, então escalonar não mudaria o modelo, só adicionaria
uma etapa sem efeito.

Nenhum parâmetro estatístico é aprendido no teste. Há auditoria de sobreposição de origem
e de preservação dos registros nas cinco dobras e no ajuste final.

### Limitação conhecida

O teste só é usado depois que os hiperparâmetros são escolhidos por validação cruzada (5 dobras, só no treino). Como a base completa foi examinada durante o desenvolvimento deste projeto, o teste não tem a independência de uma amostra nunca vista antes; a seleção, porém, não consulta o teste em nenhuma etapa.
