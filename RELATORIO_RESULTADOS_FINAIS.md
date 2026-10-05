# Relatório Técnico de Resultados Finais: Classificação de Risco de Crédito

**Projeto:** Comparativo de Modelos Preditivos — *K-Nearest Neighbors (KNN)* vs. *Árvore de Decisão (Decision Tree)*  
**Autor:** Ely do Carmo Barros  
**Data:** Outubro de 2026  
**Status do Projeto:** Concluído e Auditado  
**Ambiente de Execução:** Python 3.14.7 (`scikit-learn` 1.9.0, `pandas` 3.0.5, `numpy` 2.4.2, `matplotlib` 3.11.1)

---

## Sumário Executivo

Este relatório consolida os resultados finais do projeto de modelagem preditiva para concessão e risco de crédito, desenvolvido sobre o conjunto de dados `credit_risk_dataset.csv`. O objetivo central consistiu em construir, auditar e comparar pipelines de aprendizado de máquina para prever a inadimplência de proponentes de crédito (`loan_status = 1`), mitigando perdas financeiras decorrentes de inadimplência e preservando receitas operacionais advindas de bons pagadores.

Com base na validação cruzada em 5 dobras e na avaliação definitiva sobre a partição de teste (com 6.483 registros isolados), o modelo recomendado para homologação e entrada em produção é a **Árvore de Decisão com Profundidade Máxima 7 (`max_depth=7`)**.

### Destaques dos Resultados
* **Desempenho Geral:** A Árvore de Decisão atingiu **90,28% de acurácia** e **F1-Score de 0,7674** na classe positiva (inadimplência), superando com folga o KNN ($K=9$), que registrou **78,61% de acurácia** e **F1-Score de 0,6031**.
* **Equilíbrio de Erros:** A Árvore cometeu apenas 15 Falsos Negativos adicionais em relação ao KNN (379 vs. 364), porém evitou expressivos **772 Falsos Positivos** (251 vs. 1.023). O KNN exige recusar 1.023 bons clientes para detectar 15 inadimplentes a mais, operando com precisão de apenas 50,75% contra 80,54% da Árvore.
* **Impacto Econômico:** Em simulação financeira com parâmetros ilustrativos (custo de FN = R$ 5.000 e custo de FP = R$ 1.000), a Árvore de Decisão gera um custo total de erros de **R$ 2.146.000,00**, contra **R$ 2.843.000,00** do KNN — uma economia estimada de **R$ 697.000,00** (~24,5% de redução de custo).
* **Robustez Operacional:** A Árvore demonstrou menor dispersão entre dobras ($\sigma_{F1} = 0,0117$), interpretabilidade direta das regras de negócio e independência de padronização numérica, além de menor latência de inferência em produção.

---

## 1. Visão Geral da Base e Preparação dos Dados

### 1.1 Estatísticas da Base de Dados
A base de risco de crédito foi selecionada em detrimento da base alternativa (e-commerce) devido ao volume, desafios reais de qualidade cadastral e alinhamento com problemas de custo assimétrico.

| Métrica | Base Bruta | Base Saneada | Observação / Ação de Saneamento |
| :--- | :---: | :---: | :--- |
| **Total de Registros** | 32.581 | **32.411** | Remoção de 165 duplicatas exatas e 5 inconsistências de idade |
| **Linhas Duplicadas** | 165 | 0 | Identificadas e removidas integralmente antes da partição |
| **Idades Anômalas** | 5 | 0 | 5 registros com idades de 123 e 144 anos foram excluídos |
| **Emprego Anômalo** | 2 | 0 | 2 registros com 123 anos de serviço convertidos em nulos (`NaN`) |
| **Proporção da Classe Alvo** | 21,82% (1) | 21,87% (1) | Desbalanceamento severo: ~78,13% em dia vs. 21,87% inadimplentes |

### 1.2 Engenharia de Atributos e Prevenção de Vazamento (*Data Leakage*)
* **Comprometimento de Renda (`comprometimento_renda`):** Calculada como `(loan_amnt / person_income) * 100`. Atributo exigido pelo problema de negócio para medir a pressão da dívida sobre os vencimentos anuais. Identificou-se que o atributo nativo `loan_percent_income` era redundante e este foi excluído do espaço de atributos para evitar colinearidade estrita.
* **Protocolo Antivazamento:**
  1. A divisão Treino/Teste (80/20) foi estratificada e fixada antes de qualquer cálculo de estatísticas.
  2. Imputação de nulos (`person_emp_length` pela mediana; `loan_int_rate` pela média) aprendida estritamente dentro da partição de treino de cada dobra.
  3. Balanceamento via **Random Over-Sampling (ROS)** aplicado unicamente sobre as partições de treino. Os dados de teste e de validação jamais sofreram sobreamostragem.
  4. Padronização (`StandardScaler`) ajustada no treino balanceado e aplicada unicamente ao modelo KNN (a Árvore opera com invariância a escala monotônica).

---

## 2. Validação Cruzada e Otimização de Hiperparâmetros

A seleção dos hiperparâmetros foi conduzida via Validação Cruzada Estratificada em 5 Dobras (*Stratified 5-Fold CV*), utilizando o **F1-Score da Classe Inadimplente (Classe 1)** como métrica diretora para balancear precisão e sensibilidade.

### 2.1 Grade de Experimentos: KNN ($K \in \{3, 5, 7, 9\}$)

| Hiperparâmetro | $F_1$ Validação (Média $\pm$ Desvio) | Acurácia Validação | $F_1$ Treino | Gap de Overfitting ($\Delta F_1$) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **$K = 3$** | 0,5925 $\pm$ 0,0107 | 79,42% | 0,8444 | 0,2519 | Sobreajuste alto |
| **$K = 5$** | 0,5832 $\pm$ 0,0076 | 77,37% | 0,7586 | 0,1754 | Subótimo |
| **$K = 7$** | 0,5898 $\pm$ 0,0031 | 77,48% | 0,7197 | 0,1299 | Subótimo |
| **$K = 9$** | **0,5984 $\pm$ 0,0040** | **78,22%** | **0,7048** | **0,1064** | **Selecionado** |

> **Diagnóstico KNN:** $K=3$ apresentou forte sobreajuste à vizinhança imediata. À medida que $K$ aumentou para 9, o gap entre treino e validação caiu de 0,2519 para 0,1064, conferindo maior poder de generalização e o ápice do $F_1$ no grid.

### 2.2 Grade de Experimentos: Árvore de Decisão (`max_depth` $\in \{3, 5, 7, \text{None}\}$)

| Profundidade | $F_1$ Validação (Média $\pm$ Desvio) | Acurácia Validação | $F_1$ Treino | Gap de Overfitting ($\Delta F_1$) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Profundidade = 3** | 0,7060 $\pm$ 0,0093 | 87,45% | 0,7079 | 0,0018 | Subajuste relativo |
| **Profundidade = 5** | 0,7331 $\pm$ 0,0073 | 88,26% | 0,7368 | 0,0037 | Estável |
| **Profundidade = 7** | **0,7551 $\pm$ 0,0117** | **89,58%** | **0,7715** | **0,0164** | **Selecionado** |
| **Sem Limite (`None`)** | 0,7479 $\pm$ 0,0057 | 88,85% | 1,0000 | 0,2521 | Memorização espúria |

> **Diagnóstico Árvore:** A árvore irrestrita (`max_depth=None`) memorizou integralmente os dados de treino ($F_1 = 1,00$), sofrendo colapso na generalização com gap de 0,2521. A poda prévia em **profundidade 7** maximizou o $F_1$ médio (0,7551) mantendo um gap estritamente controlado de apenas 0,0164.

---

## 3. Avaliação Definitiva no Conjunto de Teste

Com os hiperparâmetros fixados ($K=9$ e $\text{depth}=7$), os modelos foram retreinados em todo o conjunto de treino balanceado (25.928 registros originais / 40.514 balanceados) e submetidos a uma **avaliação cega única** no conjunto de teste (6.483 registros).

### 3.1 Tabela Comparativa de Desempenho

| Modelo | Hiperparâmetro | Acurácia Global | Precisão (1) | Recall (1) | F1-Score (1) | Falsos Positivos (FP) | Falsos Negativos (FN) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **KNN** | $K = 9$ | 78,61% | 50,75% | **74,33%** | 0,6031 | 1.023 | **364** |
| **Árvore** | `depth = 7` | **90,28%** | **80,54%** | 73,27% | **0,7674** | **251** | 379 |
| **Diferença ($\Delta$)** | — | **+11,67 p.p.** | **+29,79 p.p.** | -1,06 p.p. | **+0,1643** | **-772 FP** | +15 FN |

### 3.2 Matrizes de Confusão Detalhadas

```
Matriz de Confusão — KNN (K=9)
                  Previsto Em Dia (0)    Previsto Inadimplente (1)
Real Em Dia (0)          4.042 (VN)              1.023 (FP)
Real Inadimplente (1)      364 (FN)              1.054 (VP)

Matriz de Confusão — Árvore de Decisão (depth=7)
                  Previsto Em Dia (0)    Previsto Inadimplente (1)
Real Em Dia (0)          4.814 (VN)                251 (FP)
Real Inadimplente (1)      379 (FN)              1.039 (VP)
```

### 3.3 Relatórios de Classificação Completos (*Classification Reports*)

#### Relatório do KNN ($K=9$)
```
              precision    recall  f1-score   support
  Em dia (0)     0.9174    0.7980    0.8536      5065
Inadimpl. (1)    0.5075    0.7433    0.6031      1418

    accuracy                         0.7861      6483
   macro avg     0.7124    0.7707    0.7284      6483
weighted avg     0.8277    0.7861    0.7988      6483
```

#### Relatório da Árvore de Decisão (`depth=7`)
```
              precision    recall  f1-score   support
  Em dia (0)     0.9270    0.9504    0.9386      5065
Inadimpl. (1)    0.8054    0.7327    0.7674      1418

    accuracy                         0.9028      6483
   macro avg     0.8662    0.8416    0.8530      6483
weighted avg     0.9004    0.9028    0.9011      6483
```

---

## 4. Análise de Custo e Simulação de Negócio

### 4.1 A Natureza Assimétrica dos Erros de Crédito
Na esteira de concessão de crédito, os custos de classificação incorreta são profundamente heterogêneos:
* **Falso Negativo (FN):** O modelo classifica um cliente inadimplente como bom pagador. O crédito é concedido e o cliente entra em *default*. O prejuízo envolve a perda direta do principal emprestado (capital financeiro).
* **Falso Positivo (FP):** O modelo classifica um cliente adimplente como de risco. O crédito é recusado. O prejuízo restringe-se ao custo de oportunidade (perda do *spread* de juros) e potencial atrito de relacionamento.

Portanto, $\text{Custo}(FN) \gg \text{Custo}(FP)$.

### 4.2 Equação de Indiferença (*Break-Even Analysis*)
Comparando as duas alternativas em termos de volume de erros:
$$\Delta FN = FN_{\text{Árvore}} - FN_{\text{KNN}} = 379 - 364 = +15$$
$$\Delta FP = FP_{\text{KNN}} - FP_{\text{Árvore}} = 1.023 - 251 = +772$$

O ponto de equilíbrio ocorre quando o custo total de ambos os modelos se iguala:
$$15 \times C_{\text{FN}} = 772 \times C_{\text{FP}} \implies \frac{C_{\text{FN}}}{C_{\text{FP}}} = \frac{772}{15} \approx 51,467$$

* **Conclusão:** Para que o KNN se tornasse financeiramente superior à Árvore de Decisão, **cada calote (FN) precisaria custar mais de 51,47 vezes a perda de margem de um cliente recusado (FP)**. Como a perda máxima de um calote é limitada ao valor de face do contrato e a margem de juros raramente é inferior a 2% do valor financiado, a relação real de mercado oscila tipicamente entre $3:1$ e $10:1$. Logo, a superioridade econômica da Árvore de Decisão é matematicamente robusta a quaisquer premissas de custo plausíveis.

### 4.3 Simulação Numérica Ilustrativa
Adotando-se uma ponderação conservadora de mercado ($C_{\text{FP}} = \text{R\$}~1.000$ e $C_{\text{FN}} = \text{R\$}~5.000$ — proporção $5:1$):

| Modelo | FP | FN | Custo dos FPs (R$) | Custo dos FNs (R$) | Custo Total Estimado (R$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN ($K=9$)** | 1.023 | 364 | R$ 1.023.000,00 | R$ 1.820.000,00 | R$ 2.843.000,00 |
| **Árvore (`depth=7`)** | 251 | 379 | R$ 251.000,00 | R$ 1.895.000,00 | **R$ 2.146.000,00** |
| **Vantagem Econômica** | **-772** | **+15** | **-R$ 772.000,00** | **+R$ 75.000,00** | **-R$ 697.000,00 (-24,5%)** |

---

## 5. Importância dos Atributos (*Feature Importance*)

Na Árvore de Decisão (`depth=7`), a contribuição das variáveis para a redução da impureza de Gini concentrou-se fortemente em indicadores de solvência e custo de empréstimo:

| Variável | Importância Relativa | Proporção Acumulada | Descrição e Interpretação de Negócio |
| :--- | :---: | :---: | :--- |
| `comprometimento_renda` | **31,70%** | 31,70% | Razão valor do empréstimo / renda anual (variável calculada) |
| `loan_int_rate` | **26,91%** | 58,61% | Taxa de juros contratada (precificação de risco pelo mercado) |
| `person_income` | **14,49%** | 73,10% | Renda anual declarada do proponente |
| `person_home_ownership_RENT` | **6,34%** | 79,44% | Indicador de moradia alugada (indício de despesa fixa elevada) |
| `loan_grade_D` | **5,24%** | 84,68% | Classificação de risco contratual grau D |
| `loan_grade_C` | 3,32% | 88,00% | Classificação de risco contratual grau C |
| `loan_intent_MEDICAL` | 2,83% | 90,83% | Empréstimo motivado por despesas médicas inesperadas |
| *Demais 20 atributos* | 9,17% | 100,00% | Demais variáveis categóricas, histórico e prazo |

> **Nota:** As três primeiras variáveis respondem por **73,10%** de todo o ganho informacional das divisões da árvore, validando a relevância crítica da variável gerada `comprometimento_renda`.

---

## 6. Verificação Complementar de Robustez

Para auditar o impacto das variáveis contratuais (`loan_grade` e `loan_int_rate`) — que poderiam eventualmente sofrer de endogeneidade se definidas concomitantemente à aprovação —, executou-se uma avaliação cruzada complementar com as mesmas 5 dobras de treino:

| Cenário de Modelo | $F_1$ Médio (Classe 1) | Desvio ($\sigma$) | Recall Médio | Precisão Média |
| :--- | :---: | :---: | :---: | :---: |
| **Referência Aleatória Estratificada** | 0,2219 | 0,0063 | 0,2239 | 0,2200 |
| **Árvore Completa (`depth=7`)** | **0,7551** | 0,0131 | **0,7329** | **0,7814** |
| **Árvore Reduzida (Sem Grade e Sem Juros)** | **0,6327** | 0,0076 | 0,6844 | 0,5888 |

A remoção total dos juros e da classificação de risco gera uma contração de $0,1225$ no $F_1$. Contudo, o modelo reduzido atinge $F_1 = 0,6327$, desempenho ainda superior ao KNN completo ($0,5984$), comprovando que os dados cadastrais (renda, valor e comprometimento) sustentam sozinhos um patamar preditivo de alta relevância.

---

## 7. Veredito Final e Recomendações Práticas

### 7.1 Veredito Técnico
A **Árvore de Decisão com Profundidade 7** é o modelo selecionado para implementação em produção.

**Justificativas Técnicas e Estratégicas:**
1. **Superioridade em Precisão:** Precisão de 80,54% na identificação de inadimplentes, contra apenas 50,75% do KNN, evitando o descarte desnecessário de mais de mil clientes pagadores.
2. **Robustez Financeira:** A vantagem financeira da árvore persiste para qualquer razão $\frac{C_{\text{FN}}}{C_{\text{FP}}} < 51,5$.
3. **Auditabilidade e *Compliance*:** Árvores com profundidade controlada geram caminhos de decisão lógicos transparentes, facilitando o cumprimento de normas regulatórias de concessão de crédito (*explainable AI*).
4. **Desempenho Computacional:** O tempo de inferência da Árvore é da ordem de $O(\text{depth}) = O(7)$ operações simples de comparação, dispensando o cálculo vetorial de distâncias euclidianas do KNN em bases com dezenas de milhares de clientes.

### 7.2 Recomendações para a Engenharia de Produção
* **Monitoramento Contínuo:** Estabelecer monitoramento de desvio de dados (*data drift*) na distribuição de renda e comprometimento dos proponentes.
* **Política de Alçada com Faixa de Indecisão:** Para proponentes em folhas da árvore com probabilidades estimadas entre $40\%$ e $60\%$, encaminhar a proposta para análise manual de crédito em vez de aprovação/rejeição automatizada sumária.
* **Governança:** A esteira de execução auditada conta com 12 testes unitários automatizados em `tests/test_pipeline.py` cobrindo consistência de transformações, limites numéricos e integridade dos artefatos.

---

*Relatório gerado e registrado a partir dos artefatos auditados em `projeto_saneado/resultados/`.*
