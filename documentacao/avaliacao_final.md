# Avaliação final e veredito de negócio

Foram comparados KNN (`K=3`) e Árvore de Decisão (`max_depth=7`) no teste original (6.483 registros), sem reamostragem, após seleção estrita por validação cruzada sem vazamento.

| Modelo | Acurácia | Precisão 1 | Recall 1 | F1 1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| KNN K=3 | 0.825 | 0.579 | 0.733 | 0.647 | 756 | 379 |
| Árvore depth=7 | 0.884 | 0.721 | 0.767 | 0.743 | 421 | 331 |

Um falso positivo trata como inadimplente quem pagaria em dia, gerando atrito ou recusa injustificada de um bom cliente. Um falso negativo libera crédito a quem inadimplirá, gerando perda financeira direta do principal.

## Veredito

Recomendo a **Árvore de Decisão com `max_depth=7` para um piloto controlado**.
Na validação corrigida, a Árvore superou o KNN em todas as métricas:
- F1-Score superior (0.743 vs 0.647);
- Recall superior (0.767 vs 0.733), capturando mais inadimplentes (331 FN contra 379 do KNN);
- Precisão e acurácia substancialmente maiores, reduzindo os falsos positivos quase pela metade (421 contra 756 do KNN).

## Explicabilidade: Importância das Variáveis (Feature Importance)

A análise da Árvore de Decisão (`max_depth=7`) revela os fatores determinantes na concessão do crédito:
1. **`comprometimento_renda` (31,7%)**: A variável criada na etapa de feature engineering provou ser o preditor mais decisivo do modelo, confirmando a hipótese de que o peso da parcela sobre a renda mensal é o principal sinalizador de risco de crédito;
2. **`loan_int_rate` (26,9%)**: Taxa de juros do contrato;
3. **`person_income` (14,5%)**: Renda declarada do solicitante;
4. **`person_home_ownership_RENT` (6,3%)**: Condição de moradia (aluguel vs própria/hipoteca);
5. **`loan_grade_D` (5,2%) e `loan_grade_C` (3,3%)**: Classificação interna da operação.

Gráfico completo disponível em: `resultados/avaliacao_final/feature_importance_arvore.svg`.

## Simulação de Impacto Financeiro (P&L Estimado)

Embora o problema não forneça a matriz de custos monetários exata, foi realizada uma simulação de negócio com parâmetros bancários representativos:
- **Custo do Falso Negativo (FN):** Concessão de crédito a cliente inadimplente $\rightarrow$ Perda média estimada de principal de **R$ 5.000,00**;
- **Custo do Falso Positivo (FP):** Recusa de cliente que pagaria em dia $\rightarrow$ Lucro cessante estimado (margem financeira líquida/juros perdidos) de **R$ 1.000,00**.

### Comparativo na Amostra de Teste (6.483 contratos avaliados):
* **KNN (K=3):**
  * Custo de FP (756 $\times$ R$ 1.000): R$ 756.000,00
  * Custo de FN (379 $\times$ R$ 5.000): R$ 1.895.000,00
  * **Perda Financeira Total: R$ 2.651.000,00**
* **Árvore de Decisão (depth=7):**
  * Custo de FP (421 $\times$ R$ 1.000): R$ 421.000,00
  * Custo de FN (331 $\times$ R$ 5.000): R$ 1.655.000,00
  * **Perda Financeira Total: R$ 2.076.000,00**

> **Resultado Econômico:** A adoção da Árvore de Decisão representa uma **economia direta estimada de R$ 575.000,00** somente nesta carteira de teste em relação ao KNN, reduzindo tanto a inadimplência quanto a perda de bons clientes.

Gráfico e dados disponíveis em: `resultados/avaliacao_final/simulacao_financeira_custos.svg`.

## Recomendações para Implantação

Antes de qualquer implantação em produção, o banco deve:
1. Calibrar a probabilidade de corte (limiar de decisão) utilizando a matriz de custos reais apurada pela controladoria;
2. Realizar validação fora do tempo (amostra temporal subsequente para monitorar *data drift*);
3. Monitorar métricas de disparidade e estabilidade populacional (PSI) por segmento socioeconômico.

Artefatos, relatórios e matrizes visuais estão salvos em `resultados/avaliacao_final/`. Os CSVs originais foram preservados intactos.
