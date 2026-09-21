"""Diagnóstico e limpeza estrutural; imputação fica para o pipeline de treino."""
from pathlib import Path
import hashlib, json
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
dados = pd.read_csv(RAIZ / "credit_risk_dataset.csv")
antes = len(dados)
duplicadas = int(dados.duplicated().sum())
dados = dados.drop_duplicates().copy()
inconsistencias = {
    "idade_maior_100": int((dados.person_age > 100).sum()),
    "emprego_maior_80": int((dados.person_emp_length > 80).sum()),
    "emprego_maior_igual_idade": int((dados.person_emp_length >= dados.person_age).fillna(False).sum()),
    "renda_nao_positiva": int((dados.person_income <= 0).sum()),
    "emprestimo_nao_positivo": int((dados.loan_amnt <= 0).sum()),
    "historico_maior_idade": int((dados.cb_person_cred_hist_length > dados.person_age).sum()),
}
idade_invalida = dados.person_age > 100
dados = dados.loc[~idade_invalida].copy()
saida = RAIZ / "dados_derivados"
saida.mkdir(exist_ok=True)
dados.to_csv(saida / "credito_sem_duplicatas_e_idades_invalidas.csv", index=False)
relatorio = f"""# Data prep: duplicatas, nulos e consistência

## Duplicatas

Foram encontradas {duplicadas} linhas inteiramente duplicadas em {antes:,} registros. Elas foram removidas somente na cópia derivada; o CSV original permanece intacto.

## Nulos

Continuam ausentes valores em `person_emp_length` e `loan_int_rate`. A política é imputar a mediana dentro do pipeline de treino: são variáveis numéricas com assimetria e valores extremos. O valor será aprendido apenas no treino e aplicado ao teste sem novo ajuste.

## Consistência

| Regra | Quantidade |
|---|---:|
| idade maior que 100 | {inconsistencias['idade_maior_100']} |
| tempo de emprego maior que 80 anos | {inconsistencias['emprego_maior_80']} |
| tempo de emprego maior ou igual à idade | {inconsistencias['emprego_maior_igual_idade']} |
| renda não positiva | {inconsistencias['renda_nao_positiva']} |
| empréstimo não positivo | {inconsistencias['emprestimo_nao_positivo']} |
| histórico de crédito maior que a idade | {inconsistencias['historico_maior_idade']} |

As {int(idade_invalida.sum())} linhas com idade acima de 100 foram excluídas da cópia derivada porque são inconsistentes com a população de solicitantes. Não foram aplicados limites genéricos a renda ou empréstimo: não há valores não positivos e valores altos ainda podem representar casos reais.

## Arquivo derivado

`dados_derivados/credito_sem_duplicatas_e_idades_invalidas.csv` contém a limpeza estrutural e mantém os nulos. A imputação será ajustada somente dentro do treino. A coluna `comprometimento_renda` será criada na próxima etapa.
"""
(RAIZ / "documentacao" / "data_prep.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print(f"Limpeza estrutural concluída: {antes:,} -> {len(dados):,} linhas; CSVs originais preservados.")
