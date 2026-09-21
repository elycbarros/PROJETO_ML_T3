"""Cria a variável definido em uma cópia derivada da base limpa."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
hashes = json.loads((RAIZ / "documentacao/integridade_originais.json").read_text())
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
entrada = RAIZ / "dados_derivados/credito_sem_duplicatas_e_idades_invalidas.csv"
dados = pd.read_csv(entrada)

denominador_valido = dados["person_income"].notna() & dados["person_income"].gt(0)
dados["comprometimento_renda"] = np.where(
    denominador_valido,
    (dados["loan_amnt"] / dados["person_income"]) * 100,
    np.nan,
)
dados["comprometimento_renda"] = dados["comprometimento_renda"].replace([np.inf, -np.inf], np.nan)
assert dados["comprometimento_renda"].notna().all()
assert np.isfinite(dados["comprometimento_renda"]).all()

saida = RAIZ / "dados_derivados/credito_com_feature.csv"
dados.to_csv(saida, index=False)
comparacao = dados[["comprometimento_renda", "loan_percent_income"]].dropna().copy()
comparacao["loan_percent_income_percentual"] = comparacao["loan_percent_income"] * 100
erro = (comparacao["comprometimento_renda"] - comparacao["loan_percent_income_percentual"]).abs()
relatorio = f"""# Feature engineering: comprometimento da renda

## Regra aplicada

Foi criada `comprometimento_renda` com `(loan_amnt / person_income) * 100`. O cálculo só ocorre quando `person_income` existe e é maior que zero; valores infinitos são convertidos em nulos. Na base derivada, foram produzidos {int(dados['comprometimento_renda'].isna().sum())} nulos e {int(np.isfinite(dados['comprometimento_renda']).sum())} valores finitos.

## Comparação com a variável original

`loan_percent_income` parece representar a mesma relação em escala de proporção. Após convertê-la para percentual, a diferença absoluta mediana foi {erro.median():.4f} ponto percentual e {int((erro <= 0.005).mean() * 100)}% das linhas ficaram dentro de 0,005 ponto percentual.

Essa semelhança será considerada na preparação dos modelos para evitar peso duplicado à mesma informação. A coluna exigida permanece na base derivada e no dicionário; a seleção final de atributos será registrada depois do split.

## Arquivo gerado

`dados_derivados/credito_com_feature.csv` contém a limpeza estrutural e a coluna calculada. Os CSVs originais foram apenas lidos e tiveram seus hashes conferidos.
"""
(RAIZ / "documentacao" / "feature_engineering.md").write_text(relatorio)
for nome, esperado in hashes.items():
    assert hashlib.sha256((RAIZ / nome).read_bytes()).hexdigest() == esperado
print("Feature criada; integridade dos CSVs originais confirmada.")
