"""Orquestrador principal do projeto.

Ao rodar este script, nesta ordem:
1. Executa `02_eda_graficos.py` (gera os gráficos da EDA e escreve a seção 1 de
   `documentacao/01_eda_e_preparacao.md`).
2. Chama `pipeline_credito.run_all()` — limpeza, preparação, validação cruzada,
   seleção de hiperparâmetros, avaliação no teste e gráficos de resultado. Ao final,
   `run_all()` chama `relatorios_credito.write_reports(...)`, que escreve as seções 2-4
   de `documentacao/01_eda_e_preparacao.md` e os arquivos `documentacao/02_modelagem.md`,
   `documentacao/03_avaliacao_e_veredito.md` e o `README.md`.

Nenhum outro script é chamado automaticamente por este. `01_inspecao_inicial.ipynb`
(exploração livre) e `04_analise_complementar.py` (verificação extra opcional) são
independentes e só rodam quando executados manualmente — o segundo depende dos
arquivos que este script gera em `dados_derivados/`.
"""
from pathlib import Path
import runpy
from pipeline_credito import run_all

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("02_eda_graficos.py")), run_name="__main__")
    run_all()
