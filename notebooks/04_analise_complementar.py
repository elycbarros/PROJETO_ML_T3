"""Referência simples e sensibilidade a variáveis de disponibilidade incerta.

Verificação extra, opcional (não exigida pelo problema). Roda depois do pipeline
principal (notebooks/03_executar_pipeline.py) e depende dos arquivos que ele gera em
dados_derivados/. Usa somente o treino da divisão principal; não altera seleção nem
avaliação final. Substitui a seção "Verificação complementar" de
documentacao/03_avaliacao_e_veredito.md, sem tocar no resto do arquivo. Nada mais chama
este script automaticamente — ele só roda quando você o executa.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from pipeline_credito import ROOT, SEED, classifier, prepare, scores, split_data, verify_originals


def run():
    before = verify_originals()
    data = pd.read_csv(ROOT / "dados_derivados/credito_com_feature.csv")
    origins = pd.read_csv(ROOT / "dados_derivados/origens.csv").origem_linha
    if len(data) != len(origins):
        raise ValueError("A base derivada e os índices de origem não correspondem.")
    data.index = origins.to_numpy()
    Xtrain, Xtest, ytrain, _ = split_data(data)

    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, validation) in enumerate(cv.split(Xtrain, ytrain), start=1):
        Xt, Xv = Xtrain.iloc[train], Xtrain.iloc[validation]
        yt, yv = ytrain.iloc[train], ytrain.iloc[validation]
        assert not set(Xt.index) & set(Xv.index)
        assert not set(Xv.index) & set(Xtest.index)

        baseline = DummyClassifier(strategy="stratified", random_state=SEED + fold)
        baseline.fit(Xt, yt)
        rows.append({"modelo": "Referência aleatória estratificada", "dobra": fold,
                     **scores(yv, baseline.predict(Xv))})

        matrices, balanced_y, names, _, _, _, _ = prepare(Xt, yt, Xv)
        fitted, _, evaluation = matrices["Tree"]
        full = classifier("Tree", 7).fit(fitted, balanced_y)
        rows.append({"modelo": "Árvore completa (profundidade 7)", "dobra": fold,
                     **scores(yv, full.predict(evaluation))})

        keep = [i for i, name in enumerate(names)
                if name != "juros__loan_int_rate"
                and not name.startswith("categoria__loan_grade_")]
        retained_names = [names[i] for i in keep]
        assert "juros__loan_int_rate" in names
        assert any(n.startswith("categoria__loan_grade_") for n in names)
        assert len(keep) < len(names)
        assert all(n != "juros__loan_int_rate" and not n.startswith("categoria__loan_grade_")
                   for n in retained_names)
        reduced = classifier("Tree", 7).fit(fitted[:, keep], balanced_y)
        rows.append({"modelo": "Árvore sem grade e juros (profundidade 7)",
                     "dobra": fold, **scores(yv, reduced.predict(evaluation[:, keep]))})

    results = pd.DataFrame(rows)
    output = ROOT / "resultados/analise_complementar_dobras.csv"
    results.to_csv(output, index=False)
    summary = results.groupby("modelo", sort=False).agg(
        f1_medio=("f1_1", "mean"), f1_desvio=("f1_1", "std"),
        recall_medio=("recall_1", "mean"), precisao_media=("precision_1", "mean"),
    )
    summary.to_csv(ROOT / "resultados/analise_complementar_resumo.csv")
    lines = [
        "## Verificação complementar", "",
        "Esta análise usa apenas as cinco dobras do conjunto de treino. A divisão de teste, "
        "a seleção de KNN/Árvore e o veredito principal permanecem inalterados. Gerada por "
        "`notebooks/04_analise_complementar.py`, um passo extra opcional (não exigido pelo "
        "problema), rodado manualmente depois do pipeline principal.", "",
        "| Modelo | F1 médio (classe 1) | Desvio entre dobras | Recall médio | Precisão média |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    def decimal_br(value, casas=4):
        return f"{value:.{casas}f}".replace(".", ",")
    for model, row in summary.iterrows():
        lines.append(f"| {model} | {decimal_br(row.f1_medio)} | {decimal_br(row.f1_desvio)} | "
                     f"{decimal_br(row.recall_medio)} | {decimal_br(row.precisao_media)} |")
    full_f1 = summary.loc["Árvore completa (profundidade 7)", "f1_medio"]
    reduced_f1 = summary.loc["Árvore sem grade e juros (profundidade 7)", "f1_medio"]
    lines += [
        "", "A referência aleatória estratificada preserva aproximadamente a proporção das "
        "classes, mas não aprende relações entre atributos e alvo.",
        "A árvore reduzida exclui todas as colunas codificadas de `loan_grade` e "
        "`loan_int_rate`; mantém a mesma profundidade, as mesmas dobras e os mesmos "
        "índices de balanceamento da árvore completa.",
        f"A diferença média de F1 (completa menos reduzida) foi {decimal_br(full_f1 - reduced_f1)}.",
        "Isso mede sensibilidade nesta base, não demonstra vazamento por si só. É preciso "
        "confirmar quando grade e juros ficam disponíveis no processo real.",
        "A profundidade 7 foi fixada a partir da análise principal; não foi otimizada "
        "novamente para a versão reduzida. Portanto, esta comparação é exploratória, "
        "e não uma nova seleção de modelo.", "",
        "Resultados por dobra: `resultados/analise_complementar_dobras.csv`. "
        "Resumo: `resultados/analise_complementar_resumo.csv`.", "",
    ]
    secao_nova = "\n".join(lines).rstrip() + "\n"
    alvo = ROOT / "documentacao/03_avaliacao_e_veredito.md"
    if not alvo.exists():
        raise FileNotFoundError(
            "documentacao/03_avaliacao_e_veredito.md não existe. Rode "
            "notebooks/03_executar_pipeline.py primeiro — este script só completa uma "
            "seção que o pipeline principal já deve ter criado."
        )
    atual = alvo.read_text()
    if "## Verificação complementar" not in atual:
        raise ValueError(
            "documentacao/03_avaliacao_e_veredito.md não tem a seção 'Verificação "
            "complementar' esperada. Rode notebooks/03_executar_pipeline.py de novo."
        )
    antes, resto = atual.split("## Verificação complementar", 1)
    _, depois = resto.split("\n## Nota metodológica", 1)
    alvo.write_text(antes.rstrip() + "\n\n" + secao_nova + "\n## Nota metodológica" + depois)
    assert verify_originals() == before
    print(summary.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    run()
