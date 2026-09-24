"""Pipeline canônico: preparação por dobra, seleção por CV e teste descritivo.

As transformações estatísticas nunca são ajustadas em validação ou teste.
"""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
CONFIGS = [("KNN", k) for k in (3, 5, 7, 9)] + [
    ("Tree", depth) for depth in (3, 5, 7, None)
]
CONTINUOUS = [
    "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "comprometimento_renda",
]
DISCRETE = ["person_age", "cb_person_cred_hist_length"]
LIMITATION = (
    "O teste só é usado depois que os hiperparâmetros são escolhidos por validação "
    "cruzada (5 dobras, só no treino). Como a base completa foi examinada durante o "
    "desenvolvimento deste projeto, o teste não tem a independência de uma amostra "
    "nunca vista antes; a seleção, porém, não consulta o teste em nenhuma etapa."
)


def verify_originals():
    manifest = json.loads((ROOT / "documentacao/integridade_originais.json").read_text())
    actual = {}
    for name, expected in manifest.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Arquivo original divergente: {name}")
        actual[name] = digest
    return actual


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def safe_ratio(amount, income):
    """Recusa operandos inválidos antes de efetuar qualquer divisão."""
    a, b = np.asarray(amount, dtype=float), np.asarray(income, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    result = np.full(a.shape, np.nan)
    np.divide(a, b, out=result, where=valid)
    return result * 100


def clean_data():
    verify_originals()
    raw = pd.read_csv(ROOT / "credit_risk_dataset.csv")
    clean = raw.drop_duplicates().copy()
    # O índice original é rastreabilidade, nunca um preditor.
    clean.index.name = "origem_linha"
    excluded = clean.loc[clean.person_age >= 120].copy()
    clean = clean.loc[clean.person_age < 120].copy()
    invalid_emp = clean.person_emp_length.ge(clean.person_age) | clean.person_emp_length.lt(0)
    emp_rows = clean.loc[invalid_emp, ["person_age", "person_emp_length"]].copy()
    clean.loc[invalid_emp, "person_emp_length"] = np.nan
    folder = ROOT / "dados_derivados"
    folder.mkdir(exist_ok=True)
    clean.to_csv(folder / "credito_sem_duplicatas_e_idades_invalidas.csv", index=False)
    out = ROOT / "resultados"
    out.mkdir(exist_ok=True)
    excluded.to_csv(out / "idades_excluidas.csv")
    emp_rows.to_csv(out / "emprego_invalidado.csv")
    pd.DataFrame({"origem_linha": clean.index}).to_csv(folder / "origens.csv", index=False)
    write_json(out / "limpeza.json", {
        "original": len(raw), "duplicatas_removidas": int(raw.duplicated().sum()),
        "idades_excluidas": len(excluded), "idades_observadas": excluded.person_age.tolist(),
        "empregos_convertidos_em_nulos": int(invalid_emp.sum()), "final": len(clean),
        "nulos_apos_limpeza": clean.isna().sum().astype(int).to_dict(),
    })
    assert not clean.duplicated().any()
    assert not (clean.person_emp_length >= clean.person_age).any()
    verify_originals()
    return clean


def add_feature(clean):
    data = clean.copy()
    data["comprometimento_renda"] = safe_ratio(data.loan_amnt, data.person_income)
    if data.comprometimento_renda.isna().any():
        raise ValueError("Operandos inválidos: tratar no treino antes de criar a razão.")
    assert np.isfinite(data.comprometimento_renda).all()
    data.to_csv(ROOT / "dados_derivados/credito_com_feature.csv", index=False)
    error = (data.comprometimento_renda - data.loan_percent_income * 100).abs()
    write_json(ROOT / "resultados/feature.json", {
        "nulos": int(data.comprometimento_renda.isna().sum()),
        "finitos": int(np.isfinite(data.comprometimento_renda).sum()),
        "erro_mediano_pontos_percentuais": float(error.median()),
        "fracao_dentro_meio_ponto_percentual": float(error.le(0.5001).mean()),
    })
    return data


def split_data(data):
    X = data.drop(columns=["loan_status", "loan_percent_income"])
    y = data.loan_status
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.20, stratify=y, random_state=SEED
    )
    assert not set(data.index[train]) & set(data.index[test])
    return X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]


def oversample_indices(y, seed=SEED):
    """Random Over-Sampling: inclui cada linha original uma vez e acrescenta cópias
    aleatórias, com reposição, apenas do déficit das classes minoritárias."""
    values = np.asarray(y)
    classes, counts = np.unique(values, return_counts=True)
    rng = np.random.default_rng(seed)
    indices = list(range(len(values)))
    for c, count in zip(classes, counts):
        deficit = int(counts.max() - count)
        if deficit:
            indices.extend(rng.choice(np.flatnonzero(values == c), deficit, replace=True))
    indices = np.asarray(indices, dtype=int)
    rng.shuffle(indices)
    assert np.array_equal(np.unique(indices), np.arange(len(values)))
    return indices


def prepare(Xtrain, ytrain, Xeval):
    # Taxa de juros é quase simétrica: média; emprego tem cauda direita: mediana.
    numeric = Xtrain.select_dtypes(include="number").columns.tolist()
    median_cols = [c for c in numeric if c != "loan_int_rate"]
    categorical = Xtrain.select_dtypes(exclude="number").columns.tolist()
    transformer = ColumnTransformer([
        ("mediana", SimpleImputer(strategy="median"), median_cols),
        ("juros", SimpleImputer(strategy="mean"), ["loan_int_rate"]),
        ("categoria", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    original = transformer.fit_transform(Xtrain)
    evaluation = transformer.transform(Xeval)
    names = transformer.get_feature_names_out().tolist()
    balanced_indices = oversample_indices(ytrain)
    balanced = original[balanced_indices].copy()
    balanced_y = ytrain.to_numpy()[balanced_indices]
    scale_indices = [
        i for i, name in enumerate(names) if name.split("__", 1)[1] in CONTINUOUS
    ]
    scaler = StandardScaler().fit(balanced[:, scale_indices])

    def scaled(matrix):
        result = matrix.copy()
        result[:, scale_indices] = scaler.transform(matrix[:, scale_indices])
        return result

    matrices = {
        "KNN": (scaled(balanced), scaled(original), scaled(evaluation)),
        "Tree": (balanced, original, evaluation),
    }
    assert np.isfinite(balanced).all() and np.isfinite(evaluation).all()
    assert not set(Xtrain.index[balanced_indices]) & set(Xeval.index)
    audit = {
        "train": len(Xtrain), "evaluation": len(Xeval), "balanced": len(balanced),
        "retained_originals": len(np.unique(balanced_indices)),
        "overlap_source_rows": 0,
        "balanced_classes": pd.Series(balanced_y).value_counts().sort_index().to_dict(),
        "evaluation_classes": ytrain.iloc[:0].value_counts().to_dict(),
        "same_resampling_for_both_models": True,
        "scaler_fit_rows": int(scaler.n_samples_seen_),
        "scale_columns": CONTINUOUS,
        "discrete_without_scale": DISCRETE,
        "oversampling_sha256": hashlib.sha256(
            Xtrain.index.to_numpy()[balanced_indices].tobytes()
        ).hexdigest(),
    }
    return matrices, balanced_y, names, audit, balanced_indices, transformer, scaler


def classifier(kind, param):
    if kind == "KNN":
        return KNeighborsClassifier(n_neighbors=param, weights="uniform", n_jobs=1)
    return DecisionTreeClassifier(max_depth=param, random_state=SEED)


def scores(y, pred):
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_1": float(precision_score(y, pred, zero_division=0)),
        "recall_1": float(recall_score(y, pred, zero_division=0)),
        "f1_1": float(f1_score(y, pred, zero_division=0)),
    }


def param_text(param):
    return "None" if param is None else str(param)


def select_configs(rows):
    selected = {}
    for kind in ("KNN", "Tree"):
        candidates = [r for r in rows if r["model"] == kind]
        # Empates exatos: K maior suaviza KNN; profundidade menor simplifica árvore.
        def key(r):
            param = r["param"]
            simplicity = int(param) if kind == "KNN" else -(999 if param == "None" else int(param))
            return r["cv_f1"], simplicity
        selected[kind] = max(candidates, key=key)["param"]
    return selected


def run_cv(X, y):
    fold_rows, audits = [], []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (train, validation) in enumerate(cv.split(X, y), start=1):
        Xt, Xv, yt, yv = X.iloc[train], X.iloc[validation], y.iloc[train], y.iloc[validation]
        matrices, balanced_y, _, audit, _, _, _ = prepare(Xt, yt, Xv)
        audit["fold"] = fold
        audit["evaluation_classes"] = yv.value_counts().sort_index().to_dict()
        audits.append(audit)
        for kind, param in CONFIGS:
            fit_X, natural_X, validation_X = matrices[kind]
            model = classifier(kind, param).fit(fit_X, balanced_y)
            row = {"model": kind, "param": param_text(param), "fold": fold}
            row.update({"train_" + k: v for k, v in scores(yt, model.predict(natural_X)).items()})
            row.update({"validation_" + k: v for k, v in scores(yv, model.predict(validation_X)).items()})
            fold_rows.append(row)
        print(f"Validação: dobra {fold}/5 concluída.", flush=True)
    folds = pd.DataFrame(fold_rows)
    summary = []
    for kind, param in CONFIGS:
        group = folds[(folds.model == kind) & (folds.param == param_text(param))]
        row = {"model": kind, "param": param_text(param),
               "cv_f1": float(group.validation_f1_1.mean()),
               "cv_std": float(group.validation_f1_1.std(ddof=0))}
        for prefix in ("train", "validation"):
            for metric in ("accuracy", "precision_1", "recall_1", "f1_1"):
                col = prefix + "_" + metric
                row[col + "_mean"] = float(group[col].mean())
                row[col + "_std"] = float(group[col].std(ddof=0))
        row["cv_gap_f1"] = row["train_f1_1_mean"] - row["cv_f1"]
        summary.append(row)
    return summary, folds, audits


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(filename.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(filename.with_suffix(".png"), dpi=140)
    plt.close(fig)


def model_artifacts(kind, param, model, test_X, ytest, names, directory):
    prefix = f"knn_k{param}" if kind == "KNN" else f"arvore_depth{param_text(param)}"
    pred = model.predict(test_X)
    cm = confusion_matrix(ytest, pred, labels=[0, 1])
    report = classification_report(
        ytest, pred, labels=[0, 1], target_names=["Em dia (0)", "Inadimplente (1)"],
        digits=4, zero_division=0,
    )
    (directory / f"{prefix}_classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=["Real 0", "Real 1"],
                 columns=["Previsto 0", "Previsto 1"]).to_csv(directory / f"{prefix}_matriz_confusao.csv")
    pd.DataFrame({"origem_linha": ytest.index, "real": ytest.to_numpy(), "previsto": pred}).to_csv(
        directory / f"{prefix}_predicoes.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Em dia (0)", "Inadimplente (1)"])
    ax.set_yticks([0, 1], ["Em dia (0)", "Inadimplente (1)"])
    ax.set(xlabel="Classe prevista", ylabel="Classe real", title=f"{kind} ({param_text(param)}) — teste")
    labels = [["VN", "FP"], ["FN", "VP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i,j]}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
    save_plot(fig, directory / f"{prefix}_matriz_confusao")
    if kind == "Tree":
        importance = pd.DataFrame({
            "feature": [n.split("__", 1)[1] for n in names],
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        importance.to_csv(directory / "feature_importance_arvore.csv", index=False)
        top = importance.head(10).sort_values("importance")
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.barh(top.feature, top.importance, color="#376e9f")
        ax.set(xlabel="Redução relativa de impureza (não causal)",
               title=f"Importância na árvore avaliada — profundidade {param_text(param)}")
        save_plot(fig, directory / "feature_importance_arvore")
    return prefix


def comparison_plots(experiments, directory):
    for kind, filename in [
        ("KNN", "curva_validacao_knn"), ("Tree", "curva_overfitting_arvore")
    ]:
        table = experiments[experiments.model == kind]
        pos = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(pos, table.treino_f1_1, "o-", label="Treino original, modelo final")
        ax.errorbar(pos, table.cv_f1, yerr=table.cv_std, fmt="s-", capsize=4,
                    label="Validação interna (média ± desvio)")
        ax.plot(pos, table.teste_f1_1, "^-", label="Teste: comparação após seleção")
        ax.set_xticks(pos, table.param)
        ax.set(xlabel="K" if kind == "KNN" else "Profundidade",
               ylabel="F1 da classe inadimplente", ylim=(0, 1.04),
               title=f"{kind}: treino, validação e teste")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        save_plot(fig, directory / filename)


def finance_table(final, directory, cost_fp=1000, cost_fn=5000):
    frame = final[["model", "param", "fp", "fn"]].copy()
    frame["custo_fp_hipotetico"] = frame.fp * cost_fp
    frame["custo_fn_hipotetico"] = frame.fn * cost_fn
    frame["custo_total_hipotetico"] = frame.custo_fp_hipotetico + frame.custo_fn_hipotetico
    frame.to_csv(directory / "simulacao_financeira_custos.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame.model, frame.custo_fp_hipotetico / 1000, label="FP × R$ 1.000")
    ax.bar(frame.model, frame.custo_fn_hipotetico / 1000,
           bottom=frame.custo_fp_hipotetico / 1000, label="FN × R$ 5.000")
    ax.set(ylabel="Custo hipotético (milhares de reais)",
           title="Cenário ilustrativo — valores não fornecidos pela base")
    ax.legend()
    save_plot(fig, directory / "simulacao_financeira_custos")
    return frame


def run_all():
    before = verify_originals()
    clean = clean_data()
    data = add_feature(clean)
    Xtr, Xte, ytr, yte = split_data(data)
    stats = Xtr.select_dtypes(include="number").agg(
        ["count", "mean", "median", "std", "min", "max", "skew"]
    ).T
    stats["nulos"] = Xtr.isna().sum()
    stats.to_csv(ROOT / "resultados/estatisticas_treino.csv")
    summary, folds, audits = run_cv(Xtr, ytr)
    selection = select_configs(summary)
    selection_path = ROOT / "resultados/parametros_selecionados.json"
    write_json(selection_path, {
        "criterio": "maior F1 médio da classe 1 em 5 dobras no treino",
        "desempate": "K maior; árvore menos profunda",
        "selecionados": selection, "semente": SEED, "limitacao": LIMITATION,
    })
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    # A seleção é persistida ANTES de qualquer predição de teste.
    selection = json.loads(selection_path.read_text())["selecionados"]
    matrices, balanced_y, names, audit, indices, _, _ = prepare(Xtr, ytr, Xte)
    audit["evaluation_classes"] = yte.value_counts().sort_index().to_dict()
    audits.append({"fold": "final", **audit})
    directory = ROOT / "resultados/avaliacao_final"
    directory.mkdir(parents=True, exist_ok=True)
    fold_frame = pd.DataFrame(folds)
    fold_frame.to_csv(ROOT / "resultados/metricas_por_dobra.csv", index=False)
    pd.DataFrame(summary).to_csv(ROOT / "resultados/validacao_cruzada.csv", index=False)
    pd.DataFrame({"origem_linha": Xtr.index[indices]}).to_csv(
        ROOT / "resultados/indices_treino_balanceado.csv", index=False
    )
    pd.concat([
        pd.DataFrame({"origem_linha": Xtr.index, "particao": "treino"}),
        pd.DataFrame({"origem_linha": Xte.index, "particao": "teste"}),
    ]).to_csv(ROOT / "resultados/indices_split.csv", index=False)
    experiment_rows, final_rows = [], []
    for kind, param in CONFIGS:
        train_X, natural_X, test_X = matrices[kind]
        model = classifier(kind, param).fit(train_X, balanced_y)
        pred = model.predict(test_X)
        train_metrics = scores(ytr, model.predict(natural_X))
        test_metrics = scores(yte, pred)
        row = next(r.copy() for r in summary if r["model"] == kind and r["param"] == param_text(param))
        row.update({"treino_" + k: v for k, v in train_metrics.items()})
        row.update({"teste_" + k: v for k, v in test_metrics.items()})
        row["gap_f1"] = train_metrics["f1_1"] - test_metrics["f1_1"]
        row["selecionado"] = selection[kind] == param_text(param)
        experiment_rows.append(row)
        if row["selecionado"]:
            tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
            prefix = model_artifacts(kind, param, model, test_X, yte, names, directory)
            final_rows.append({
                "model": kind, "param": param_text(param), "prefix": prefix,
                "cv_f1": row["cv_f1"],
                **{"test_" + k: v for k, v in test_metrics.items()},
                "fp": int(fp), "fn": int(fn), "tn": int(tn), "tp": int(tp),
            })
    assert hashlib.sha256(selection_path.read_bytes()).hexdigest() == selection_hash
    experiments, final = pd.DataFrame(experiment_rows), pd.DataFrame(final_rows)
    experiments.to_csv(ROOT / "resultados/experimentos.csv", index=False)
    final.to_csv(ROOT / "resultados/avaliacao_teste.csv", index=False)
    final.to_csv(directory / "comparacao_final.csv", index=False)
    for kind, name in [("KNN", "knn"), ("Tree", "arvore")]:
        experiments[experiments.model == kind].to_csv(
            ROOT / f"resultados/{name}_experimentos.csv", index=False
        )
    comparison_plots(experiments, directory)
    finance = finance_table(final, directory)
    np.savez_compressed(
        ROOT / "dados_derivados/matrizes_preparadas.npz",
        Xtr_knn=matrices["KNN"][0], Xte_knn=matrices["KNN"][2],
        Xtr_arvore=matrices["Tree"][0], Xte_arvore=matrices["Tree"][2],
        Xtr_knn_original=matrices["KNN"][1], Xtr_arvore_original=matrices["Tree"][1],
        ytr=balanced_y, ytr_original=ytr.to_numpy(), yte=yte.to_numpy(),
        origem_treino=Xtr.index.to_numpy(), origem_teste=Xte.index.to_numpy(),
        indices_balanceados=indices, features=np.array(names),
    )
    write_json(ROOT / "resultados/separacao_preparacao.json", audit)
    write_json(ROOT / "resultados/auditoria_execucao.json", {
        "hashes_originais": before, "folds": audits, "selection_sha256": selection_hash,
        "configuration_count": len(experiments), "test_consulted_only_after_selection_this_run": True,
        "historical_test_exposure": LIMITATION,
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "matplotlib")},
    })
    from relatorios_credito import write_reports
    write_reports(ROOT, data, stats, experiments, final, finance, selection, LIMITATION)
    assert verify_originals() == before
    print("Pipeline concluído; oito configurações e integridade verificadas.", flush=True)
    return experiments, final


if __name__ == "__main__":
    run_all()

