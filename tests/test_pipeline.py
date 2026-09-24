"""Provas dos contratos de preparação, seleção e resultados."""
import sys
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from pipeline_credito import (
    safe_ratio, oversample_indices, prepare, select_configs, verify_originals,
    split_data, classifier,
)


class PreparationContracts(unittest.TestCase):
    def fixture(self):
        X = pd.DataFrame({
            "person_age": [20, 30, 40, 50, 60, 70],
            "person_income": [100, 200, 300, 400, 500, 600],
            "person_emp_length": [1., 2., np.nan, 4., 5., 6.],
            "loan_amnt": [10, 20, 30, 40, 50, 60],
            "loan_int_rate": [10., 12., np.nan, 16., 18., 20.],
            "cb_person_cred_hist_length": [1, 2, 3, 4, 5, 6],
            "comprometimento_renda": [10., 10., 10., 10., 10., 10.],
            "person_home_ownership": ["RENT", "OWN", "RENT", "OWN", "RENT", "OWN"],
        }, index=[10, 11, 12, 13, 14, 15])
        y = pd.Series([0, 0, 0, 0, 1, 1], index=X.index)
        V = X.iloc[:2].copy()
        V.index = [20, 21]
        return X, y, V

    def test_safe_ratio_masks_invalid_before_division(self):
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            result = safe_ratio([10, 10, 10, np.inf, np.nan, 0],
                                [100, 0, -1, 100, 100, 100])
        self.assertEqual(result[0], 10)
        self.assertTrue(np.isnan(result[1:]).all())

    def test_oversampling_retains_all_and_adds_only_minority(self):
        y = np.array([0, 0, 0, 0, 1, 1])
        idx = oversample_indices(y)
        np.testing.assert_array_equal(np.unique(idx), np.arange(len(y)))
        self.assertEqual(len(idx), 8)
        self.assertTrue(all(np.sum(idx == i) == 1 for i in range(4)))
        np.testing.assert_array_equal(np.bincount(y[idx]), [4, 4])
        np.testing.assert_array_equal(idx, oversample_indices(y))

    def test_evaluation_cannot_change_learned_parameters(self):
        X, y, V = self.fixture()
        first = prepare(X, y, V)
        poisoned = V.copy()
        poisoned.loc[:, "person_income"] = 10**9
        poisoned.loc[:, "person_home_ownership"] = "UNSEEN"
        second = prepare(X, y, poisoned)
        np.testing.assert_allclose(first[0]["KNN"][0], second[0]["KNN"][0])
        np.testing.assert_allclose(first[0]["Tree"][0], second[0]["Tree"][0])
        np.testing.assert_allclose(first[6].mean_, second[6].mean_)
        self.assertNotIn("categoria__person_home_ownership_UNSEEN", second[2])
        self.assertEqual(first[5].named_transformers_["juros"].statistics_[0], 15.2)

    def test_scaler_uses_balanced_continuous_only(self):
        X, y, V = self.fixture()
        matrices, by, names, audit, ids, _, scaler = prepare(X, y, V)
        self.assertEqual(scaler.n_samples_seen_, len(ids))
        self.assertEqual(audit["retained_originals"], len(X))
        numeric_scaled = [i for i, n in enumerate(names) if n.split("__",1)[1] in audit["scale_columns"]]
        np.testing.assert_allclose(matrices["KNN"][0][:, numeric_scaled].mean(axis=0), 0, atol=1e-12)
        for name in ["mediana__person_age", "mediana__cb_person_cred_hist_length"]:
            i = names.index(name)
            np.testing.assert_array_equal(matrices["KNN"][0][:, i], matrices["Tree"][0][:, i])
        self.assertTrue(audit["same_resampling_for_both_models"])
        np.testing.assert_array_equal(by, y.to_numpy()[ids])

    def test_origin_overlap_is_rejected(self):
        X, y, V = self.fixture()
        with self.assertRaises(AssertionError):
            prepare(X, y, X.iloc[:2])

    def test_selection_ignores_test_and_prefers_simpler_tie(self):
        rows = [{"model": "KNN", "param": str(k), "cv_f1": .5, "test_f1": 1/k}
                for k in [3, 5, 7, 9]]
        rows += [{"model": "Tree", "param": p, "cv_f1": .5, "test_f1": 1}
                 for p in ["3", "5", "7", "None"]]
        chosen = select_configs(rows)
        self.assertEqual(chosen, {"KNN": "9", "Tree": "3"})
        for row in rows:
            row["test_f1"] = -999
        self.assertEqual(select_configs(rows), chosen)


class OutputContracts(unittest.TestCase):
    def test_originals(self):
        self.assertEqual(len(verify_originals()), 1)

    def test_eight_corrected_experiments_and_selection(self):
        results = pd.read_csv(ROOT / "resultados/experimentos.csv", keep_default_na=False)
        self.assertEqual(results.groupby("model").size().to_dict(), {"KNN": 4, "Tree": 4})
        self.assertTrue(results[["treino_f1_1", "teste_f1_1", "cv_f1"]].notna().all().all())
        selection = json.loads((ROOT / "resultados/parametros_selecionados.json").read_text())["selecionados"]
        rows = results.to_dict("records")
        for row in rows:
            row["param"] = str(row["param"])
        self.assertEqual(select_configs(rows), selection)
        self.assertEqual(results.selecionado.sum(), 2)

    def test_partitions_and_audits(self):
        audit = json.loads((ROOT / "resultados/auditoria_execucao.json").read_text())
        self.assertEqual(len(audit["folds"]), 6)
        for fold in audit["folds"]:
            self.assertEqual(fold["overlap_source_rows"], 0)
            self.assertEqual(fold["retained_originals"], fold["train"])
            self.assertEqual(fold["scaler_fit_rows"], fold["balanced"])
            self.assertTrue(fold["same_resampling_for_both_models"])
        ids = pd.read_csv(ROOT / "resultados/indices_split.csv")
        self.assertFalse(ids.origem_linha.duplicated().any())
        resampled = pd.read_csv(ROOT / "resultados/indices_treino_balanceado.csv")
        self.assertEqual(set(resampled.origem_linha),
                         set(ids.loc[ids.particao == "treino", "origem_linha"]))

    def test_clean_feature_and_predictions(self):
        d = pd.read_csv(ROOT / "dados_derivados/credito_com_feature.csv")
        self.assertFalse((d.person_emp_length >= d.person_age).any())
        self.assertTrue(np.isfinite(d.comprometimento_renda).all())
        np.testing.assert_allclose(d.comprometimento_renda, d.loan_amnt/d.person_income*100)
        final = pd.read_csv(ROOT / "resultados/avaliacao_teste.csv")
        from sklearn.metrics import confusion_matrix, f1_score
        for row in final.itertuples():
            pred = pd.read_csv(ROOT / f"resultados/avaliacao_final/{row.prefix}_predicoes.csv")
            tn, fp, fn, tp = confusion_matrix(pred.real, pred.previsto).ravel()
            self.assertEqual((tn, fp, fn, tp), (row.tn, row.fp, row.fn, row.tp))
            self.assertAlmostEqual(row.test_f1_1, f1_score(pred.real, pred.previsto), places=12)
            self.assertEqual(len(pred), 6483)

    def test_finance_reads_current_counts(self):
        final = pd.read_csv(ROOT / "resultados/avaliacao_teste.csv").set_index("model")
        costs = pd.read_csv(ROOT / "resultados/avaliacao_final/simulacao_financeira_custos.csv").set_index("model")
        np.testing.assert_array_equal(final.fp, costs.fp)
        np.testing.assert_array_equal(final.fn, costs.fn)
        np.testing.assert_array_equal(costs.custo_total_hipotetico, final.fp*1000+final.fn*5000)

    def test_importance_from_evaluated_tree(self):
        d = pd.read_csv(ROOT / "dados_derivados/credito_com_feature.csv")
        d.index = pd.read_csv(ROOT / "dados_derivados/origens.csv").origem_linha.to_numpy()
        Xtr, Xte, ytr, yte = split_data(d)
        matrices, yb, names, _, _, _, _ = prepare(Xtr, ytr, Xte)
        selected = json.loads((ROOT / "resultados/parametros_selecionados.json").read_text())["selecionados"]["Tree"]
        depth = None if selected == "None" else int(selected)
        model = classifier("Tree", depth).fit(matrices["Tree"][0], yb)
        stored = pd.read_csv(ROOT / "resultados/avaliacao_final/feature_importance_arvore.csv").set_index("feature")
        actual = pd.Series(model.feature_importances_, index=[n.split("__",1)[1] for n in names])
        np.testing.assert_allclose(stored.importance.reindex(actual.index), actual, atol=1e-12)
        row = pd.read_csv(ROOT / "resultados/avaliacao_teste.csv").set_index("model").loc["Tree"]
        pred = pd.read_csv(ROOT / f"resultados/avaliacao_final/{row.prefix}_predicoes.csv")
        np.testing.assert_array_equal(model.predict(matrices["Tree"][2]), pred.previsto)


if __name__ == "__main__":
    unittest.main()

