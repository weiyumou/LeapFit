"""The console scripts, run in-process: what each one writes, and what it
refuses. ``test_package`` runs them as subprocesses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leapfit.cli import main as cli_main
from leapfit.cli import main_lfa, main_pfa

from helpers import EXAMPLE, minimal_frame, mixed, rollup, simulate_pfa

# --------------------------------------------------------------------------
# leapfit-afm
# --------------------------------------------------------------------------


def test_cli_predictions_writes_one_column_per_model(tmp_path):
    df = minimal_frame()
    df["KC (M2)"] = df["KC (M)"]
    df["Opportunity (M2)"] = df["Opportunity (M)"]
    export = tmp_path / "export.txt"
    df.to_csv(export, sep="\t", index=False, lineterminator="\n")

    out = tmp_path / "annotated.txt"
    assert cli_main([str(export), "--cv", "none", "--predictions", str(out)]) == 0

    written = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert len(written) == len(df)
    for col in ("Predicted Error Rate (M)", "Predicted Error Rate (M2)"):
        assert col in written.columns
        assert (written[col] != "").all()
    # Identical KC models must produce identical predictions.
    assert written["Predicted Error Rate (M)"].tolist() == \
        written["Predicted Error Rate (M2)"].tolist()


def _cli_export(tmp_path, name="export.txt"):
    export = tmp_path / name
    minimal_frame().to_csv(export, sep="\t", index=False, lineterminator="\n")
    return export


def test_cli_one_scheme_keeps_the_unsuffixed_column_names(tmp_path):
    """Existing invocations, and anything parsing their CSV, must not move."""
    out = tmp_path / "table.csv"
    assert cli_main([str(_cli_export(tmp_path)), "--cv", "item_blocked",
                     "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert {"cv_rmse", "cv_rmse_sd", "cv_runs"} <= set(table.columns)
    assert not [c for c in table.columns if c.endswith("_item_blocked")]


def test_cli_scores_several_schemes_on_one_fit(tmp_path):
    out = tmp_path / "table.csv"
    assert cli_main([str(_cli_export(tmp_path)),
                     "--cv", "student_blocked", "--cv", "item_blocked",
                     "--out", str(out)]) == 0
    table = pd.read_csv(out)
    assert {"cv_rmse_student_blocked", "cv_rmse_item_blocked"} <= set(table.columns)
    assert "cv_rmse" not in table.columns, "no single score to name any more"
    assert table["cv_rmse_student_blocked"].notna().all()


def test_cli_none_cannot_be_combined_with_a_scheme(tmp_path):
    assert cli_main([str(_cli_export(tmp_path)), "--cv", "none",
                     "--cv", "item_blocked"]) == 1


def test_cli_cv_folds_writes_one_row_per_model_scheme_and_seed(tmp_path):
    folds = tmp_path / "cv-folds.csv"
    assert cli_main([str(_cli_export(tmp_path)),
                     "--cv", "student_blocked", "--cv", "item_blocked",
                     "--seeds", "0:3", "--cv-folds", str(folds)]) == 0
    detail = pd.read_csv(folds)
    assert len(detail) == 1 * 2 * 3, "one KC model x two schemes x three seeds"
    assert set(detail["scheme"]) == {"student_blocked", "item_blocked"}
    assert detail["kc_model"].unique().tolist() == ["M"]


def test_cli_cv_folds_falls_back_to_per_fold_rows_without_seeds(tmp_path):
    folds = tmp_path / "cv-folds.csv"
    assert cli_main([str(_cli_export(tmp_path)), "--cv", "item_blocked",
                     "--folds", "3", "--cv-folds", str(folds)]) == 0
    detail = pd.read_csv(folds)
    assert len(detail) == 3 and set(detail["fold"]) == {0, 1, 2}
    assert detail["scheme"].unique().tolist() == ["item_blocked"]


def test_cli_identification_writes_every_aliased_column(tmp_path):
    path = tmp_path / "identification.csv"
    assert cli_main([str(_cli_export(tmp_path)), "--cv", "none",
                     "--identification", str(path)]) == 0
    dropped = pd.read_csv(path)
    assert dropped["column"].str.startswith("student:").any()
    assert dropped["reason"].str.contains("reference level").any()


def test_cli_identification_writes_a_header_when_nothing_is_aliased(tmp_path):
    """An empty file is a result; a missing one is indistinguishable from a
    run that never asked for it."""
    path = tmp_path / "identification.csv"
    assert cli_main([str(_cli_export(tmp_path, "compat.txt")), "--cv", "none",
                     "--learnsphere-compat", "--identification", str(path)]) == 0
    dropped = pd.read_csv(path)
    assert dropped.empty
    assert dropped.columns.tolist() == ["kc_model", "column", "reason"]


# --------------------------------------------------------------------------
# leapfit-afm: models over the same rows are paired
# --------------------------------------------------------------------------


def _two_model_export(tmp_path, shared_rows=True):
    """An export carrying KC models M and M2 — identical mappings, unless
    ``shared_rows=False`` blanks one of M2's labels so their coverage differs."""
    df = minimal_frame()
    df["KC (M2)"] = df["KC (M)"]
    df["Opportunity (M2)"] = df["Opportunity (M)"]
    if not shared_rows:
        df.loc[df.index[3], ["KC (M2)", "Opportunity (M2)"]] = ""
    export = tmp_path / "export.txt"
    df.to_csv(export, sep="\t", index=False, lineterminator="\n")
    return export


def test_cli_pairs_models_that_share_rows(tmp_path, capsys):
    """Two models over the same rows: shared folds, contrasts, full detail."""
    out, contrasts, folds = (tmp_path / n for n in
                             ("table.csv", "contrasts.csv", "folds.csv"))
    assert cli_main([str(_two_model_export(tmp_path)), "--cv", "item_blocked",
                     "--out", str(out), "--contrasts", str(contrasts),
                     "--cv-folds", str(folds)]) == 0
    assert "paired, folds shared" in capsys.readouterr().err

    table = pd.read_csv(out)
    assert {"cv_rmse", "cv_rmse_sd", "cv_runs"} <= set(table.columns)
    assert table["cv_runs"].tolist() == [1, 1], "default paired protocol: seed 0"

    detail = pd.read_csv(folds)
    assert len(detail) == 2 * 3, "two models x three folds x one seed"
    sizes = detail.pivot_table(index=["seed", "fold"], columns="kc_model",
                               values="n_test")
    np.testing.assert_array_equal(sizes["M"].to_numpy(), sizes["M2"].to_numpy())

    c = pd.read_csv(contrasts)
    assert len(c) == 1 and {c["model"].iloc[0], c["baseline"].iloc[0]} == {"M", "M2"}
    # Identical KC models fit identically, so the paired difference vanishes —
    # which only shared folds can show exactly.
    assert c["mean_diff"].abs().max() < 1e-12


def test_cli_paired_baseline_can_be_named(tmp_path):
    contrasts = tmp_path / "contrasts.csv"
    assert cli_main([str(_two_model_export(tmp_path)), "--cv", "item_blocked",
                     "--baseline", "M2", "--contrasts", str(contrasts)]) == 0
    c = pd.read_csv(contrasts)
    assert c["baseline"].unique().tolist() == ["M2"]
    assert cli_main([str(_two_model_export(tmp_path)),
                     "--baseline", "nope"]) == 1


def test_cli_falls_back_when_models_cover_different_rows(tmp_path, capsys):
    """Equal exports, unequal coverage: paired CV is impossible, and the run
    must say so rather than silently comparing across different splits."""
    contrasts = tmp_path / "contrasts.csv"
    assert cli_main([str(_two_model_export(tmp_path, shared_rows=False)),
                     "--cv", "item_blocked", "--contrasts", str(contrasts)]) == 0
    err = capsys.readouterr().err
    assert "cover different rows" in err
    assert not contrasts.exists(), "no paired contrasts without shared folds"


def test_cli_no_paired_restores_independent_folds(tmp_path, capsys):
    out = tmp_path / "table.csv"
    assert cli_main([str(_two_model_export(tmp_path)), "--cv", "item_blocked",
                     "--no-paired", "--out", str(out)]) == 0
    assert "independent folds per model" in capsys.readouterr().err
    assert pd.read_csv(out)["cv_rmse"].notna().all()


# --------------------------------------------------------------------------
# leapfit-pfa
# --------------------------------------------------------------------------


def test_the_pfa_cli_end_to_end(tmp_path):
    data, _ = simulate_pfa(n_students=8, n_reps=6)
    export = tmp_path / "export.txt"
    data.source.to_csv(export, sep="\t", index=False, lineterminator="\n")

    out_dir = tmp_path / "kc"
    assert main_pfa([str(export), "--cv", "none", "--pooled-slopes",
                     "--kc-values", str(out_dir)]) == 0
    written = pd.read_csv(out_dir / "M_kc-values.csv")
    assert "Success Slope" in written.columns and "Failure Slope" in written.columns


# --------------------------------------------------------------------------
# leapfit-lfa
# --------------------------------------------------------------------------


def _lfa_cli(*extra, export=EXAMPLE):
    return main_lfa([export, "--max-iterations", "2", "--validate", "0", *extra])


def test_cli_lfa_writes_the_frontier_and_the_refusals(tmp_path):
    frontier, refusals = tmp_path / "f.csv", tmp_path / "r.csv"
    assert _lfa_cli("--out", str(frontier), "--refusals", str(refusals)) == 0
    table = pd.read_csv(frontier)
    assert {"rank", "n_kcs", "n_params", "bic", "is_optimal", "history"} <= set(table.columns)
    assert table["rank"].tolist() == list(range(1, len(table) + 1))
    # Written even when nothing was refused: an empty file is a result, a
    # missing one cannot be told from a run that never asked.
    assert set(pd.read_csv(refusals).columns) == {"move", "reason"}


def test_cli_lfa_writes_a_kc_model_ready_to_join(tmp_path):
    out = tmp_path / "q.txt"
    assert _lfa_cli("--qmatrix", str(out), "--kc-model-name", "Discovered") == 0
    table = pd.read_csv(out, sep="\t", dtype=str)
    assert list(table.columns) == ["Problem Name", "Step Name", "KC (Discovered)"]
    assert len(table) == 40, "one row per step, not per observation"
    assert table["KC (Discovered)"].nunique() >= 2, "the search split something"


def test_cli_lfa_annotates_the_export_under_the_discovered_model(tmp_path):
    out = tmp_path / "annotated.txt"
    assert _lfa_cli("--predictions", str(out), "--kc-model-name", "Found") == 0
    written = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert "Predicted Error Rate (Found)" in written.columns
    assert (written["Predicted Error Rate (Found)"] != "").all()
    assert "KC (Topics)" in written.columns, "the export round-trips verbatim"


def test_cli_lfa_validates_the_shortlist_and_writes_it(tmp_path):
    out = tmp_path / "v.csv"
    assert main_lfa([EXAMPLE, "--max-iterations", "2", "--validate", "2",
                     "--compare", "Topics", "--validation", str(out)]) == 0
    table = pd.read_csv(out)
    assert {"cv_rmse", "search_rank", "cv_rank"} <= set(table.columns)
    assert "Topics" in set(table["model"]), "--compare joins the comparison"
    assert "root" in set(table["model"])


def test_cli_lfa_says_so_rather_than_writing_an_empty_validation(tmp_path, capsys):
    out = tmp_path / "v.csv"
    assert _lfa_cli("--validation", str(out)) == 0
    assert not out.exists()
    assert "--validate 0 skipped it" in capsys.readouterr().err


def test_cli_lfa_excludes_a_multi_kc_model_from_the_factors(tmp_path, capsys):
    """The reference aborts; this reports and proceeds with what is eligible."""
    rows = mixed("s1", "a", "X", 4) + mixed("s1", "b", "Y", 4) + \
        mixed("s2", "a", "X", 4) + mixed("s2", "b", "Y", 4)
    df = rollup(rows, "clean")
    df["KC (wide)"] = "P~~Q"
    df["Opportunity (wide)"] = df["Opportunity (clean)"] + "~~1"
    export = tmp_path / "export.txt"
    df.to_csv(export, sep="\t", index=False, lineterminator="\n")

    assert _lfa_cli("--min-opportunities", "1", export=str(export)) == 0
    err = capsys.readouterr().err
    assert "excluding 'wide'" in err and "more than one KC" in err


def test_cli_lfa_refuses_an_unknown_model_name(tmp_path, capsys):
    assert _lfa_cli("--factors", "Nope") == 1
    assert "Unknown KC model" in capsys.readouterr().err
    assert _lfa_cli("--compare", "Nope") == 1
    assert "is not a KC model" in capsys.readouterr().err


def test_cli_lfa_lists_the_models_and_exits(capsys):
    assert main_lfa([EXAMPLE, "--list-models"]) == 0
    assert capsys.readouterr().out.split() == ["Skills", "Topics"]


def test_the_cli_offers_the_merge_operators(tmp_path):
    out = tmp_path / "f.csv"
    assert _lfa_cli("--merges", "both", "--out", str(out)) == 0
    assert len(pd.read_csv(out)) > 1


def test_the_cli_can_start_from_an_authored_kc_model(tmp_path):
    out = tmp_path / "f.csv"
    assert _lfa_cli("--root", "Skills", "--merges", "pairwise",
                    "--out", str(out)) == 0
    assert len(pd.read_csv(out)) > 1
