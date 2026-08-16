"""Equivalence against R's ``stats::glm``, when an R interpreter is available.

leapfit's model families are penalized logistic regressions, so wherever the
penalty is zero the fits must agree with R's IRLS — the de facto reference
implementation of logistic regression — to numerical precision, not to a
tolerance band. These tests build the same design in both systems on the
shipped example data and compare log-likelihoods and coefficients.

Skipped when ``Rscript`` is not on ``PATH``; only base R is used, so any R
installation suffices. A disposable environment works too::

    micromamba create -p /tmp/r-env -c conda-forge r-base
    PATH="/tmp/r-env/bin:$PATH" pytest tests/test_r_equivalence.py

The mixed-effects variants LearnSphere's PFA components fit (``lme4::glmer``)
are a different estimator class and are deliberately not compared here: a
penalized fixed-effects fit and an integrated-likelihood random-effects fit
have no agreement to require.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from leapfit import (
    build_afm_design,
    build_pfa_design,
    fit_afm,
    fit_pfa,
    load_student_step,
    success_failure_counts,
)

RSCRIPT = shutil.which("Rscript")
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "student-step.txt"

pytestmark = pytest.mark.skipif(
    RSCRIPT is None or not EXAMPLE.exists(),
    reason="needs Rscript on PATH and the shipped example data",
)

#: Fit tightly on both sides: R's IRLS converges on the deviance to ~1e-8
#: relative, so the comparison must not be limited by our optimizer instead.
TIGHT = {"method": "L-BFGS-B", "max_fun": 500_000, "tol": 1e-14,
         "warn_not_converged": False}


@pytest.fixture(scope="module")
def data():
    return load_student_step(str(EXAMPLE), kc_model="Topics")


def _run_r(script: str, tmp_path: Path) -> dict[str, float]:
    """Run an R script and parse ``name value`` lines from its stdout."""
    path = tmp_path / "check.R"
    path.write_text(script)
    result = subprocess.run([RSCRIPT, "--vanilla", str(path)],
                            capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    out = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = float(parts[1])
    return out


def _write_long(data, tmp_path: Path) -> Path:
    """One row per (observation, KC): y, student, kc, prior s and f counts."""
    s, f = success_failure_counts(data)
    csv = tmp_path / "long.csv"
    with csv.open("w") as fh:
        fh.write("y,student,kc,s,f\n")
        for i in range(len(data)):
            for kc, si, fi in zip(data.kcs[i], s[i], f[i]):
                fh.write(f"{int(data.y[i])},{data.students[i]},{kc},{si},{fi}\n")
    return csv


GLM = """
d <- read.csv("{csv}")
d$kc <- factor(d$kc)
m <- glm({formula}, family = binomial(), data = d,
         control = glm.control(epsilon = 1e-12, maxit = 100))
cat("loglik", sprintf("%.10f", as.numeric(logLik(m))), "\\n")
co <- coef(m)
for (i in seq_along(co)) cat(names(co)[i], sprintf("%.10f", co[i]), "\\n")
"""


def test_per_kc_pfa_matches_r_glm(data, tmp_path):
    """Same design, two independent implementations, one answer."""
    csv = _write_long(data, tmp_path)
    r = _run_r(GLM.format(csv=csv, formula="y ~ 0 + kc + kc:s + kc:f"), tmp_path)

    design = build_pfa_design(data)
    assert len(design.aliased) == 0, "aliased columns would break the term mapping"
    fit = fit_pfa(design, data.y, **TIGHT)

    assert fit.ll_unpenalized == pytest.approx(r.pop("loglik"), abs=1e-8)
    ours = dict(zip(design.columns, fit.weights))
    for term, estimate in r.items():
        kc, suffix = (term[2:-2], term[-1]) if term.endswith((":s", ":f")) else (term[2:], "b")
        column = {"s": f"kc_success:{kc}", "f": f"kc_failure:{kc}",
                  "b": f"kc_intercept:{kc}"}[suffix]
        assert ours[column] == pytest.approx(estimate, abs=1e-6), column


def test_pooled_pfa_matches_r_glm(data, tmp_path):
    csv = _write_long(data, tmp_path)
    r = _run_r(GLM.format(csv=csv, formula="y ~ 0 + kc + s + f"), tmp_path)

    fit = fit_pfa(build_pfa_design(data, slopes="pooled"), data.y, **TIGHT)
    assert fit.ll_unpenalized == pytest.approx(r["loglik"], abs=1e-8)
    assert fit.block("success")[0] == pytest.approx(r["s"], abs=1e-6)
    assert fit.block("failure")[0] == pytest.approx(r["f"], abs=1e-6)


def test_afm_matches_r_glm_in_likelihood_and_predictions(data, tmp_path):
    """AFM's identified design spans the same space as R's treatment coding,
    so the maximized likelihood and every fitted probability must agree even
    though the parameterizations differ."""
    csv = tmp_path / "afm.csv"
    T = data.recomputed_opportunities()
    with csv.open("w") as fh:
        fh.write("y,student,kc,t\n")
        for i in range(len(data)):
            for kc, t in zip(data.kcs[i], T[i]):
                fh.write(f"{int(data.y[i])},{data.students[i]},{kc},{t}\n")

    script = """
d <- read.csv("{csv}")
d$kc <- factor(d$kc); d$student <- factor(d$student)
m <- glm(y ~ student + kc + kc:t, family = binomial(), data = d,
         control = glm.control(epsilon = 1e-12, maxit = 100))
cat("loglik", sprintf("%.10f", as.numeric(logLik(m))), "\\n")
write.csv(data.frame(p = fitted(m)), "{fitted}", row.names = FALSE)
""".format(csv=csv, fitted=tmp_path / "fitted.csv")
    r = _run_r(script, tmp_path)

    design = build_afm_design(data, recompute_opportunities=True)
    fit = fit_afm(design, data.y, **TIGHT)
    assert fit.ll_unpenalized == pytest.approx(r["loglik"], abs=1e-8)

    # The MLE's fitted values are unique even where coefficients are not; the
    # tolerance covers each optimizer stopping within its own criterion.
    r_fitted = np.loadtxt(tmp_path / "fitted.csv", skiprows=1)
    np.testing.assert_allclose(fit.predict_proba(design), r_fitted, atol=2e-6)
