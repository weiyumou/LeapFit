"""The install check: that ``leapfit`` is a package, not just a directory.

These tests are deliberately cheap and data-free, so they run as the first
thing after ``pip install leapfit`` and fail loudly if the packaging metadata
and the code have drifted apart. Everything here would have passed silently as
a source checkout while a built wheel was broken.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

import leapfit

REPO = Path(__file__).resolve().parent.parent

#: Modules the package promises. Shared infrastructure first, then one module
#: per model family — the split that makes adding PFA/BKT/IRT a new module
#: rather than a rewrite.
SHARED = ["leapfit.data", "leapfit.design", "leapfit.fit", "leapfit.crossval"]
FAMILIES = ["leapfit.afm", "leapfit.pfa"]

#: Modules that sit *above* a family rather than beside it. A search over KC
#: models is scored by AFM's own AIC/BIC, so ``leapfit.lfa`` imports
#: ``leapfit.afm`` deliberately — and nothing below it may import back.
CONSUMERS = ["leapfit.lfa"]


def test_every_promised_module_imports():
    for name in [*SHARED, *FAMILIES, *CONSUMERS, "leapfit.cli"]:
        assert importlib.import_module(name) is not None, name


def test_public_api_is_complete():
    """Every name in ``__all__`` resolves, and nothing is listed twice.

    Ordering is not checked here — ruff's RUF022 already enforces it, and
    duplicating that rule by hand got its convention wrong.
    """
    missing = [n for n in leapfit.__all__ if not hasattr(leapfit, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"
    duplicates = {n for n in leapfit.__all__ if leapfit.__all__.count(n) > 1}
    assert not duplicates, f"duplicated in __all__: {sorted(duplicates)}"


def test_star_import_matches_the_declared_api():
    """``from leapfit import *`` yields exactly ``__all__`` and nothing more."""
    namespace: dict = {}
    exec("from leapfit import *", namespace)
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(leapfit.__all__) - {"__version__"}


def test_version_agrees_with_pyproject():
    """A wheel whose metadata version differs from ``__version__`` is a trap."""
    with (REPO / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert leapfit.__version__ == declared


def test_the_model_layer_depends_on_the_shared_layer_and_not_the_reverse():
    """The layering that makes a second model family cheap.

    ``leapfit.afm`` may import the shared modules; the shared modules must not
    import ``leapfit.afm``. If this inverts, adding PFA means editing the
    solver instead of adding a file.
    """
    for name in SHARED:
        source = (REPO / f"{name.replace('.', '/')}.py").read_text()
        for family in FAMILIES:
            assert f"import {family}" not in source and f"from {family}" not in source, (
                f"{name} imports {family}; the shared layer must not know about "
                "a specific model family")


def test_shared_modules_carry_no_family_specific_api():
    """Each family's design builder and reporting live in its own module."""
    import leapfit.design
    import leapfit.fit

    for module in (leapfit.design, leapfit.fit):
        for symbol in ("build_afm_design", "AFMFit", "build_pfa_design",
                       "PFAFit", "kc_values", "success_failure_counts"):
            assert not hasattr(module, symbol), (
                f"{module.__name__} exposes {symbol}, which is family-specific")


def test_the_family_modules_do_not_import_each_other():
    """AFM and PFA are siblings over the shared layer, not layered on each other."""
    for name in FAMILIES:
        source = (REPO / f"{name.replace('.', '/')}.py").read_text()
        for other in FAMILIES:
            if other == name:
                continue
            assert f"import {other}" not in source and f"from {other}" not in source, (
                f"{name} imports {other}")


def test_nothing_below_a_search_imports_it():
    """The edge into ``leapfit.lfa`` runs one way only.

    A search consumes a model family; if the shared layer or a family ever
    imports the search back, the layering that keeps a new model family a new
    file has inverted, and `leapfit.lfa` becomes load-bearing for fits that
    have nothing to do with a search.
    """
    for name in [*SHARED, *FAMILIES]:
        source = (REPO / f"{name.replace('.', '/')}.py").read_text()
        for consumer in CONSUMERS:
            assert (f"import {consumer}" not in source
                    and f"from {consumer}" not in source), (
                f"{name} imports {consumer}; a search sits above a family, "
                "never below one")


def test_a_search_module_may_import_the_family_it_scores_with():
    """The complement, stated so the one-way rule is not read as "no edge"."""
    source = (REPO / "leapfit/lfa.py").read_text()
    assert "from leapfit.afm import" in source, (
        "leapfit.lfa scores states with AFM; that import is the design")


@pytest.mark.parametrize("entry", ["leapfit.cli"])
def test_the_cli_entry_point_runs(entry):
    """``python -m`` the module the console script points at."""
    result = subprocess.run(
        [sys.executable, "-c", f"import {entry}; raise SystemExit({entry}.main(['--help']))"],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--kc-model" in result.stdout


def test_the_search_entry_point_runs():
    """``leapfit-lfa`` is a separate entry point, so it needs its own check."""
    script = "import leapfit.cli as c; raise SystemExit(c.main_lfa(['--help']))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--factors" in result.stdout and "--qmatrix" in result.stdout


def test_console_scripts_are_declared():
    with (REPO / "pyproject.toml").open("rb") as fh:
        scripts = tomllib.load(fh)["project"]["scripts"]
    assert scripts["leapfit-afm"] == "leapfit.cli:main"
    assert scripts["leapfit-pfa"] == "leapfit.cli:main_pfa"
    assert scripts["leapfit-lfa"] == "leapfit.cli:main_lfa"
