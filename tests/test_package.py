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
FAMILIES = ["leapfit.afm", "leapfit.pfa", "leapfit.lkt"]

#: Modules that sit *above* a family rather than beside it. A search over KC
#: models is scored by AFM's own AIC/BIC, so ``leapfit.lfa`` imports
#: ``leapfit.afm`` deliberately — and nothing below it may import back.
CONSUMERS = ["leapfit.lfa"]


def test_every_promised_module_imports():
    for name in [*SHARED, *FAMILIES, *CONSUMERS, "leapfit.cli"]:
        assert importlib.import_module(name) is not None, name


def test_star_import_matches_the_declared_api():
    """``from leapfit import *`` yields exactly ``__all__`` and nothing more, and
    nothing is listed twice. A listed name that does not exist makes the import
    itself raise.

    Ordering is not checked here — ruff's RUF022 already enforces it, and
    duplicating that rule by hand got its convention wrong.
    """
    namespace: dict = {}
    exec("from leapfit import *", namespace)
    exported = {k for k in namespace if not k.startswith("__")}
    assert exported == set(leapfit.__all__) - {"__version__"}
    duplicates = {n for n in leapfit.__all__ if leapfit.__all__.count(n) > 1}
    assert not duplicates, f"duplicated in __all__: {sorted(duplicates)}"


def test_version_agrees_with_pyproject():
    """A wheel whose metadata version differs from ``__version__`` is a trap."""
    with (REPO / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["version"]
    assert leapfit.__version__ == declared


def _imports(importer: str, imported: str) -> bool:
    source = (REPO / f"{importer.replace('.', '/')}.py").read_text()
    return f"import {imported}" in source or f"from {imported}" in source


def test_no_module_imports_a_layer_beside_or_above_it():
    """The layering that makes a second model family cheap.

    The shared modules know no family: if that inverts, adding PFA means
    editing the solver instead of adding a file. The families are siblings
    over the shared layer, not layered on each other. And the edge into
    ``leapfit.lfa`` runs one way only: a search consumes a model family, so if
    the shared layer or a family ever imports it back, ``leapfit.lfa`` becomes
    load-bearing for fits that have nothing to do with a search.
    """
    forbidden = ([(shared, family) for shared in SHARED for family in FAMILIES]
                 + [(a, b) for a in FAMILIES for b in FAMILIES if a != b]
                 + [(below, consumer) for below in [*SHARED, *FAMILIES]
                    for consumer in CONSUMERS])
    offending = [f"{a} imports {b}" for a, b in forbidden if _imports(a, b)]
    assert not offending, offending


def test_shared_modules_carry_no_family_specific_api():
    """Each family's design builder and reporting live in its own module."""
    import leapfit.design
    import leapfit.fit

    for module in (leapfit.design, leapfit.fit):
        for symbol in ("build_afm_design", "AFMFit", "build_pfa_design",
                       "PFAFit", "kc_values", "success_failure_counts",
                       "build_lkt_design", "LKTFit", "lkt_terms"):
            assert not hasattr(module, symbol), (
                f"{module.__name__} exposes {symbol}, which is family-specific")


def test_a_search_module_may_import_the_family_it_scores_with():
    """The complement, stated so the one-way rule is not read as "no edge"."""
    assert _imports("leapfit.lfa", "leapfit.afm"), (
        "leapfit.lfa scores states with AFM; that import is the design")


#: Each console script, and options its ``--help`` must list.
SCRIPTS = {"leapfit-afm": ("--kc-model",), "leapfit-pfa": ("--pooled-slopes",),
           "leapfit-lfa": ("--factors", "--qmatrix")}


@pytest.mark.parametrize("script", sorted(SCRIPTS))
def test_each_console_script_runs(script):
    """The entry point ``[project.scripts]`` declares runs ``--help`` in a
    fresh interpreter, as the installed script would."""
    with (REPO / "pyproject.toml").open("rb") as fh:
        declared = tomllib.load(fh)["project"]["scripts"]
    assert set(declared) == set(SCRIPTS), "every declared script is checked here"
    module, function = declared[script].split(":")
    result = subprocess.run(
        [sys.executable, "-c",
         f"import {module}; raise SystemExit({module}.{function}(['--help']))"],
        capture_output=True, text=True, cwd=REPO, check=False,
    )
    assert result.returncode == 0, result.stderr
    for option in SCRIPTS[script]:
        assert option in result.stdout, option


