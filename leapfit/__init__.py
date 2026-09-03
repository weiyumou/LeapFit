"""leapfit — student models for learning analytics, over DataShop student-step data.

The implementations are grounded in and adapted from LearnSphere's reference
components, validated against LearnSphere's own output where that output
exists, so a baseline column in a paper is something you can inspect rather
than something you downloaded. Provenance and the deliberate divergences are
recorded where they apply — in each module's docstring — and the validation
record is the equivalence suite itself: ``tests/test_learnsphere_equivalence.py``
against LearnSphere workflow output, ``tests/test_r_equivalence.py`` against
R's ``stats::glm``.

**One input format.** Every model here reads the same six columns of a DataShop
student-step export — ``Anon Student Id``, ``Problem Name``, ``Step Name``,
``First Attempt``, ``KC (<model>)``, ``Opportunity (<model>)`` — so switching
model families never means reshaping data.

    from leapfit import load_student_step, build_afm_design, fit_afm, cross_validate

    data   = load_student_step("examples/student-step.txt", kc_model="Topics")
    design = build_afm_design(data)                 # or build_pfa_design(data)
    fit    = fit_afm(design, data.y)                # or fit_pfa(...)
    print(fit.summary())
    print(cross_validate(design, data, scheme="item_blocked").summary())

Layout — shared infrastructure, one module per model family, then searches
over the KC model a family conditions on:

    leapfit.data      the export -> StepData (parsing rules, practice order)
    leapfit.design    Block / Design: columns carrying their own penalty+bounds
    leapfit.fit       the penalized-logistic solver and its KKT certificate
    leapfit.crossval  fold schemes and both RMSE conventions
    leapfit.afm       Additive Factors Model
    leapfit.pfa       Performance Factors Analysis
    leapfit.lfa       Learning Factors Analysis: a search over KC models,
                      scored by AFM
"""

__version__ = "0.4.0"

from leapfit.afm import (
    STUDENT_L2,
    AFMFit,
    build_afm_design,
    fit_afm,
)
from leapfit.crossval import (
    CONVENTIONS,
    SCHEMES,
    CVResult,
    cross_validate,
    make_folds,
    paired_contrasts,
    paired_cross_validate,
    paired_scores,
    repeated_cross_validate,
)
from leapfit.data import (
    FIRST_ATTEMPT_VALUES,
    StepData,
    from_frame,
    list_kc_models,
    load_student_step,
)
from leapfit.design import (
    Aliased,
    Block,
    Design,
    Separated,
    accumulator_block,
    coefficient_frame,
)
from leapfit.fit import DEFAULT_METHOD, LogisticFit, fit_logistic
from leapfit.lfa import (
    HEURISTICS,
    MIN_OPPORTUNITIES,
    FactorMatrix,
    LFAResult,
    LFAState,
    Move,
    Rejected,
    build_factor_matrix,
    lfa_search,
)
from leapfit.pfa import (
    PFAFit,
    build_pfa_design,
    fit_pfa,
    success_failure_counts,
)

__all__ = [
    "CONVENTIONS",
    "DEFAULT_METHOD",
    "FIRST_ATTEMPT_VALUES",
    "HEURISTICS",
    "MIN_OPPORTUNITIES",
    "SCHEMES",
    "STUDENT_L2",
    "AFMFit",
    "Aliased",
    "Block",
    "CVResult",
    "Design",
    "FactorMatrix",
    "LFAResult",
    "LFAState",
    "LogisticFit",
    "Move",
    "PFAFit",
    "Rejected",
    "Separated",
    "StepData",
    "__version__",
    "accumulator_block",
    "build_afm_design",
    "build_factor_matrix",
    "build_pfa_design",
    "coefficient_frame",
    "cross_validate",
    "fit_afm",
    "fit_logistic",
    "fit_pfa",
    "from_frame",
    "lfa_search",
    "list_kc_models",
    "load_student_step",
    "make_folds",
    "paired_contrasts",
    "paired_cross_validate",
    "paired_scores",
    "repeated_cross_validate",
    "success_failure_counts",
]
