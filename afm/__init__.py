"""A local Additive Factors Model, matched to LearnSphere's PyAFM.

Why this exists: LearnSphere fits vanilla AFM from an uploaded Q-matrix and
nothing else. Every model in this project's line of work — a congruity-weighted
practice term, hierarchical shrinkage over KC granularity, kernel-smoothed item
parameters — is a change to the design matrix or the penalty, so all of them
need a fitter we control. This one reproduces the reference objective exactly
(see ``model.py``) so that "AFM" stays a trustworthy baseline column.

    from afm import load_student_step, build_afm_design, fit_afm, cross_validate

    data = load_student_step("ds5426_student_step.txt", kc_model="LOs-new")
    design = build_afm_design(data)
    fit = fit_afm(design, data.y)
    print(fit.summary())
    print(cross_validate(design, data, scheme="item_blocked").summary())
"""

from afm.crossval import (
    CONVENTIONS,
    SCHEMES,
    CVResult,
    cross_validate,
    make_folds,
    paired_contrasts,
    paired_cross_validate,
    repeated_cross_validate,
)
from afm.data import StepData, from_frame, list_kc_models, load_student_step
from afm.design import Aliased, Block, Design, build_afm_design, congruity_block
from afm.model import AFMFit, fit_afm

__all__ = [
    "CONVENTIONS",
    "SCHEMES",
    "AFMFit",
    "Aliased",
    "Block",
    "CVResult",
    "Design",
    "StepData",
    "build_afm_design",
    "congruity_block",
    "cross_validate",
    "fit_afm",
    "from_frame",
    "list_kc_models",
    "load_student_step",
    "make_folds",
    "paired_contrasts",
    "paired_cross_validate",
    "repeated_cross_validate",
]
