"""
martingale.exact — property-test your off-policy estimator in exact arithmetic.

    from fractions import Fraction
    from martingale.exact import check_unbiased, defendants

    report = check_unbiased(defendants.truncated_is(cap=Fraction(2)))
    print(report.all_unbiased, report.certificates[0].rel_sq_bias)

A user-defined estimator is any callable (traj, target, behavior) -> dict
mapping (state, action) -> Fraction; see martingale.exact.defendants for the
trajectory format and the built-in defendants.
"""
from martingale.exact import defendants
from martingale.exact.bench import (
    DEFAULT_LAGS,
    DEFAULT_SIZES,
    BenchReport,
    BiasCertificate,
    check_unbiased,
    expected_gradient,
)
from martingale.exact.defendants import Estimator

__all__ = [
    "BenchReport", "BiasCertificate", "Estimator", "check_unbiased",
    "expected_gradient", "defendants", "DEFAULT_LAGS", "DEFAULT_SIZES",
]
