"""Tiny helpers for the self-check cells, so a student gets a green light without ever
seeing the answer.

    from aeolis.checks import done
    done(mean_temp)                       # did you replace the ... placeholder?
    assert 60 < mean_temp < 70            # is it right?
    print("correct")

The asserts in the notebooks are written out in full on purpose: a student who reads
`assert 60 < mean_temp < 70` learns what a plausible answer looks like, which is more
useful than a helper that hides the criterion.
"""

from __future__ import annotations


def done(*values) -> None:
    """Fail with a clear message if any TODO placeholder (`...`) is still unfilled.

    Without this, an unfilled `...` produces a TypeError from the next comparison, which
    reads to a beginner as "my code is broken" rather than "I have not written it yet".
    """
    for i, value in enumerate(values, 1):
        if value is Ellipsis:
            raise AssertionError(
                f"Placeholder {i} is still `...` - fill in the TODO above, then rerun this cell."
            )


def close(actual: float, expected: float, tol: float = 1e-6) -> bool:
    """Float comparison for check cells. `0.1 + 0.2 == 0.3` is False, and a beginner
    should meet that fact once, in a controlled place, rather than debug it at random."""
    return abs(float(actual) - float(expected)) <= tol
