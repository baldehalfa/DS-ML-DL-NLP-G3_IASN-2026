"""Evaluation helpers - the part of the course that students must not get wrong.

Deliberately small and readable: students are expected to open this file and see that
there is no magic in a metric, a threshold or a backtest.

    from aeolis import evaluate as ev
    ev.expected_cost(y_true, y_prob, threshold=0.3, cost_fp=500, cost_fn=20000)
    ev.best_threshold(y_true, y_prob, cost_fp=500, cost_fn=20000)
    for train_idx, test_idx in ev.walk_forward(len(series), n_folds=5, horizon=24): ...
    ev.hit_at_k(retrieved, relevant, k=3)
    ev.groundedness(answer, context)
"""

from __future__ import annotations

import re

import numpy as np


# ------------------------------------------------------- classification with a cost

def confusion_counts(y_true, y_prob, threshold: float = 0.5) -> dict[str, int]:
    """Counts at a chosen threshold. A confusion matrix is a decision, not a metric."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
    }


def expected_cost(y_true, y_prob, threshold: float, cost_fp: float, cost_fn: float) -> float:
    """Total cost of the decisions this threshold produces, in whatever unit you priced.

    This is the metric that actually matters to the maintenance department: a false alarm
    costs one technician visit, a missed failure costs a gearbox. Accuracy prices them
    the same, which is why accuracy is the wrong objective here.
    """
    counts = confusion_counts(y_true, y_prob, threshold)
    return counts["fp"] * cost_fp + counts["fn"] * cost_fn


def best_threshold(
    y_true, y_prob, cost_fp: float, cost_fn: float, grid: int = 101
) -> tuple[float, float]:
    """Return (threshold, cost) minimising expected cost over a grid of thresholds.

    Note what this shows students: 0.5 is a default, not a decision. Under a strongly
    asymmetric cost ratio the optimum lands nowhere near the middle, and it cuts the bill
    sharply on the same trained model. The measured threshold and both costs are recorded in
    docs/verified-teaching-results.md, not here: students read this module, and
    Session 2 has them commit to a guess first.
    """
    thresholds = np.linspace(0.0, 1.0, grid)
    costs = [expected_cost(y_true, y_prob, t, cost_fp, cost_fn) for t in thresholds]
    best = int(np.argmin(costs))
    return float(thresholds[best]), float(costs[best])


def alarm_rate(y_true, y_prob, threshold: float) -> dict[str, float]:
    """How this threshold behaves for the person who has to answer the alarms.

    The cost-optimal threshold is frequently unusable: it fires several alarms for every
    real failure caught, at a rate no maintenance team can work through, whatever the
    spreadsheet says. The measured counts are in docs/verified-teaching-results.md.
    """
    counts = confusion_counts(y_true, y_prob, threshold)
    alarms = counts["tp"] + counts["fp"]
    n = len(np.asarray(y_true))
    positives = counts["tp"] + counts["fn"]
    # Rounded, because students print this dictionary directly and
    # `alarms_per_catch: 3.5308641975308643` is noise in front of a class.
    return {
        "alarms": alarms,
        "caught": counts["tp"],
        "missed": counts["fn"],
        "alarms_per_catch": round(alarms / counts["tp"], 1) if counts["tp"] else float("inf"),
        "periods_between_alarms": round(n / alarms, 1) if alarms else float("inf"),
        "recall": round(counts["tp"] / positives, 3) if positives else float("nan"),
    }


def best_threshold_under_capacity(
    y_true, y_prob, cost_fp: float, cost_fn: float, max_alarms: int, grid: int = 201
) -> tuple[float, float]:
    """Cheapest threshold that fires no more than `max_alarms` alarms.

    This is the Level-5 exercise: the maintenance team can investigate two alarms a week,
    so unconstrained cost minimisation is the wrong problem. Optimising under a capacity
    constraint is what an engineer actually does, and it is nearly always absent from
    textbook treatments of thresholds.

    Returns (threshold, cost). Raises if no threshold satisfies the constraint.
    """
    feasible = [
        t
        for t in np.linspace(0.0, 1.0, grid)
        if alarm_rate(y_true, y_prob, t)["alarms"] <= max_alarms
    ]
    if not feasible:
        raise ValueError(f"no threshold fires at most {max_alarms} alarms")
    best = min(feasible, key=lambda t: expected_cost(y_true, y_prob, t, cost_fp, cost_fn))
    return float(best), expected_cost(y_true, y_prob, best, cost_fp, cost_fn)


# ------------------------------------------------------------------- time series

def walk_forward(
    n: int,
    n_folds: int = 5,
    horizon: int = 24,
    min_train: int | None = None,
    stride: int | None = None,
):
    """Yield (train_idx, test_idx) for an expanding-window backtest.

    The training window only ever grows, and every test index is strictly after every
    training index. This is the whole reason Session 4 exists: a shuffled k-fold would
    train on the future and report a score that cannot happen in production.

    `stride` is the gap between the start of one test fold and the next; it defaults to
    `horizon`, which places the folds back to back at the end of the series. Set it larger
    to spread the folds across the whole history - with hourly data, `stride=24*30` tests
    one day per month. That matters: eight consecutive days in November tells you how the
    model does in November, and a forecast evaluated in one season is a forecast you know
    nothing about in the others.
    """
    stride = horizon if stride is None else stride
    span = (n_folds - 1) * stride + horizon
    if min_train is None:
        min_train = max(horizon, n - span)
    if min_train + span > n:
        raise ValueError(
            f"not enough data: {n} points cannot support {n_folds} folds of horizon "
            f"{horizon} at stride {stride} after a {min_train}-point initial training window"
        )
    for fold in range(n_folds):
        train_end = min_train + fold * stride
        yield np.arange(0, train_end), np.arange(train_end, train_end + horizon)


def seasonal_naive(series, season_length: int, horizon: int) -> np.ndarray:
    """The baseline that beats a surprising number of forecasting models."""
    series = np.asarray(series, dtype=float)
    if len(series) < season_length:
        raise ValueError("series shorter than one season")
    last_season = series[-season_length:]
    return np.array([last_season[i % season_length] for i in range(horizon)])


def forecast_errors(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and MAPE together - because each one hides something different."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    errors = y_pred - y_true
    nonzero = y_true != 0
    mape = float(np.mean(np.abs(errors[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape_pct": mape,  # undefined near zero - that is the point, not a bug
    }


# -------------------------------------------------------------------- RAG / LLM

def hit_at_k(retrieved_ids, relevant_ids, k: int = 3) -> bool:
    """Did retrieval put at least one relevant chunk in the top k?

    Measured before looking at the generated answer. If hit@k is 0, the answer was
    doomed regardless of which LLM produced it - that is the attribution students must
    learn to make.
    """
    return bool(set(list(retrieved_ids)[:k]) & set(relevant_ids))


def _words(text: str) -> set[str]:
    """Content tokens, keeping short ones because engineering answers are mostly numbers.

    The first version required len > 3, which threw away short numeric tokens - so a
    perfectly correct answer made of a grade and a number scored zero groundedness. Keeping
    tokens of length >= 2 fixes the worst of it. It is still a proxy, and the notebook says
    so.
    """
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2}


def groundedness(answer: str, context: str) -> float:
    """Crude proxy for "is this answer supported by the retrieved context?" in [0, 1].

    Lexical overlap only: it cannot detect a fluent paraphrase of a fabrication, and it
    penalises correct paraphrase. Students are told this explicitly and asked to also
    score 12 answers by hand - comparing the two is the lesson. Automatic evaluation of
    generated text is a genuinely open problem, and pretending otherwise would be the
    worst thing this course could teach.
    """
    answer_words = _words(answer)
    if not answer_words:
        return 0.0
    return len(answer_words & _words(context)) / len(answer_words)
