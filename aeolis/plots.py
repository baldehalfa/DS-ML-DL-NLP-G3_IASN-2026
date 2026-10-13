"""Every figure the notebooks need, one call each.

Rationale: plotting is pure syntax cost with zero AI content, and it is where beginner
programmers lose the most lab time. No student exercise in this course asks anyone to write
plotting code. They call one of these, then spend their time reading the figure - which is
the skill that actually transfers.

Because students only ever *read* these figures, each one is responsible for being
readable on its own: axes labelled with units, the important value annotated, and the
comparison the session cares about drawn in rather than left to be inferred.

    from aeolis import plots
    plots.missingness(df)
    plots.threshold_sweep(y_test, probabilities, cost_fp=500, cost_fn=20000)

Every function draws on the current axes (or makes a figure if there is none) and returns
the Axes, so a notebook can add to it afterwards.

All 25 figures are implemented and every one is exercised by a notebook that executes in the
verification gate, `embedding_space`, `attention_map` and `retrieval_report` included - they
draw real embeddings and real attention weights, not illustrations of them.

The last five differ in kind from the rest. `kmeans_steps`, `dbscan_mechanism`,
`pca_rotation`, `computation_graph` and `split_layout` show a *mechanism* rather than a
result: what the algorithm did, not what it produced. Each one runs the algorithm and plots
the outcome, so the picture cannot drift away from the behaviour it claims to show - which is
the reason none of them is a drawing.
"""

from __future__ import annotations

import numpy as np

# One small palette, used consistently, so a colour means the same thing in every session.
BLUE = "#3b6ea5"    # the normal case, the training signal, the actual values
RED = "#c0392b"     # the failure case, the alarm, the thing that costs money
ORANGE = "#e08214"  # the comparison: median vs mean, validation vs training, the forecast
GREY = "#8c8c8c"    # context, reference lines, anything deliberately in the background


def _ax(figsize=(9, 3.2)):
    """Current axes, or a new figure sized for a notebook column."""
    import matplotlib.pyplot as plt

    if not plt.get_fignums():
        plt.figure(figsize=figsize)
    return plt.gca()


def _human(value: float) -> str:
    """Numbers a student can read aloud. `24,900`, not `2.49e+04`.

    Scientific notation on a cost figure is a small cruelty: the reader has to decode the
    exponent before they can compare two cells, which is exactly the comparison the figure
    exists to make.
    """
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.1f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _money_axis(ax) -> None:
    """Absolute values with thousands separators, never a `1e6` offset in the corner."""
    from matplotlib.ticker import FuncFormatter

    # A FuncFormatter already suppresses the `1e6` offset label; calling
    # ticklabel_format(style="plain") on top of it raises, because that method only
    # works with a ScalarFormatter.
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


# ============================================================== Session 1: look at data

def missingness(df, max_cols: int = 25):
    """Where the data is absent, and whether the gaps are structured.

    Two panels, because the two questions are different: *how much* is missing per column,
    and *when* was it missing. A column that is 3% missing at random is a nuisance; a
    column that is 3% missing in one continuous week is a sensor that was offline, which is
    a different problem with a different fix.
    """
    import matplotlib.pyplot as plt

    frac = df.isna().mean().sort_values(ascending=False)[:max_cols]
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(13, max(3.0, 0.28 * len(frac))), width_ratios=[1, 1.4]
    )

    colours = [RED if f > 0.05 else BLUE if f > 0 else GREY for f in frac.values]
    left.barh(range(len(frac)), frac.values * 100, color=colours)
    left.set_yticks(range(len(frac)))
    left.set_yticklabels(frac.index, fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("missing (%)")
    left.set_title("How much is missing, per column")
    if frac.max() == 0:
        left.text(0.5, 0.5, "nothing missing", transform=left.transAxes,
                  ha="center", va="center", color=GREY)

    present = df[frac.index].notna().to_numpy().T.astype(float)
    right.imshow(present, aspect="auto", cmap="Greys_r", interpolation="nearest",
                 vmin=0, vmax=1)
    right.set_yticks(range(len(frac)))
    right.set_yticklabels(frac.index, fontsize=8)
    right.set_xlabel("row order (time, if the data is sorted)")
    right.set_title("When it was missing (black = absent)")
    fig.tight_layout()
    return left, right


def distributions(df, columns=None, bins: int = 50):
    """Histograms with mean and median marked - the Stat Shot 1 figure.

    The mean-vs-median gap is drawn in rather than described, because that gap is the whole
    lesson: when they separate, a tail is pulling the average away from the typical value,
    and anyone quoting the average is misleading their reader.
    """
    import matplotlib.pyplot as plt

    columns = list(columns or df.select_dtypes("number").columns[:6])
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 2.7 * rows), squeeze=False)

    for ax, column in zip(axes.ravel(), columns):
        values = df[column].dropna()
        ax.hist(values, bins=bins, color=BLUE, alpha=0.85)
        mean, median = values.mean(), values.median()
        ax.axvline(mean, color=RED, lw=2, label=f"mean {mean:,.1f}")
        ax.axvline(median, color=ORANGE, lw=2, ls="--", label=f"median {median:,.1f}")
        spread = values.std() or 1.0
        gap = abs(mean - median) / spread
        ax.set_title(f"{column}" + ("   <- skewed" if gap > 0.15 else ""), fontsize=9)
        ax.set_ylabel("count", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for ax in axes.ravel()[len(columns):]:
        ax.axis("off")
    fig.tight_layout()
    return axes


def correlation_heatmap(df, columns=None):
    """Correlation structure, with the causation warning attached to the figure itself.

    The warning is in the caption on purpose. Students read a heatmap as a diagram of
    mechanism, and the strongest correlation in an industrial dataset is quite often an
    artefact of how the data was collected.
    """
    import matplotlib.pyplot as plt

    numeric = df[list(columns)] if columns is not None else df.select_dtypes("number")
    corr = numeric.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=(0.75 * n + 3, 0.68 * n + 2.4))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.columns, fontsize=8)

    for i in range(n):
        for j in range(n):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else "black")

    fig.colorbar(image, ax=ax, shrink=0.8, label="correlation")
    ax.set_title("Correlation between measurements\n"
                 "Strong here means 'they move together', NOT 'one causes the other'",
                 fontsize=9)
    fig.tight_layout()
    return ax


# ========================================================= Session 2: models and costs

def coefficient_path(alphas, coefficients, feature_names):
    """THE regularisation figure: coefficients shrinking as the penalty grows.

    This plot replaces the Ridge and Lasso formulas entirely. Students predict the shape
    before it is drawn, then explain what Lasso did that Ridge did not - which is visible
    here as lines reaching exactly zero and staying there.
    """
    ax = _ax((9, 4))
    coefficients = np.asarray(coefficients)
    for column, name in enumerate(feature_names):
        ax.plot(alphas, coefficients[:, column], lw=1.8, label=name)

    ax.set_xscale("log")
    ax.axhline(0, color=GREY, lw=1, ls=":")
    ax.set_xlabel("regularisation strength (alpha) - stronger to the right")
    ax.set_ylabel("coefficient value")
    ax.set_title("What regularisation does: every coefficient is pulled towards zero")
    ax.legend(fontsize=7, ncol=2)

    zeroed = int((np.abs(coefficients[-1]) < 1e-8).sum())
    if zeroed:
        ax.text(0.98, 0.04, f"{zeroed} coefficient(s) driven to exactly zero",
                transform=ax.transAxes, ha="right", fontsize=8, color=RED)
    return ax


def learning_curves(train_scores, val_scores, x_values, x_label="training set size",
                    metric="score"):
    """The over/underfitting diagnosis picture, reused in Sessions 2 and 3.

    The gap between the two lines is shaded and measured, because the gap - not either
    line on its own - is the diagnosis.
    """
    ax = _ax((8, 3.6))
    train_scores = np.asarray(train_scores, dtype=float)
    val_scores = np.asarray(val_scores, dtype=float)

    ax.plot(x_values, train_scores, "o-", color=BLUE, lw=2, ms=4, label="training")
    ax.plot(x_values, val_scores, "s--", color=ORANGE, lw=2, ms=4, label="validation")
    ax.fill_between(x_values, val_scores, train_scores, color=GREY, alpha=0.18)

    gap = float(train_scores[-1] - val_scores[-1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric)
    verdict = ("large gap: memorising the training data" if gap > 0.1
               else "both low: too simple, or the features do not carry the signal"
               if max(train_scores[-1], val_scores[-1]) < 0.6
               else "small gap: generalising")
    ax.set_title(f"Final gap {gap:+.3f} - {verdict}")
    ax.legend(fontsize=8)
    return ax


def confusion(y_true, y_prob, threshold: float = 0.5, cost_fp=None, cost_fn=None):
    """Confusion matrix at a chosen threshold, optionally priced.

    Priced, because a 2x2 table of counts invites students to compare the two error cells
    as if they were comparable. They are not: on the AEOLIS maintenance problem one costs a
    technician visit and the other costs a gearbox.
    """
    from .evaluate import confusion_counts

    ax = _ax((5.2, 4.2))
    counts = confusion_counts(y_true, y_prob, threshold)
    grid = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])

    ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() or 1)
    labels = [["true negative", "FALSE ALARM"], ["MISSED FAILURE", "true positive"]]
    for i in range(2):
        for j in range(2):
            text = f"{labels[i][j]}\n{grid[i, j]:,}"
            if cost_fp is not None and (i, j) in {(0, 1), (1, 0)}:
                unit = cost_fp if (i, j) == (0, 1) else cost_fn
                text += f"\ncost {grid[i, j] * unit:,.0f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > grid.max() * 0.55 else "black")

    ax.set_xticks([0, 1], ["predicted: fine", "predicted: failure"], fontsize=8)
    ax.set_yticks([0, 1], ["actually fine", "actually failed"], fontsize=8)
    title = f"Decisions at threshold {threshold:.2f}"
    if cost_fp is not None and cost_fn is not None:
        total = counts["fp"] * cost_fp + counts["fn"] * cost_fn
        title += f"\ntotal cost of being wrong: {total:,.0f}"
    ax.set_title(title, fontsize=9)
    return ax


def threshold_sweep(y_true, y_prob, cost_fp: float, cost_fn: float, max_alarms=None):
    """Cost against threshold, with the default and the optimum both marked.

    The distance between those two vertical lines is the Session 2 lesson, and it is a
    distance no leaderboard reports: same model, same data, one number changed. The measured
    figures are in docs/verified-teaching-results.md and deliberately not repeated
    here - this module ships to students, and the session asks them to guess before the sweep
    is plotted.

    Pass `max_alarms` to add the operational constraint: the cost-optimal threshold fires
    alarms faster than a maintenance team can answer them.
    """
    from .evaluate import alarm_rate, best_threshold, best_threshold_under_capacity, expected_cost

    ax = _ax((9, 3.8))
    thresholds = np.linspace(0.0, 1.0, 201)
    costs = [expected_cost(y_true, y_prob, t, cost_fp, cost_fn) for t in thresholds]
    ax.plot(thresholds, costs, color=BLUE, lw=2)

    default = expected_cost(y_true, y_prob, 0.5, cost_fp, cost_fn)
    ax.axvline(0.5, color=GREY, lw=1.5, ls=":")
    ax.annotate(f"the default, 0.50\ncosts {default:,.0f}", (0.5, default), fontsize=8,
                color="black", xytext=(16, 34), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GREY, alpha=0.9))

    t_best, c_best = best_threshold(y_true, y_prob, cost_fp, cost_fn, grid=201)
    ax.axvline(t_best, color=RED, lw=1.5)
    ax.plot([t_best], [c_best], "o", color=RED, ms=7)
    ax.annotate(f"the cheapest, {t_best:.2f}\ncosts {c_best:,.0f}", (t_best, c_best),
                fontsize=8, color=RED, xytext=(30, 30), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, alpha=0.9))
    _money_axis(ax)

    if max_alarms is not None:
        t_cap, c_cap = best_threshold_under_capacity(
            y_true, y_prob, cost_fp, cost_fn, max_alarms=max_alarms
        )
        ax.axvline(t_cap, color=ORANGE, lw=1.5, ls="--")
        ax.annotate(f"best within {max_alarms} alarms\n{t_cap:.2f} -> {c_cap:,.0f}",
                    (t_cap, c_cap), fontsize=8, color=ORANGE,
                    xytext=(10, -28), textcoords="offset points")

    rate = alarm_rate(y_true, y_prob, t_best)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("total cost of being wrong")
    ax.set_title(
        f"The threshold is a business decision, not a default\n"
        f"at the cheapest threshold: {rate['alarms']} alarms, "
        f"1 every {rate['periods_between_alarms']:.0f} rows, {rate['missed']} still missed",
        fontsize=9,
    )
    return ax


def model_comparison(rows, highlight: str | None = None):
    """Cost-of-ownership table as a figure - the Session 3 conclusion.

    `rows` is a list of dicts with a 'name' key plus any numeric columns. Each column is
    scaled independently and the best value per column is marked, so students can see at a
    glance that "better" points at different models depending on which row you care about.
    That disagreement is the finding, not a defect.
    """
    import matplotlib.pyplot as plt

    names = [r["name"] for r in rows]
    metrics = [k for k in rows[0] if k != "name"]
    fig, ax = plt.subplots(figsize=(1.9 * len(metrics) + 3, 0.7 * len(names) + 2.2))

    lower_is_better = ("cost", "time", "size", "error", "mae", "rmse", "s", "kb")
    for column, metric in enumerate(metrics):
        values = np.array([float(r[metric]) for r in rows])
        smaller_better = any(k in metric.lower() for k in lower_is_better)
        best = values.min() if smaller_better else values.max()
        span = np.ptp(values) or 1.0  # np.ptp, not values.ptp(): the method went in NumPy 2.0
        shade = (values - values.min()) / span
        if not smaller_better:
            shade = 1 - shade
        for row in range(len(names)):
            ax.add_patch(plt.Rectangle((column, row), 1, 1, color=BLUE,
                                       alpha=0.10 + 0.35 * (1 - shade[row])))
            is_best = np.isclose(values[row], best)
            ax.text(column + 0.5, row + 0.5,
                    _human(values[row]) + ("  *" if is_best else ""),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if is_best else "normal")

    ax.set_xticks(np.arange(len(metrics)) + 0.5)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.set_yticks(np.arange(len(names)) + 0.5)
    ax.set_yticklabels(
        [f"{'-> ' if n == highlight else ''}{n}" for n in names], fontsize=9
    )
    ax.set_xlim(0, len(metrics))
    ax.set_ylim(0, len(names))
    ax.invert_yaxis()
    ax.set_title("Cost of ownership: * marks the best in each column", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    return ax


# ================================================= Session 3: what training looks like

def loss_surface_1d(weights, losses, path=None):
    """The one-parameter loss curve with the gradient-descent path drawn on it.

    Session 3's whole conceptual load is carried by this picture: the loss is a landscape,
    the gradient is its slope, and training is walking downhill. One parameter, so the
    landscape is a curve a student can see all of.
    """
    ax = _ax((8, 3.8))
    ax.plot(weights, losses, color=BLUE, lw=2, label="loss for each possible weight")

    if path is not None:
        path = np.asarray(path, dtype=float)
        ax.plot(path[:, 0], path[:, 1], "o-", color=RED, ms=5, lw=1.2,
                label=f"gradient descent, {len(path)} steps")
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=RED, lw=1.1, alpha=0.75))
        ax.plot([path[-1, 0]], [path[-1, 1]], "*", color=RED, ms=16)
        ax.annotate(f"stopped at w={path[-1, 0]:.2f}", path[-1], fontsize=8, color=RED,
                    xytext=(10, 12), textcoords="offset points")

    best = int(np.argmin(losses))
    ax.axvline(weights[best], color=GREY, ls=":", lw=1.2)
    ax.annotate(f"lowest loss: w={weights[best]:.2f}", (weights[best], np.max(losses)),
                fontsize=8, color=GREY, xytext=(8, -12), textcoords="offset points")
    ax.set_xlabel("the single weight being learned")
    ax.set_ylabel("loss (how wrong the model is)")
    ax.set_title("Training is walking downhill on this curve")
    ax.legend(fontsize=8)
    return ax


# ============================================================ Session 4: temporal data

def decomposition(series, period: int):
    """Observed / trend / seasonal / remainder, stacked and aligned.

    Four panels sharing an x axis so students can read a single date downwards through all
    four and see the sum. The remainder panel is the interesting one: structure left in it
    is signal the decomposition did not capture.
    """
    import matplotlib.pyplot as plt
    from statsmodels.tsa.seasonal import STL

    values = series.interpolate().dropna()
    result = STL(values, period=period, robust=True).fit()

    fig, axes = plt.subplots(4, 1, figsize=(12, 7.5), sharex=True)
    for ax, (data, label, colour) in zip(
        axes,
        [
            (values, "observed", BLUE),
            (result.trend, "trend", RED),
            (result.seasonal, f"seasonal (period {period})", ORANGE),
            (result.resid, "remainder", GREY),
        ],
    ):
        ax.plot(data.index, np.asarray(data), lw=0.8, color=colour)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)

    strength = 1 - result.resid.var() / (result.seasonal + result.resid).var()
    # No verdict from this number alone. Measured on the AEOLIS consumption series, a
    # "seasonal strength" of 0.39 sits with a seasonal-naive baseline that beats plain
    # naive by 1.6x - so seasonality was clearly where the signal was, while the index
    # called it weak. The strength says how much of the wiggle the seasonal component
    # accounts for; only a baseline comparison says whether that is useful to you.
    axes[0].set_title(
        f"Decomposition - the seasonal component accounts for {strength:.0%} of the "
        f"variation left after the trend\n"
        f"Whether that is useful is decided by the baseline comparison, not by this number",
        fontsize=9,
    )
    axes[-1].set_xlabel("date")
    fig.tight_layout()
    return axes


def backtest(actual, folds, label="forecast"):
    """Every walk-forward fold drawn over the actuals, with the training windows shaded.

    The shading is the point. A student who can see that each forecast only ever used data
    to its left has understood temporal validation, and will not accidentally shuffle a
    time series again.

    `folds` is a list of (index, prediction) pairs, one per fold.
    """
    ax = _ax((12, 4))
    ax.plot(getattr(actual, "index", np.arange(len(actual))), np.asarray(actual),
            color=GREY, lw=1.0, label="actual")

    for number, (index, prediction) in enumerate(folds):
        ax.plot(index, np.asarray(prediction), lw=2, color=ORANGE if number % 2 else RED,
                label=label if number == 0 else None)
        ax.axvspan(index[0], index[-1], color=BLUE, alpha=0.06)
        ax.annotate(f"fold {number + 1}", (index[0], ax.get_ylim()[1]), fontsize=7,
                    color=GREY, va="top")

    ax.set_xlabel("date - every forecast used only data to its left")
    ax.set_ylabel("value")
    ax.set_title(f"Walk-forward backtest: {len(folds)} folds, each trained on its own past")
    ax.legend(fontsize=8)
    return ax


def forecast_vs_baseline(dates, actual, predictions: dict):
    """One window, the actuals, and every candidate forecast on the same axes.

    Including the naive baseline, always, and in the legend with its error - because the
    comparison a student needs is not "is my model good" but "is my model better than
    doing almost nothing".
    """
    from .evaluate import forecast_errors

    ax = _ax((12, 3.8))
    ax.plot(dates, np.asarray(actual), color=GREY, lw=2.2, label="actual")
    for number, (name, values) in enumerate(predictions.items()):
        errors = forecast_errors(actual, values)
        ax.plot(dates, np.asarray(values), lw=1.6, ls="--" if number else "-",
                color=[RED, ORANGE, BLUE][number % 3],
                label=f"{name} (MAE {errors['mae']:.2f})")
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.set_title("Every forecast is judged against the baseline, not against zero")
    ax.legend(fontsize=8)
    return ax


# ==================================================== Sessions 5 and 6: still contracts

def image_examples(samples, columns: int = 8, title="Inspection images"):
    """A grid of example images, labelled, so a student sees a defect rather than reading about one.

    `samples` is a list of (path, defective) pairs from `aeolis.data.load_inspection_samples`.
    Defective examples are outlined in red. Worth looking at for thirty seconds before any
    modelling: on these parts the defect is a small local disturbance, often a few pixels
    wide, which is exactly why the gradient features carry the signal and the mean intensity
    does not.
    """
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    samples = list(samples)
    rows = int(np.ceil(len(samples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(1.3 * columns, 3.1 * rows))
    for ax, (path, defective) in zip(np.ravel(axes), samples):
        ax.imshow(mpimg.imread(str(path)), cmap="gray")
        for spine in ax.spines.values():
            spine.set_edgecolor(RED if defective else GREY)
            spine.set_linewidth(2.5 if defective else 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("defect" if defective else "ok", fontsize=7,
                      color=RED if defective else GREY)
    for ax in np.ravel(axes)[len(samples):]:
        ax.axis("off")
    fig.suptitle(f"{title} - red outline = defective", fontsize=9)
    fig.tight_layout()
    return axes


def embedding_space(vectors, labels, annotate=None, title="Embedding space, projected to 2-D"):
    """A 2-D projection of an embedding space, coloured by class.

    The projection is PCA, not t-SNE, on purpose: PCA is a fixed linear map, so distances
    on the page mean something consistent and a student cannot be fooled by the layout
    changing between runs. t-SNE makes prettier clusters and invites exactly that mistake.

    The caption states the share of variance retained, because a 2-D view of 384 dimensions
    throws away most of what the model actually uses - and a figure that looks like clean
    clusters can still sit on top of a classifier that fails.
    """
    from sklearn.decomposition import PCA

    ax = _ax((8.5, 6))
    vectors = np.asarray(vectors, dtype=float)
    projector = PCA(n_components=2, random_state=42)
    flat = projector.fit_transform(vectors)
    retained = projector.explained_variance_ratio_.sum()

    palette = [BLUE, RED, ORANGE, GREY, "#5b8c5a", "#7d5ba6"]
    for number, label in enumerate(sorted(set(labels))):
        mask = np.array([lab == label for lab in labels])
        ax.scatter(flat[mask, 0], flat[mask, 1], s=14, alpha=0.55,
                   color=palette[number % len(palette)], label=str(label))

    if annotate:
        for index, text in annotate:
            ax.annotate(text[:38], flat[index], fontsize=7, alpha=0.9,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("first principal component")
    ax.set_ylabel("second principal component")
    ax.set_title(f"{title}\n"
                 f"these two axes carry {retained:.0%} of the variation - "
                 f"the classifier uses all {vectors.shape[1]} dimensions",
                 fontsize=9)
    ax.legend(fontsize=8, markerscale=1.6)
    return ax


def attention_map(tokens, weights, focus=None):
    """Which tokens attend to which - the attention intuition figure.

    Pass real attention weights, not an illustration. Measured on the course encoder, the
    word "noise" in "gearbox making grinding noise at high load" attends most strongly to
    "grinding" and "gearbox" - which is the point, and it is more convincing than any
    hand-drawn diagram because a student can change the sentence and watch it move.
    """
    import matplotlib.pyplot as plt

    weights = np.asarray(weights, dtype=float)
    fig, ax = plt.subplots(figsize=(0.55 * len(tokens) + 3, 0.5 * len(tokens) + 2.5))
    image = ax.imshow(weights, cmap="Blues", vmin=0)

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("attending TO this token")
    ax.set_ylabel("this token is looking...")

    subtitle = "each row shows where one word looks for its context"
    if focus is not None and focus in tokens:
        row = tokens.index(focus)
        ax.add_patch(plt.Rectangle((-0.5, row - 0.5), len(tokens), 1,
                                   fill=False, edgecolor=RED, lw=2))
        ranked = np.argsort(-weights[row])
        best = [tokens[j] for j in ranked if tokens[j] not in ("[CLS]", "[SEP]", focus)][:3]
        subtitle = f"'{focus}' draws its meaning mostly from: {', '.join(best)}"

    ax.set_title(f"Self-attention (last layer, averaged over heads)\n{subtitle}", fontsize=9)
    fig.colorbar(image, ax=ax, shrink=0.7, label="attention weight")
    fig.tight_layout()
    return ax


def retrieval_report(results, k: int = 3):
    """Per-question retrieval hit and groundedness, with the unanswerable questions flagged.

    `results` is a list of dicts with keys: question, answerable, hit (bool or None),
    grounded (float), ok (bool).

    The unanswerable questions are drawn in a separate band on purpose. For those, a *hit*
    is meaningless and a high groundedness score is actively misleading: measured, a flat
    refusal scores well on the lexical proxy despite containing nothing from the context
    (docs/verified-teaching-results.md has the figure). Showing them apart stops that
    number being read as if it meant the same thing.
    """
    import matplotlib.pyplot as plt

    answerable = [r for r in results if r.get("answerable")]
    unanswerable = [r for r in results if not r.get("answerable")]
    rows = answerable + unanswerable

    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 2.4))
    positions = np.arange(len(rows))

    for y, row in zip(positions, rows):
        ax.barh(y, row.get("grounded", 0.0), height=0.6,
                color=BLUE if row.get("answerable") else GREY, alpha=0.55)
        if row.get("answerable"):
            marker, colour = ("hit" if row.get("hit") else "MISS"), (
                BLUE if row.get("hit") else RED)
        else:
            marker, colour = "n/a", GREY
        ax.text(-0.02, y, marker, ha="right", va="center", fontsize=7.5, color=colour)
        verdict = "correct" if row.get("ok") else "WRONG"
        ax.text(1.02, y, verdict, ha="left", va="center", fontsize=7.5,
                color=BLUE if row.get("ok") else RED)

    ax.set_yticks(positions)
    ax.set_yticklabels([r["question"][:56] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("groundedness (lexical overlap with the retrieved context) - a PROXY")

    if answerable and unanswerable:
        ax.axhline(len(answerable) - 0.5, color="black", lw=1.2)
        ax.text(0.5, len(answerable) - 0.42, "below: no document answers these",
                ha="center", va="top", fontsize=8, style="italic")

    hits = sum(1 for r in answerable if r.get("hit"))
    right = sum(1 for r in rows if r.get("ok"))
    ax.set_title(
        f"hit@{k} {hits}/{len(answerable)} on answerable questions - "
        f"{right}/{len(rows)} answers acceptable\n"
        f"left label = did retrieval find the source; right label = was the answer acceptable",
        fontsize=9,
    )
    fig.tight_layout()
    return ax


# ============================================ Session 7: structure without a label

def cluster_search(ks, inertias, silhouettes):
    """The two diagnostics people use to choose k, side by side so they can disagree.

    They usually do. The elbow is a judgement call about where a smooth curve stops
    bending, and on real data it is rarely a corner; the silhouette is an actual number
    and has a maximum. Drawing them together stops a student reading a confident answer
    out of the elbow that the elbow did not give.

    The silhouette panel is annotated with its own best k, and the scale is left honest:
    a peak at 0.26 is a weak peak, and the figure should not hide that by zooming in.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

    axes[0].plot(ks, inertias, marker="o", color=BLUE)
    axes[0].set_xlabel("number of clusters, k")
    axes[0].set_ylabel("inertia")
    axes[0].set_title("Elbow: where does it stop bending?", fontsize=9)
    _money_axis(axes[0])

    best = int(ks[int(np.argmax(silhouettes))])
    axes[1].plot(ks, silhouettes, marker="o", color=ORANGE)
    axes[1].axvline(best, color=RED, lw=1.1, ls="--")
    axes[1].annotate(f"best k = {best}\n{max(silhouettes):.3f}",
                     (best, max(silhouettes)), textcoords="offset points",
                     xytext=(8, -4), fontsize=8, color=RED)
    axes[1].set_xlabel("number of clusters, k")
    axes[1].set_ylabel("mean silhouette")
    axes[1].set_ylim(0, max(0.5, max(silhouettes) * 1.25))
    axes[1].set_title("Silhouette: how separated are they, really?", fontsize=9)

    fig.tight_layout()
    return axes


def clusters_2d(vectors, found, truth=None, titles=("What the clustering found",
                                                    "What was actually there")):
    """Cluster assignments beside the truth, on the same 2-D projection.

    Two panels on one projection, because the question students need to answer is not
    "are there clusters" - k-means always returns k of them - but "are these the groups
    I care about". Side by side, a clustering that has found real structure and one that
    has sliced a single blob into k arbitrary pieces look completely different.

    PCA, not t-SNE: a fixed linear map, so the two panels are directly comparable and the
    distances mean the same thing in both.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    flat = np.asarray(vectors, dtype=float)
    if flat.shape[1] > 2:
        flat = PCA(n_components=2, random_state=42).fit_transform(flat)

    panels = 1 if truth is None else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.4 * panels, 3.8), squeeze=False)

    for ax, labels, title in zip(axes[0], [found, truth][:panels], titles):
        labels = np.asarray(labels)
        for value in sorted(set(labels.tolist())):
            mask = labels == value
            noise = value == -1
            ax.scatter(flat[mask, 0], flat[mask, 1], s=12, alpha=0.6,
                       color=GREY if noise else None,
                       label=f"noise ({mask.sum()})" if noise else f"{value} ({mask.sum()})")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.legend(fontsize=7, frameon=False, loc="best")

    fig.tight_layout()
    return axes[0]


def scree(explained_variance_ratio, targets=(0.90,)):
    """Per-component variance and the running total, with the targets marked.

    The running total is the one that answers the engineering question - how many columns
    can I throw away - so it gets the annotation. The per-component bars are there to show
    why the answer is never "one": the first component is large and the rest decay slowly,
    which is what a real feature table looks like.
    """
    import matplotlib.pyplot as plt

    ratio = np.asarray(explained_variance_ratio, dtype=float)
    cumulative = np.cumsum(ratio)
    x = np.arange(1, len(ratio) + 1)

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x, ratio, color=BLUE, alpha=0.55, label="each component")
    ax.set_xlabel("component")
    ax.set_ylabel("share of variance")

    twin = ax.twinx()
    twin.plot(x, cumulative, color=ORANGE, lw=1.6, label="running total")
    twin.set_ylabel("cumulative share")
    twin.set_ylim(0, 1.02)

    for target in targets:
        n = int(np.searchsorted(cumulative, target) + 1)
        twin.axhline(target, color=GREY, lw=0.9, ls=":")
        twin.annotate(f"{target:.0%} needs {n} of {len(ratio)}",
                      (n, target), textcoords="offset points", xytext=(6, -12),
                      fontsize=8, color=RED)
        twin.plot([n], [cumulative[n - 1]], marker="o", color=RED, ms=5)

    ax.set_title(f"{len(ratio)} columns in; the running total says how many you actually need",
                 fontsize=9)
    fig.tight_layout()
    return ax


def inspection_budget(scores, y_true, budgets=None, base_rate=None):
    """What a ranking is worth to someone who can only inspect so many items.

    An AUC does not tell a maintenance planner anything. "Inspect the worst 200 and you
    will find 43 of the 339 failures, at 21% rather than 3%" does. This is the Session 2
    alarm-capacity argument again, arriving from the unsupervised side: the ranking is
    only as useful as the number of items somebody can actually look at.
    """
    import matplotlib.pyplot as plt

    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(scores)[::-1]
    base = float(y_true.mean()) if base_rate is None else base_rate
    budgets = list(budgets) if budgets is not None else [50, 100, 200, 400, 800]

    precision = [y_true[order[:b]].mean() for b in budgets]
    caught = [int(y_true[order[:b]].sum()) for b in budgets]

    fig, ax = plt.subplots(figsize=(9, 3.4))
    bars = ax.bar([str(b) for b in budgets], precision, color=BLUE, alpha=0.8)
    ax.axhline(base, color=RED, lw=1.2, ls="--")
    ax.annotate(f"picking at random: {base:.1%}", (len(budgets) - 0.5, base),
                textcoords="offset points", xytext=(-4, 6), ha="right",
                fontsize=8, color=RED)

    for bar, p, c in zip(bars, precision, caught):
        ax.text(bar.get_x() + bar.get_width() / 2, p, f"{p:.1%}\n{c} found",
                ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("how many items you inspect")
    ax.set_ylabel("share that are real failures")
    ax.set_ylim(0, max(precision) * 1.35)
    ax.set_title(f"{int(y_true.sum())} real failures among {len(y_true):,} machines - "
                 "what the ranking buys you", fontsize=9)
    fig.tight_layout()
    return ax


# ------------------------------------------------- Session 7: how the algorithms work
#
# The four functions above draw *results*. These three draw *mechanisms*: what the
# algorithm is actually doing, step by step. Every other figure in this course is computed
# from real data rather than illustrated, and so are these - they run the algorithm and
# plot what it did, so the picture cannot drift away from the behaviour it claims to show.

def kmeans_steps(vectors, k: int = 3, steps: int = 3, seed: int = 42):
    """Lloyd's algorithm, one panel per half-step, until it stops moving.

    k-means is four lines of pseudocode and almost everybody nods along without ever
    watching it run. Assign, recompute, assign, recompute: the panels make it obvious that
    the centres are just means, that the boundaries are always straight, and that the whole
    thing has converged long before the student expects it to.

    Run here rather than drawn from memory, so what the figure shows is what the algorithm
    did on this data.
    """
    import matplotlib.pyplot as plt

    X = np.asarray(vectors, dtype=float)[:, :2]
    rng = np.random.default_rng(seed)
    centres = X[rng.choice(len(X), size=k, replace=False)].copy()

    frames = [("Initialisation: k centres, placed at random", centres.copy(), None)]
    labels = None
    for step in range(steps):
        labels = np.argmin(((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1), axis=1)
        frames.append((f"Assign {step + 1}: every point to its nearest centre",
                       centres.copy(), labels.copy()))
        moved = np.array([X[labels == j].mean(axis=0) if (labels == j).any() else centres[j]
                          for j in range(k)])
        shift = float(np.abs(moved - centres).max())
        centres = moved
        frames.append((f"Recompute {step + 1}: each centre moves to its mean "
                       f"(largest move {shift:.2f})", centres.copy(), labels.copy()))

    cols = 3
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows), squeeze=False)
    palette = [BLUE, ORANGE, RED, GREY]

    for ax, (title, cen, lab) in zip(axes.ravel(), frames):
        if lab is None:
            ax.scatter(X[:, 0], X[:, 1], s=12, color=GREY, alpha=0.45)
        else:
            for j in range(k):
                ax.scatter(X[lab == j, 0], X[lab == j, 1], s=12, alpha=0.55,
                           color=palette[j % len(palette)])
        ax.scatter(cen[:, 0], cen[:, 1], marker="X", s=150, color="black",
                   edgecolor="white", linewidth=1.2, zorder=5)
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")

    fig.suptitle("k-means: assign, recompute, repeat. The crosses are the centres.",
                 fontsize=9.5, y=1.0)
    fig.tight_layout()
    return axes


def dbscan_mechanism(vectors, eps: float, min_samples: int):
    """Core, border and noise, with one neighbourhood drawn to scale.

    DBSCAN's two parameters are usually explained in words and remain abstract. Drawing the
    `eps` circle on an actual point, and colouring the three kinds of point the algorithm
    distinguishes, turns "density-based" into something a student can see: a core point has
    at least `min_samples` neighbours inside that circle, a border point does not but sits
    inside somebody else's, and noise is neither.
    """
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    X = np.asarray(vectors, dtype=float)[:, :2]
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = model.labels_

    core = np.zeros(len(X), dtype=bool)
    core[model.core_sample_indices_] = True
    noise = labels == -1
    border = ~core & ~noise

    ax = _ax((6.4, 5.0))
    ax.scatter(X[core, 0], X[core, 1], s=36, color=BLUE, alpha=0.85,
               label=f"core: {min_samples} or more neighbours within eps ({core.sum()})")
    ax.scatter(X[border, 0], X[border, 1], s=36, facecolor="white", edgecolor=ORANGE,
               linewidth=1.4, label=f"border: inside someone else's circle ({border.sum()})")
    ax.scatter(X[noise, 0], X[noise, 1], s=30, color=GREY, marker="x",
               label=f"noise: neither, labelled -1 ({noise.sum()})")

    if core.any():
        centre = X[core][np.argmax(X[core][:, 0])]
        ax.add_patch(plt.Circle(tuple(centre), eps, fill=False, color=RED,
                                lw=1.4, ls="--"))
        ax.annotate(f"eps = {eps}", centre + np.array([eps * 0.72, eps * 0.72]),
                    fontsize=8.5, color=RED)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_title(f"DBSCAN with eps={eps}, min_samples={min_samples}: density, not distance "
                 "to a centre", fontsize=9)
    return ax


def pca_rotation(vectors):
    """The same points before and after, with the components drawn as arrows.

    PCA is a rotation onto new axes, and saying so rarely lands. Two panels do: the original
    cloud with the two components drawn from the mean and scaled by how much variance each
    carries, then the same cloud expressed in those coordinates. Nothing has been lost yet -
    that only happens when you drop the second axis, which is the next thing the notebook
    does.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    X = np.asarray(vectors, dtype=float)[:, :2]
    pca = PCA(n_components=2, random_state=42).fit(X)
    rotated = pca.transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))

    axes[0].scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.45)
    for length, vector, colour, name in zip(pca.explained_variance_, pca.components_,
                                            (RED, ORANGE), ("component 1", "component 2")):
        tip = pca.mean_ + vector * 2.4 * np.sqrt(length)
        axes[0].annotate("", tip, pca.mean_,
                         arrowprops=dict(arrowstyle="->", color=colour, lw=2.0))
        axes[0].annotate(name, tip, textcoords="offset points", xytext=(6, 4),
                         fontsize=8.5, color=colour)
    axes[0].set_title("The data, with the directions it varies in", fontsize=9)

    axes[1].scatter(rotated[:, 0], rotated[:, 1], s=14, color=BLUE, alpha=0.45)
    axes[1].axhline(0, color=GREY, lw=0.9)
    axes[1].axvline(0, color=GREY, lw=0.9)
    axes[1].set_xlabel(f"component 1  ({pca.explained_variance_ratio_[0]:.0%} of the variance)")
    axes[1].set_ylabel(f"component 2  ({pca.explained_variance_ratio_[1]:.0%})")
    axes[1].set_title("The same points, in those coordinates. A rotation, nothing dropped",
                      fontsize=9)

    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    return axes


# ================================================ Session 3: what backpropagation does

def computation_graph(trace):
    """One forward pass and one backward pass through a tiny network, drawn.

    Autograd is the point at which a session stops being arithmetic a student can follow and
    starts being a method call they trust. This draws the thing `loss.backward()` walks: a
    graph small enough to read, with the forward value on every node and the gradient on
    every weight, so that "the chain rule, run backwards" has a picture attached to it.

    `trace` is built by the notebook from a real `loss.backward()`, not written down here, so
    the numbers on the figure are the numbers autograd produced. Keys: `x`, `w1`, `h`, `w2`,
    `yhat`, `y`, `loss`, `grad_w1`, `grad_w2`.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(trace["x"], dtype=float).ravel()
    h = np.asarray(trace["h"], dtype=float).ravel()
    w1 = np.asarray(trace["w1"], dtype=float).reshape(len(h), len(x))
    w2 = np.asarray(trace["w2"], dtype=float).ravel()
    g1 = np.asarray(trace["grad_w1"], dtype=float).reshape(len(h), len(x))
    g2 = np.asarray(trace["grad_w2"], dtype=float).ravel()
    yhat, loss = float(trace["yhat"]), float(trace["loss"])

    def column(count, spacing=2.0):
        """Node y-positions for one layer, index 0 at the top, centred on the figure."""
        return -(np.arange(count) - (count - 1) / 2) * spacing

    # Generous horizontal spacing: the edge labels, not the nodes, set how wide this
    # needs to be, and four edges converging on two nodes collide at anything tighter.
    CX = (0.0, 2.6, 5.0, 6.8)
    xs, hs, out = column(len(x)), column(len(h)), column(1)

    ax = _ax((11.5, 5.0))

    def edge(x0, y0, x1, y1, weight, gradient):
        """One connection, with its weight above the line and its gradient below."""
        ax.annotate("", (x1 - 0.30, y1), (x0 + 0.30, y0),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1, alpha=0.7))
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        px, py = -dy / length, dx / length          # perpendicular, so steep edges stay legible
        for fraction, text, colour, side in ((0.27, f"{weight:+.2f}", BLUE, 1.0),
                                             (0.60, f"{gradient:+.3f}", RED, -1.0)):
            ax.text(x0 + fraction * dx + side * 0.20 * px,
                    y0 + fraction * dy + side * 0.20 * py,
                    text, fontsize=8, color=colour, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))

    for j, hy in enumerate(hs):
        for i, xy in enumerate(xs):
            edge(CX[0], xy, CX[1], hy, w1[j, i], g1[j, i])
        edge(CX[1], hy, CX[2], out[0], w2[j], g2[j])
    ax.annotate("", (CX[3] - 0.30, out[0]), (CX[2] + 0.30, out[0]),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.1, alpha=0.7))

    nodes = ([(CX[0], y, f"x{i + 1}\n{v:.2f}", "input") for i, (y, v) in enumerate(zip(xs, x))]
             + [(CX[1], y, f"h{j + 1}\n{v:.2f}", "ReLU") for j, (y, v) in enumerate(zip(hs, h))]
             + [(CX[2], out[0], f"y\u0302\n{yhat:.2f}", "prediction"),
                (CX[3], out[0], f"L\n{loss:.3f}", f"loss  (y = {float(trace['y']):.2f})")])
    for nx, ny, label, kind in nodes:
        ax.scatter([nx], [ny], s=2600, zorder=4, color="white",
                   edgecolor=RED if kind.startswith("loss") else BLUE, linewidth=1.8)
        ax.text(nx, ny, label, ha="center", va="center", fontsize=9, zorder=5)
        ax.text(nx, ny - 0.62, kind, ha="center", va="top", fontsize=7.5, color=GREY)

    top = max(hs.max(), xs.max()) + 1.15
    bottom = min(hs.min(), xs.min()) - 1.35
    ax.annotate("", (CX[3] + 0.3, top), (CX[0] - 0.3, top),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
    ax.text((CX[0] + CX[3]) / 2, top + 0.14,
            "forward pass: multiply, add, activate   (blue = the weight)",
            fontsize=8.5, color=BLUE, ha="center")
    ax.annotate("", (CX[0] - 0.3, bottom), (CX[3] + 0.3, bottom),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
    ax.text((CX[0] + CX[3]) / 2, bottom - 0.22,
            "backward pass: the chain rule, one layer at a time   (red = dL/dw for that weight)",
            fontsize=8.5, color=RED, ha="center", va="top")

    ax.set_xlim(CX[0] - 0.9, CX[3] + 0.9)
    ax.set_ylim(bottom - 0.7, top + 0.55)
    ax.axis("off")
    ax.set_title("What loss.backward() walks: every weight gets its own dL/dw", fontsize=9.5)
    return ax


# ============================================ Session 4: where each fold trains and tests

def split_layout(n, splits, shuffled_test=None, seed=0, unit="rows"):
    """A random split and a walk-forward backtest, on the same time axis.

    Students accept "split temporally" as a rule and then cannot say what it looks like. The
    top strip shows a random split at true scale - the test rows are scattered through the
    whole history, so the model is fitted on both sides of every one of them. The rows below
    show the folds the student actually built: training window, then its test window, and
    nothing to the right of it.

    `splits` is the list of `(train_idx, test_idx)` pairs the notebook produced, so the
    figure is a picture of their own backtest and not a generic illustration.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    n = int(n)
    splits = [(np.asarray(tr), np.asarray(te)) for tr, te in splits]
    if shuffled_test is None:
        rng = np.random.default_rng(seed)
        shuffled_test = rng.choice(n, size=int(round(n * 0.2)), replace=False)

    strip = np.zeros(n, dtype=int)
    strip[np.asarray(shuffled_test, dtype=int)] = 1

    fig, axes = plt.subplots(2, 1, figsize=(10, 1.1 + 0.34 * len(splits)),
                             gridspec_kw={"height_ratios": [1, max(2.2, 0.55 * len(splits))]})

    axes[0].imshow(strip[None, :], aspect="auto", interpolation="nearest",
                   cmap=ListedColormap([BLUE, RED]), vmin=0, vmax=1, extent=(0, n, 0, 1))
    axes[0].set_yticks([])
    axes[0].set_xticks([])
    axes[0].set_title(f"A random 80/20 split: the {len(shuffled_test):,} test {unit} sit "
                      f"inside the training period, scattered across the whole history",
                      fontsize=9)

    # A 24-hour test window inside 35,000 rows is a fraction of a pixel wide. Drawing it at
    # a floor keeps it visible; the caption says so rather than letting the reader assume
    # the blocks are to scale.
    floor = max(1.0, n * 0.006)
    for row, (train_idx, test_idx) in enumerate(splits):
        y = len(splits) - row - 1
        axes[1].broken_barh([(0, n)], (y + 0.15, 0.7), facecolors="#ebebeb")
        axes[1].broken_barh([(train_idx.min(), np.ptp(train_idx) + 1)], (y + 0.15, 0.7),
                            facecolors=BLUE)
        axes[1].broken_barh([(test_idx.min(), max(len(test_idx), floor))], (y + 0.15, 0.7),
                            facecolors=RED)
    axes[1].set_ylim(0, len(splits))
    axes[1].set_yticks(np.arange(len(splits)) + 0.5)
    axes[1].set_yticklabels([f"fold {i + 1}" for i in range(len(splits))][::-1], fontsize=7.5)
    axes[1].set_xlim(0, n)
    axes[1].set_xlabel(f"position in the series ({unit}, oldest on the left)")
    axes[1].set_title("Walk-forward: every red block is to the right of its own blue one "
                      "(test windows drawn at a minimum visible width)", fontsize=9)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=BLUE), plt.Rectangle((0, 0), 1, 1, fc=RED),
               plt.Rectangle((0, 0), 1, 1, fc="#ebebeb")]
    fig.legend(handles, ["trained on", "tested on", "not used by this fold"],
               fontsize=8, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    return axes
