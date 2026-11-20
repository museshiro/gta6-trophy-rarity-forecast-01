"""GTA VI trophy earn-rate forecast (gta6-trophy-rarity-forecast-01), Python 3.13 stdlib only.

Reads every <id>.in case in the input directory and writes one <id>.out forecast
per case: five quantiles (5/25/50/75/95%) of each trophy's Exophase earn rate
(percent, 0-100) at the case's target.

Method, per trophy (my own design):
  1. Trend on the logit scale.  For each history read I take the trophy's rate,
     clamp it to [0.01, 99.99] and take its logit, and pair it with
     x = ln(1 + hours since the game's release midnight).  I fit a straight line
     logit(rate) = a + b*x by weighted least squares, weighting recent reads more
     (exponential decay, 12-hour half-life), because trophy earning is fast at
     launch and then settles: the recent slope is what the future will follow.
     With too few reads, or a flat history, the slope is 0 (persistence).
  2. Median.  I read the fitted line at the target's x.  I cap how far the median
     may move from the last read's logit (4 logit units) so a steep early slope
     cannot extrapolate the rate to absurdity over a multi-day horizon.
  3. Spread as a mixture, not a widened Gaussian.  Both the fit's residual
     scatter and a random-walk term that grows with the horizon (0.5 logit
     units per root-day, as the house baseline uses) set a base spread.  On top
     of that, a refresh is a discrete event: with some probability one lands
     before the target and dilutes every rate by about ln(the players
     multiplier it brings).  So the belief is a mixture -- usually the trend,
     sometimes the trend minus that drop -- and the five quantiles are the
     mixture's own quantiles (found by bisection on its CDF).  The 5% and 25%
     quantiles therefore sit much further below the median than the 75% and
     95% sit above it whenever a refresh is plausible, and the spread turns
     symmetric once the page is refreshing regularly and the trend already
     prices it in.  The refresh probability comes from how stale the page
     looks (how long players_tracked has sat unchanged) against the horizon;
     the drop size comes from the jump sizes the history has shown, or a
     default first-refresh jump when it has shown none.
  4. Quantiles.  Each row is the mixture's quantiles at the five levels,
     rounded to 4 decimals and clamped to [0, 100], forced non-decreasing.

The program is deterministic (no randomness) and dependency-free, so two runs
write identical bytes, and it runs in well under the 8-second cap.

Run locally:  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs:  python3 -P -s /bot/main.py  (no network, /bot read-only).
"""

import json
import math
import os
from pathlib import Path

# The grader's -P leaves /bot off sys.path; nothing beside this file is imported,
# so no path fix is needed.  (This file is self-contained on purpose.)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# How fast a read's influence fades when fitting the trend (hours).
HALF_LIFE_HOURS = 12.0
# Random-walk spread, logit units per square-root day (same scale as the baseline).
SIGMA_RW_PER_ROOT_DAY = 0.5
# The furthest the median may move from the last read, in logit units.
MAX_MEDIAN_MOVE = 4.0
# Floor on the fit's residual scatter, so a perfectly flat history still gets a spread.
SIGMA_FIT_FLOOR = 0.05
# Default residual scatter when too few reads exist to estimate one.
SIGMA_FIT_DEFAULT = 0.30
# Mean time between Exophase refreshes when the history shows none (hours).
# Launch night went ~4.5 h unchanged before its first refresh.
MEAN_REFRESH_HOURS = 4.0
# Players multiplier a refresh is assumed to bring when none has been seen.
# Launch night's first refresh took tracked players 638 -> 3868 (about 6x);
# a stale page hides a backlog, so the first refresh is assumed large.
DEFAULT_JUMP = 6.0
# Reference instant for the log-time axis: the game's release midnight (UTC).
RELEASE_MIDNIGHT = "2026-11-19T00:00:00Z"


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit_pct(pct: float) -> float:
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def sigmoid_pct(logit_value: float) -> float:
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, returning an int for whole numbers."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value.is_integer() else value


def parse_time(text: str):
    """Parse an ISO-8601 UTC timestamp (with a trailing Z) into epoch seconds."""
    from datetime import datetime, timezone

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()


def fit_line(xs, ys, ws):
    """Weighted least squares for y = a + b*x. Returns (a, b, residual_sd).

    With fewer than 2 distinct x values, or fewer than 3 points, the slope is
    0 and the intercept is the weighted mean of y.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, SIGMA_FIT_DEFAULT
    sw = sum(ws)
    if sw <= 0:
        return sum(ys) / n, 0.0, SIGMA_FIT_DEFAULT
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if n < 3 or sxx <= 1e-12:
        # No usable slope: persistence, and the scatter around the mean.
        var = sum(w * (y - my) ** 2 for w, y in zip(ws, ys)) / sw
        return my, 0.0, max(SIGMA_FIT_FLOOR, math.sqrt(max(var, 0.0)))
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    # Weighted residual scatter (dof-adjusted when possible).
    dof = max(n - 2, 1)
    var = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    var *= n / dof if n > 2 else 1.0
    return a, b, max(SIGMA_FIT_FLOOR, math.sqrt(max(var, 0.0)))


def refresh_risk(history, horizon_hours):
    """Estimate the dilution risk of an Exophase refresh before the target.

    Returns (prob_refresh, dilution_logit): the chance a refresh lands inside
    the horizon, and the logit drop it would cause if it did (ln of the players
    multiplier it brings).  The page is stale when players_tracked has sat
    unchanged; a stale page over a long horizon is likely to refresh.  When the
    history already shows refreshes, the observed jump sizes set the magnitude
    and a recent change means the page is fresh, so the risk is small.
    """
    if not history:
        return 0.0, 0.0
    # When players_tracked last changed, and the jump sizes seen so far.
    last_players = None
    last_change_hours = None
    jumps = []
    times = [parse_time(h["at"]) for h in history]
    for k, h in enumerate(history):
        p = h.get("players_tracked")
        if p is None:
            continue
        if last_players is not None and p != last_players:
            if p > last_players:
                jumps.append(p / last_players)
            last_change_hours = times[k]
        last_players = p
    ref = parse_time(history[-1]["at"])
    if last_change_hours is None:
        # Never changed in this history: stale since the first read.
        stale_hours = (ref - times[0]) / 3600.0
    else:
        stale_hours = (ref - last_change_hours) / 3600.0
    if jumps:
        jump = sorted(jumps)[len(jumps) // 2]
        mean_interval = MEAN_REFRESH_HOURS
    else:
        jump = DEFAULT_JUMP
        mean_interval = MEAN_REFRESH_HOURS
    # A refresh is more likely the longer the page has sat stale and the longer
    # the horizon; model its arrival as exponential with the mean interval.
    lam = max(stale_hours, 0.0) / mean_interval
    prob = 1.0 - math.exp(-(horizon_hours / mean_interval) * (1.0 + lam))
    prob = min(max(prob, 0.0), 0.95)
    dilution = math.log(max(jump, 1.0))
    return prob, dilution


def phi_cdf(z: float) -> float:
    """Standard normal CDF, via the error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mixture_quantile(median_logit, sigma_base, prob_refresh, dilution, tau):
    """The tau-quantile of a two-component mixture for a trophy's logit rate.

    With probability (1 - p) the rate follows the trend: Normal(median, sigma).
    With probability p a refresh lands before the target and dilutes the rate:
    Normal(median - dilution, sigma).  The refresh is a discrete drop, not a
    Gaussian widening, so the mixture's lower quantiles sit much further below
    the median than a single widened Gaussian would put them -- which is what
    the pinball loss rewards, since being too high on a low quantile costs
    far more than being too low.  Found by bisection on the mixture CDF.
    """
    if sigma_base <= 1e-9:
        sigma_base = 1e-9
    p = min(max(prob_refresh, 0.0), 0.95)
    d = max(dilution, 0.0)
    if p <= 0.0 or d <= 0.0:
        # No refresh risk: a plain Gaussian around the median.
        from statistics import NormalDist

        return median_logit + NormalDist().inv_cdf(tau) * sigma_base
    lo = median_logit - d - 6.0 * sigma_base
    hi = median_logit + 6.0 * sigma_base

    def cdf(x):
        return (1.0 - p) * phi_cdf((x - median_logit) / sigma_base) + p * phi_cdf(
            (x - (median_logit - d)) / sigma_base
        )

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_trophy(series, target_x):
    """Fit the trend for one trophy and return (median_logit, sigma_fit).

    series: list of (x, logit_rate, weight). The median is the fitted line
    read at the target's x, capped in how far it may move from the last read.
    """
    xs = [s[0] for s in series]
    ys = [s[1] for s in series]
    ws = [s[2] for s in series]
    last_logit = ys[-1]
    a, b, sigma_fit = fit_line(xs, ys, ws)
    median_logit = a + b * target_x
    # Cap the move from the last read so a steep early slope cannot run away.
    median_logit = last_logit + max(-MAX_MEDIAN_MOVE, min(MAX_MEDIAN_MOVE, median_logit - last_logit))
    median_logit = max(-9.21, min(9.21, median_logit))
    return median_logit, sigma_fit


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    history = case.get("history") or []
    trophies = case["trophies"]
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    release = parse_time(case.get("game", {}).get("release_date", "2026-11-19") + "T00:00:00Z"
                        if len(str(case.get("game", {}).get("release_date", ""))) == 10
                        else RELEASE_MIDNIGHT)
    target_t = parse_time(case["target"])
    target_x = math.log(1.0 + max(0.0, (target_t - release) / 3600.0))
    checkpoint_t = parse_time(case["at"])

    prob_refresh, dilution = refresh_risk(history, horizon_hours)
    sigma_rw = SIGMA_RW_PER_ROOT_DAY * math.sqrt(max(horizon_days, 0.0))

    rows = []
    for i, trophy in enumerate(trophies):
        series = []
        for h in history:
            rates = h.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            t = parse_time(h["at"])
            x = math.log(1.0 + max(0.0, (t - release) / 3600.0))
            y = logit_pct(rates[i])
            age_h = max(0.0, (checkpoint_t - t) / 3600.0)
            w = 0.5 ** (age_h / HALF_LIFE_HOURS)
            series.append((x, y, w))
        if not series:
            # No read ever listed this trophy: a neutral low rate, wide spread.
            median_logit, sigma_fit = logit_pct(0.5), SIGMA_FIT_DEFAULT
        else:
            median_logit, sigma_fit = forecast_trophy(series, target_x)
        sigma_base = math.sqrt(sigma_fit ** 2 + sigma_rw ** 2)
        qs = []
        for tau in LEVELS:
            lq = mixture_quantile(median_logit, sigma_base, prob_refresh, dilution, tau)
            qs.append(round4(min(100.0, max(0.0, sigmoid_pct(lq)))))
        # Force non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
