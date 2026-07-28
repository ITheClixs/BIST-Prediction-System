"""Figures for the executed portfolio and its cost sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bist_predict.figures.artifacts import RunArtifacts
from bist_predict.figures.style import COLOURS, caption, figure, save_figure

__all__ = ["plot_equity_curve", "plot_cost_sensitivity"]


def plot_equity_curve(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Plot gross and net equity against the equal-weight universe.

    The gap between the two strategy lines is the whole transaction-cost bill;
    the gap between net and the reference line is what an investor would have
    given up by trading the signal instead of holding the universe.
    """
    equity = artifacts.daily_equity.copy()
    equity["date"] = pd.to_datetime(equity["date"])
    equity = equity.sort_values("date").reset_index(drop=True)
    starting = float(equity["starting_equity"].iloc[0])
    net_curve = np.concatenate([[starting], equity["ending_equity"].to_numpy(dtype=np.float64)])
    gross_curve = starting * np.cumprod(
        np.concatenate([[1.0], 1.0 + equity["gross_return"].to_numpy(dtype=np.float64)])
    )

    target = artifacts.target_panel()
    universe = target.mean(axis=1).reindex(equity["date"].dt.date.astype(str)).fillna(0.0)
    universe_curve = starting * np.cumprod(
        np.concatenate([[1.0], 1.0 + universe.to_numpy(dtype=np.float64)])
    )
    dates = [equity["date"].iloc[0] - pd.Timedelta(days=1), *equity["date"].tolist()]

    with figure(7.2, 4.0) as fig:
        axes = fig.add_subplot(111)
        axes.plot(dates, gross_curve, color=COLOURS["model"], label="strategy, before costs")
        axes.plot(dates, net_curve, color=COLOURS["portfolio"], label="strategy, after costs")
        axes.plot(
            dates,
            universe_curve,
            color=COLOURS["reference"],
            linestyle=(0, (4, 3)),
            label="equal-weight eligible universe",
        )
        axes.axhline(starting, color=COLOURS["neutral"], linewidth=1.0, zorder=2)
        axes.fill_between(
            dates,
            net_curve,
            gross_curve,
            color=COLOURS["model"],
            alpha=0.10,
            linewidth=0,
            zorder=1,
        )
        axes.set_ylabel("portfolio value (TRY)")
        axes.set_title(
            "Costs, not signal quality, decide the outcome", loc="left", color=COLOURS["ink"]
        )
        axes.legend(loc="lower left")
        fig.autofmt_xdate(rotation=0, ha="center")

        costs = float(artifacts.metrics["portfolio"]["cost_decomposition"]["total"])
        gross_return = float(artifacts.metrics["portfolio"]["gross_return"])
        net_return = float(artifacts.metrics["portfolio"]["net_return"])
        turnover = float(artifacts.metrics["portfolio"]["turnover"])
        sessions_below = int(np.sum(net_curve < starting))
        invested = int(artifacts.metrics["portfolio"]["invested_sessions"])
        sessions = int(artifacts.metrics["portfolio"]["session_count"])
        caption(
            fig,
            f"Gross {gross_return * 100:+.2f}%, net {net_return * 100:+.2f}%: "
            f"TRY {costs:,.0f} of modelled costs on {turnover:.1f}x turnover. The flat stretches "
            f"are sessions with no position: expected net return was non-positive for every "
            f"eligible name, so the strategy holds risk on only {invested} of {sessions} sessions "
            f"({invested / sessions * 100:.0f}%). Any figure annualised as if the capital were "
            f"continuously deployed therefore overstates the deployment.",
        )
        png, pdf = save_figure(fig, directory, "fig06_equity_curve")
    return {
        "figure": "fig06_equity_curve",
        "png": png.name,
        "pdf": pdf.name,
        "gross_return": gross_return,
        "net_return": net_return,
        "total_costs": costs,
        "sessions_below_start": sessions_below,
        "mark_count": int(len(net_curve)),
        "invested_sessions": invested,
        "session_count": sessions,
    }


def plot_cost_sensitivity(artifacts: RunArtifacts, directory: Path) -> dict[str, object]:
    """Show the breakeven cost level, holding every trading decision fixed.

    The signal is not re-optimised across the three cases, so the only thing
    that moves is the bill. Where the net line crosses zero is the cost
    multiple at which this strategy stops being profitable.
    """
    cases = artifacts.metrics["cost_sensitivity"]
    multipliers = sorted(float(case["cost_multiplier"]) for case in cases.values())
    by_multiplier = {float(case["cost_multiplier"]): case["metrics"] for case in cases.values()}
    gross = [float(by_multiplier[value]["gross_return"]) * 100 for value in multipliers]
    net = [float(by_multiplier[value]["net_return"]) * 100 for value in multipliers]
    breakeven = float(np.interp(0.0, net[::-1], multipliers[::-1])) if net[0] > 0 > net[-1] else 0.0

    with figure(7.2, 3.6) as fig:
        axes = fig.add_subplot(111)
        axes.plot(multipliers, gross, color=COLOURS["model"], marker="o", label="gross return")
        axes.plot(multipliers, net, color=COLOURS["portfolio"], marker="o", label="net return")
        axes.axhline(0.0, color=COLOURS["ink"], linewidth=1.0, zorder=4)
        if breakeven:
            axes.axvline(
                breakeven,
                color=COLOURS["adverse"],
                linewidth=1.2,
                linestyle=(0, (4, 3)),
                zorder=4,
            )
            axes.annotate(
                f"breakeven at {breakeven:.2f}x",
                xy=(breakeven, 0.0),
                xytext=(breakeven + 0.08, max(gross) * 0.45),
                fontsize=8,
                color=COLOURS["adverse"],
            )
        axes.axvline(1.0, color=COLOURS["neutral"], linewidth=1.0, zorder=2)
        axes.text(1.02, min(net) * 0.92, "modelled costs", fontsize=7.5, color=COLOURS["muted"])
        axes.set_xlabel("transaction-cost multiplier")
        axes.set_ylabel("total return over the window (%)")
        axes.set_title(
            "Fixed decisions, varying cost: where the edge disappears",
            loc="left",
            color=COLOURS["ink"],
        )
        axes.legend(loc="center right")
        axes.set_xlim(min(multipliers) - 0.12, max(multipliers) + 0.12)
        for value, level in zip(multipliers, net, strict=True):
            axes.annotate(
                f"{level:+.2f}%",
                xy=(value, level),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLOURS["ink"],
            )
        monotone = all(
            later <= earlier + 1e-12 for earlier, later in zip(net, net[1:], strict=False)
        )
        caption(
            fig,
            "Decisions are held fixed across the three cases, so only the bill moves; net return "
            f"is {'monotonically decreasing' if monotone else 'NOT monotone, which would be a bug'}"
            f" in the cost multiplier. Breakeven sits at {breakeven:.2f}x the modelled cost, so "
            f"the gross edge is exhausted below the costs an actual desk would pay.",
        )
        png, pdf = save_figure(fig, directory, "fig07_cost_sensitivity")
    return {
        "figure": "fig07_cost_sensitivity",
        "png": png.name,
        "pdf": pdf.name,
        "breakeven_multiplier": breakeven,
        "net_is_monotone": monotone,
        "net_returns": net,
    }
