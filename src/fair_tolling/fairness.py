"""Who gains and who loses from a toll scheme.

Travellers are grouped by O-D pair and all have value of time 1, so costs, tolls and
refunds are in time units. For an O-D pair w:

* ``cost_before``  travel time at the untolled user equilibrium (the same for everyone
  on w, because all used routes are equally fast),
* ``cost``         travel time plus tolls at the tolled equilibrium. It is again the
  same for everyone on w: routes that take longer are tolled less,
* ``refund``       what each traveller gets back. Nothing, or the toll revenue divided
  equally over all travellers,
* ``net_cost``     ``cost - refund``,
* ``ratio``        ``net_cost / cost_before``; above 1 the pair is worse off.

The following fairness criteria is used:

* F1 equal treatment: identical travellers face the same net cost. True by
  construction for ``net_cost``, but not for travel time alone (see ``od_times``).
* F2 Pareto improvement: nobody is worse off, ``share_worse_off == 0``.
* F3 bounded harm: ``max_ratio`` is the largest ratio of any O-D pair.
* F4 inequality: ``gini`` of net cost over travellers, against ``gini_before``.
"""

import numpy as np
import pandas as pd
from aequilibrae import Project

from fair_tolling.assignment import bpr_time, link_flows, load_demand, load_links, load_results, od_costs, TOLL

WORSE_OFF = 1e-3  # a pair is worse off when its net cost rises by more than 0.1 %


def od_outcomes(
    links: pd.DataFrame,
    demand: pd.Series,
    ue_flow: pd.Series,
    flow: pd.Series,
    tolls: pd.Series | None = None,
    refund: bool = False,
) -> pd.DataFrame:
    """Cost of every O-D pair before (``ue_flow``, no tolls) and after (``flow`` with ``tolls``).

    ``refund=True`` hands the toll revenue back in equal shares to all travellers.
    Returns a table indexed by (origin, destination).
    """
    tolls = pd.Series(0.0, index=links.index) if tolls is None else tolls.reindex(links.index).fillna(0.0)
    out = pd.DataFrame({"demand": demand})
    out["cost_before"] = od_costs(links, demand, bpr_time(links, ue_flow))
    out["cost"] = od_costs(links, demand, bpr_time(links, flow) + tolls)
    out["refund"] = float((tolls * flow).sum()) / demand.sum() if refund else 0.0
    out["net_cost"] = out.cost - out.refund
    out["ratio"] = out.net_cost / out.cost_before
    return out


def gini(values, weights=None) -> float:
    """Gini coefficient of ``values``, optionally weighted (0 = all equal)."""
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values, weights = values[order], weights[order] / weights.sum()
    share = np.cumsum(weights * values) / np.sum(weights * values)  
    return float(1 - np.sum(weights * (share + np.concatenate([[0.0], share[:-1]]))))


def fairness_summary(outcomes: pd.DataFrame) -> pd.Series:
    """Fairness measures of an ``od_outcomes`` table, weighted by travellers."""
    trips = outcomes.demand
    worse = outcomes.ratio > 1 + WORSE_OFF
    return pd.Series(
        {
            "mean_cost_before": np.average(outcomes.cost_before, weights=trips),
            "mean_net_cost": np.average(outcomes.net_cost, weights=trips),
            "max_ratio": outcomes.ratio.max(),
            "min_ratio": outcomes.ratio.min(),
            "share_worse_off": trips[worse].sum() / trips.sum(),
            "gini_before": gini(outcomes.cost_before, trips),
            "gini": gini(outcomes.net_cost, trips),
        }
    )


def od_times(paths: pd.DataFrame) -> pd.DataFrame:
    """Travel time experienced within each O-D pair, from a table of route flows.

    Returns the mean (flow weighted), the fastest and the slowest used route, and
    ``spread = slowest / fastest``. At a user equilibrium the spread is 1. At a system
    optimum it is not: some travellers of the same pair are sent on longer routes.
    """
    weighted = paths.assign(flow_time=paths.flow * paths.time)
    by_od = weighted.groupby(["origin", "destination"])
    out = pd.DataFrame(
        {
            "routes": by_od.size(),
            "mean_time": by_od.flow_time.sum() / by_od.flow.sum(),
            "fastest": by_od.time.min(),
            "slowest": by_od.time.max(),
        }
    )
    out["spread"] = out.slowest / out.fastest
    return out


def evaluate(
    project: Project,
    matrix_name: str,
    table_name: str,
    toll_table: str | None = None,
    refund: bool = False,
    ue_table: str = "ue",
) -> pd.DataFrame:
    """``od_outcomes`` for results tables of a project.

    ``table_name`` holds the link flows after the intervention, ``toll_table`` the
    tolls that go with it and ``ue_table`` the untolled user equilibrium.
    """
    tolls = load_results(project, toll_table)[TOLL] if toll_table else None
    return od_outcomes(
        load_links(project),
        load_demand(project, matrix_name),
        link_flows(project, ue_table),
        link_flows(project, table_name),
        tolls,
        refund,
    )
