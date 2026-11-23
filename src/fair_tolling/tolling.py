""" congestion tolls for an AequilibraE project 
refer - Hearn and Ramana, 1998.

we solve the system optimum for the link flows v* ,assignment.solve_assignment
the toll set is all tolls b for which v* is a tolled user equilibrium we try to 
optimise a secondary objective over the toll set as according to the paper. 


solve_tolls(project, matrix_name, method) runs for one of ``METHODS``
and saves the tolls as a results table with a toll column. Tolls are in time units.
"""

import gurobipy as gp
import pandas as pd
from aequilibrae import Project

from fair_tolling.assignment import (
    TIME_LIMIT,
    TOLL,
    bpr_time,
    link_flows,
    load_demand,
    load_links,
    new_model,
    od_costs,
    save_results,
    solution_report,
    total_time,
)

METHODS = {
    "MSCP": "marginal social cost pricing, toll = v * t'(v)",
    "MINSYS": "smallest total toll collected, no subsidies",
    "MINMAX": "smallest maximum link toll, no subsidies",
    "MINTB": "fewest toll booths, no subsidies",
    "RH": "Robin Hood: tolls and subsidies that net to zero, least money moved",
    "MINTB_RH": "fewest toll booths with tolls and subsidies that net to zero",
    "MINHARM": "smallest worst-case cost increase of an O-D pair, revenue refunded equally",
}
SUBSIDY_METHODS = ("RH", "MINTB_RH")
ZERO_TOLL = 1e-6  # tolls smaller than this are treated as no toll


def mscp_tolls(links: pd.DataFrame, flow: pd.Series) -> pd.Series:
    """Marginal social cost tolls ``v * t'(v) = t0 * alpha * beta * (v / c) ** beta``."""
    return (links.fftt * links.alpha * links.beta * (flow / links.capacity) ** links.beta).rename(TOLL)


def toll_model(
    links: pd.DataFrame,
    demand: pd.Series,
    flow: pd.Series,
    method: str,
    rgap: float = 1e-4,
    ue_cost: pd.Series | None = None,
) -> gp.Model:
    """Gurobi model that picks a toll vector from the toll set of the optimal ``flow``.

    * "MINSYS":   min total toll collected,                 tolls >= 0
    * "MINMAX":   min the largest link toll,                tolls >= 0
    * "MINTB":    min number of tolled links (binary),      tolls >= 0
    * "RH":       min tolls collected (= subsidies paid),   total toll = 0
    * "MINTB_RH": min number of tolled links,               total toll = 0
    * "MINHARM":  min the largest (cost - refund) / ``ue_cost`` over O-D pairs, with
      the revenue refunded equally to all travellers,       tolls >= 0

    With subsidies the paper's Robin Hood programme has a constant objective, so every
    point of the toll set with zero revenue is optimal. Here "RH" moves the least
    money, and no subsidy exceeds the travel time of its link (cost stays >= 0).
    ``ue_cost`` (per O-D pair, the cost without tolls) is only needed for "MINHARM".
    The toll variables are kept on the model as ``model._toll``.
    """
    if method not in METHODS or method == "MSCP":
        raise ValueError(f"method must be one of {[m for m in METHODS if m != 'MSCP']}")
    time = bpr_time(links, flow)
    subsidies = method in SUBSIDY_METHODS
    nodes = sorted(set(links.a_node) | set(links.b_node))
    origins = sorted(demand.index.unique("origin"))

    model = new_model(f"tolls_{method}")
    toll = model.addVars(links.index, lb=-gp.GRB.INFINITY if subsidies else 0.0, name="toll")
    rho = model.addVars(origins, nodes, lb=-gp.GRB.INFINITY, name="rho")

    # Toll set
    model.addConstrs((rho[o, o] == 0 for o in origins), name="origin")
    model.addConstrs(
        (rho[o, links.b_node[a]] - rho[o, links.a_node[a]] <= time[a] + toll[a] for o in origins for a in links.index),
        name="route",
    )
    revenue = gp.quicksum(flow[a] * toll[a] for a in links.index)
    cheapest = gp.quicksum(trips * rho[o, d] for (o, d), trips in demand.items())
    system_time = float((flow * time).sum())
    model.addConstr(system_time + revenue - cheapest <= rgap * system_time, name="equilibrium")

    # Secondary objective
    if subsidies:
        model.addConstrs((toll[a] >= -time[a] for a in links.index), name="no_negative_cost")
        model.addConstr(revenue == 0, name="revenue_neutral")
    if method == "MINSYS":
        model.setObjective(revenue)
    elif method == "MINMAX":
        largest = model.addVar(name="largest")
        model.addConstrs((largest >= toll[a] for a in links.index), name="largest")
        model.setObjective(largest)
    elif method == "RH":
        collected = model.addVars(links.index, name="collected")
        model.addConstrs((collected[a] >= toll[a] for a in links.index), name="collected")
        model.setObjective(gp.quicksum(flow[a] * collected[a] for a in links.index))
    elif method in ("MINTB", "MINTB_RH"):
        big = float(time.sum())  # exceeds any sensible toll
        booth = model.addVars(links.index, vtype=gp.GRB.BINARY, name="booth")
        model.addConstrs((toll[a] <= big * booth[a] for a in links.index), name="booth_toll")
        if subsidies:
            model.addConstrs((toll[a] >= -big * booth[a] for a in links.index), name="booth_subsidy")
        model.setObjective(booth.sum())
    elif method == "MINHARM":
        if ue_cost is None:
            raise ValueError("MINHARM needs ue_cost, the O-D costs without tolls")
        harm = model.addVar(name="harm")
        refund = revenue / float(demand.sum())
        model.addConstrs((rho[o, d] - refund <= harm * ue_cost[o, d] for o, d in demand.index), name="harm")
        model.setObjective(harm)

    model._toll = toll
    return model


def toll_summary(links: pd.DataFrame, flow: pd.Series, tolls: pd.Series) -> dict:
    """Total toll collected, its share of total travel time, toll booths and largest toll."""
    system_time = total_time(links, flow)
    revenue = float((tolls * flow).sum())
    return {
        "total_time": system_time,
        "total_tolls": revenue,
        "tolls_over_time_pct": 100 * revenue / system_time,
        "toll_booths": int((tolls.abs() > ZERO_TOLL).sum()),
        "max_toll": float(tolls.max()),
    }


def solve_tolls(
    project: Project,
    matrix_name: str,
    method: str,
    so_table: str = "so",
    ue_table: str = "ue",
    table_name: str | None = None,
) -> None:
    """Compute first-best tolls for a project and save them in its results database.

    ``so_table`` is the results table with the system-optimal link flows. The tolls go
    to ``table_name`` (default ``toll_<method>``, lower case) with the column ``toll``;
    ``assignment.report_of`` gives the totals and the solver status stored with it.
    ``ue_table`` (untolled user equilibrium) is only read by "MINHARM".
    """
    links, demand = load_links(project), load_demand(project, matrix_name)
    flow = link_flows(project, so_table)

    if method == "MSCP":
        tolls, report = mscp_tolls(links, flow), {"status": "closed form"}
    else:
        rgap = project.parameters["assignment"]["equilibrium"]["rgap"]
        ue_cost = None
        if method == "MINHARM":
            ue_cost = od_costs(links, demand, bpr_time(links, link_flows(project, ue_table)))
        model = toll_model(links, demand, flow, method, rgap, ue_cost)
        model.Params.TimeLimit = TIME_LIMIT
        model.optimize()
        report = solution_report(model)
        tolls = pd.Series({a: model._toll[a].X for a in links.index}, name=TOLL)

    tolls = tolls.where(tolls.abs() > ZERO_TOLL, 0.0)
    report.update(method=method, so_table=so_table, **toll_summary(links, flow, tolls))
    save_results(project, table_name or f"toll_{method.lower()}", tolls.to_frame(), "toll pricing", report)


def load_tolls(project: Project, table_name: str) -> pd.Series:
    """Link tolls of a results table written by ``solve_tolls``, indexed by ``link_id``."""
    return project.results.get_results(table_name).set_index("link_id")[TOLL]
