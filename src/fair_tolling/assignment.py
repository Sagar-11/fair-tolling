"""Static traffic assignment on an AequilibraE project, in link flows.

Every top-level function takes an open AequilibraE ``Project`` plus the names of things
stored in it (a demand matrix, a results table) and writes its output to the project's
results database, the same place ``TrafficAssignment.save_results`` writes to. Nothing
is returned and the network tables are never modified.

Two solvers are available and write the same kind of table:

* AequilibraE's own equilibrium assignment (``build_graph`` -> ``traffic_class`` ->
  ``traffic_assignment`` -> ``run_assignment``),
* a Gurobi model of the link-node formulation (``assignment_model`` ->
  ``set_parameters`` -> ``run_model``, or ``solve_assignment`` for all three steps).

Only link flows are solved for. ``path_flows`` recovers route flows from the link flows
of any results table by flow decomposition, and ``od_costs`` gives the cost of each
O-D pair.

Assumptions, which hold for the projects built by ``create_ae_projects``: one mode
(car), one-way links, BPR travel times with the link fields ``vdf_fftt``, ``capacity``,
``vdf_alpha`` and ``vdf_beta``, and demand in the matrix core ``volume``. Tolls are in
time units (value of time 1) unless a value of time is given.
"""

import json
from functools import cache

import gurobipy as gp
import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.paths import TrafficAssignment, TrafficClass

# Link fields and matrix core written by create_ae_projects
TIME_FIELD = "vdf_fftt"
CAPACITY_FIELD = "capacity"
ALPHA_FIELD = "vdf_alpha"
BETA_FIELD = "vdf_beta"
MATRIX_CORE = "volume"

# Graph-only fields (never written to the project database)
SO_ALPHA_FIELD = "so_alpha"
TOLL_FIELD = "toll"

# Columns of a results table. FLOW and TIME are the names AequilibraE itself uses.
FLOW = f"{MATRIX_CORE}_ab"
TIME = "Congested_Time_AB"
TOLL = "toll"

TIME_LIMIT = 60  # seconds, for every Gurobi model
STATUS = {gp.GRB.OPTIMAL: "optimal", gp.GRB.TIME_LIMIT: "time limit"}


# ---------------------------------------------------------------------------
# Reading the project
# ---------------------------------------------------------------------------


def load_links(project: Project) -> pd.DataFrame:
    """Links of the project, indexed by ``link_id``, with their BPR parameters."""
    links = project.network.links.data.set_index("link_id").sort_index()
    if (links.direction != 1).any():
        raise NotImplementedError("Only one-way links (direction 1) are supported")
    return pd.DataFrame(
        {
            "a_node": links.a_node.astype(int),
            "b_node": links.b_node.astype(int),
            "fftt": links[TIME_FIELD].astype(float),
            "capacity": links[f"{CAPACITY_FIELD}_ab"].astype(float),
            "alpha": links[ALPHA_FIELD].astype(float),
            "beta": links[BETA_FIELD].astype(float),
        }
    )


def load_demand(project: Project, matrix_name: str) -> pd.Series:
    """Positive O-D demand of a project matrix, indexed by (origin, destination)."""
    matrix = project.matrices.get_matrix(matrix_name)
    zones = np.asarray(matrix.index).flatten()
    volume = pd.DataFrame(np.array(matrix.get_matrix(MATRIX_CORE)), index=zones, columns=zones)
    matrix.close()
    demand = volume.stack().rename("demand").rename_axis(["origin", "destination"])
    origin, destination = demand.index.get_level_values(0), demand.index.get_level_values(1)
    return demand[(demand > 0) & (origin != destination)]


def load_results(project: Project, table_name: str) -> pd.DataFrame:
    """A table of the results database, indexed by ``link_id``."""
    return project.results.get_results(table_name).set_index("link_id")


def link_flows(project: Project, table_name: str) -> pd.Series:
    """Link flows of a results table (from either solver), indexed by ``link_id``."""
    return load_results(project, table_name)[FLOW].rename("flow")


def save_results(project: Project, table_name: str, data: pd.DataFrame, procedure: str, report: dict) -> None:
    """Save a table indexed by ``link_id`` to the results database, replacing ``table_name``."""
    drop_results(project, table_name)
    record = project.results.new_record(
        table_name, procedure=procedure, procedure_report=report, timestamp=str(pd.Timestamp.now())
    )
    record.set_data(data.rename_axis("link_id"))


def drop_results(project: Project, table_name: str) -> None:
    """Delete a results table if it exists, so that a run can be repeated."""
    if project.results.check_exists(table_name):
        project.results.delete_record(table_name)


# ---------------------------------------------------------------------------
# Link costs
# ---------------------------------------------------------------------------


def bpr_time(links: pd.DataFrame, flow: pd.Series) -> pd.Series:
    """BPR travel time of each link, ``t0 * (1 + alpha * (v / c) ** beta)``."""
    return links.fftt * (1 + links.alpha * (flow / links.capacity) ** links.beta)


def total_time(links: pd.DataFrame, flow: pd.Series) -> float:
    """Total system travel time, ``sum of v * t(v)``."""
    return float((flow * bpr_time(links, flow)).sum())


def od_costs(links: pd.DataFrame, demand: pd.Series, link_cost: pd.Series) -> pd.Series:
    """Cost of the cheapest route of every O-D pair for the given link costs.

    Uses label correcting (Bellman-Ford), so negative link costs such as subsidies are
    fine as long as there is no negative cycle.
    """
    nodes = pd.Index(sorted(set(links.a_node) | set(links.b_node)))
    tail, head = nodes.get_indexer(links.a_node), nodes.get_indexer(links.b_node)
    cost = link_cost.reindex(links.index).to_numpy()

    labels = {}
    for origin in demand.index.unique("origin"):
        dist = np.full(len(nodes), np.inf)
        dist[nodes.get_loc(origin)] = 0.0
        for _ in nodes:
            shorter = dist.copy()
            np.minimum.at(shorter, head, dist[tail] + cost)
            if np.array_equal(shorter, dist):
                break
            dist = shorter
        labels[origin] = pd.Series(dist, index=nodes)
    return pd.Series([labels[o][d] for o, d in demand.index], index=demand.index, name="cost")


# ---------------------------------------------------------------------------
# AequilibraE assignment
# ---------------------------------------------------------------------------


def build_graph(project: Project, tolls: pd.Series | None = None, mode: str = "c"):
    """Graph for one mode with free-flow time as the cost and through traffic at centroids.

    ``tolls`` (indexed by ``link_id``) are added to the graph only, as the field
    ``toll``; the project database is left untouched. The graph also gets the field
    ``so_alpha`` = alpha * (beta + 1), which turns BPR into the marginal travel time.
    """
    project.network.build_graphs(modes=[mode])
    graph = project.network.graphs[mode]
    graph.set_graph(TIME_FIELD)
    graph.set_skimming([TIME_FIELD])
    graph.set_blocked_centroid_flows(False)
    graph.graph[SO_ALPHA_FIELD] = graph.graph[ALPHA_FIELD] * (graph.graph[BETA_FIELD] + 1)
    if tolls is not None:
        graph.graph[TOLL_FIELD] = graph.graph.link_id.map(tolls).fillna(0.0).to_numpy()
    return graph


def traffic_class(project: Project, graph, matrix_name: str, vot: float = 1.0) -> TrafficClass:
    """Car traffic class for a project demand matrix. Pays the graph's tolls, if it has any.

    The generalised cost of a link is ``time + toll / vot``, so ``vot`` is in money per
    unit of time. AequilibraE does not accept negative tolls.
    """
    demand = project.matrices.get_matrix(matrix_name)
    demand.computational_view([MATRIX_CORE])
    cars = TrafficClass("car", graph, demand)
    if TOLL_FIELD in graph.graph.columns:
        cars.set_fixed_cost(TOLL_FIELD)
        cars.set_vot(vot)
    return cars


def traffic_assignment(cars: TrafficClass, objective: str = "UE") -> TrafficAssignment:
    """BPR assignment by bi-conjugate Frank-Wolfe.

    ``objective`` is "UE" (user equilibrium) or "SO" (system optimum). The system
    optimum is the user equilibrium for marginal travel times, which for BPR only
    means replacing alpha by alpha * (beta + 1). The gap and the iteration limit come
    from the project's ``parameters.yml``.
    """
    assignment = TrafficAssignment()
    assignment.add_class(cars)
    assignment.set_vdf("BPR")
    assignment.set_vdf_parameters({"alpha": ALPHA_FIELD if objective == "UE" else SO_ALPHA_FIELD, "beta": BETA_FIELD})
    assignment.set_capacity_field(CAPACITY_FIELD)
    assignment.set_time_field(TIME_FIELD)
    assignment.set_algorithm("bfw")
    return assignment


def run_assignment(project: Project, assignment: TrafficAssignment, table_name: str) -> None:
    """Execute an assignment and save its link results as ``table_name``.

    After a system-optimum run ``Congested_Time_AB`` holds marginal times. Use
    ``bpr_time`` on the saved flows when travel times are needed.
    """
    assignment.execute()
    drop_results(project, table_name)
    assignment.save_results(table_name, project=project)


# ---------------------------------------------------------------------------
# Gurobi assignment
# ---------------------------------------------------------------------------


@cache
def _quiet_env() -> gp.Env:
    return gp.Env(params={"OutputFlag": 0})


def new_model(name: str) -> gp.Model:
    """Empty Gurobi model that prints nothing."""
    return gp.Model(name, env=_quiet_env())


def add_origin_flows(model: gp.Model, links: pd.DataFrame, demand: pd.Series) -> gp.tupledict:
    """Add the variables ``x[origin, link] >= 0`` and their flow conservation constraints.

    At every node the flow of an origin that leaves minus the flow that enters equals
    the trips the origin sends out (at the origin itself) or minus the trips it sends
    to that node (everywhere else).
    """
    nodes = sorted(set(links.a_node) | set(links.b_node))
    out_links = links.groupby("a_node").groups
    in_links = links.groupby("b_node").groups
    origins = sorted(demand.index.unique("origin"))

    x = model.addVars(origins, links.index, name="x")
    for origin in origins:
        trips = demand.xs(origin, level="origin")
        for node in nodes:
            net = trips.sum() if node == origin else -trips.get(node, 0.0)
            leaving = gp.quicksum(x[origin, a] for a in out_links.get(node, []))
            entering = gp.quicksum(x[origin, a] for a in in_links.get(node, []))
            model.addConstr(leaving - entering == net, name=f"node[{origin},{node}]")
    return x


def assignment_model(
    links: pd.DataFrame, demand: pd.Series, objective: str = "UE", tolls: pd.Series | None = None
) -> gp.Model:
    """Gurobi model of the traffic assignment in link flows.

    Variables: ``x[o, a]`` flow from origin o on link a, ``v[a]`` total flow and
    ``z[a] = (v[a] / c[a]) ** (beta[a] + 1)``. With z the objective is linear:

    * "UE": ``sum (t0 + toll) * v + t0 * alpha * c / (beta + 1) * z``  (Beckmann)
    * "SO": ``sum t0 * v + t0 * alpha * c * z``                        (total travel time)

    ``tolls`` (per link, in time units, may be negative) only enter the user equilibrium.
    The flow variables are kept on the model as ``model._flow``.
    """
    if objective not in ("UE", "SO"):
        raise ValueError("objective must be 'UE' or 'SO'")
    model = new_model(f"assignment_{objective}")
    x = add_origin_flows(model, links, demand)
    v = model.addVars(links.index, name="v")
    model.addConstrs((v[a] == x.sum("*", a) for a in links.index), name="link")

    congested = links.index[links.alpha > 0]
    z = model.addVars(congested, name="z")
    for a in congested:
        model.addGenConstrNL(z[a], (v[a] / links.capacity[a]) ** (links.beta[a] + 1), name=f"power[{a}]")

    cost = links.fftt + (tolls.reindex(links.index).fillna(0.0) if tolls is not None and objective == "UE" else 0.0)
    weight = links.fftt * links.alpha * links.capacity / (links.beta + 1 if objective == "UE" else 1.0)
    model.setObjective(
        gp.quicksum(cost[a] * v[a] for a in links.index) + gp.quicksum(weight[a] * z[a] for a in congested)
    )
    model._links, model._flow = links, v
    return model


def set_parameters(model: gp.Model, project: Project) -> None:
    """Take the relative gap from the project's ``parameters.yml`` and cap the run time."""
    model.Params.MIPGap = project.parameters["assignment"]["equilibrium"]["rgap"]
    model.Params.TimeLimit = TIME_LIMIT


def solution_report(model: gp.Model) -> dict:
    """Status, objective and run time of an optimised model. Fails if it has no solution."""
    if model.SolCount == 0:
        raise RuntimeError(f"{model.ModelName}: no solution (Gurobi status {model.Status})")
    return {
        "solver": "gurobi",
        "model": model.ModelName,
        "status": STATUS.get(model.Status, str(model.Status)),
        "objective": model.ObjVal,
        "seconds": round(model.Runtime, 3),
    }


def run_model(project: Project, model: gp.Model, table_name: str) -> None:
    """Optimise an assignment model and save link flows and travel times as ``table_name``."""
    model.optimize()
    report = solution_report(model)
    links = model._links
    flow = pd.Series({a: max(model._flow[a].X, 0.0) for a in links.index})
    data = pd.DataFrame({FLOW: flow, TIME: bpr_time(links, flow)})
    report["total_time"] = total_time(links, flow)
    save_results(project, table_name, data, "traffic assignment", report)


def solve_assignment(
    project: Project, matrix_name: str, table_name: str, objective: str = "UE", toll_table: str | None = None
) -> None:
    """Solve the assignment of a project matrix with Gurobi and save it as ``table_name``.

    With ``toll_table`` (a results table with a ``toll`` column) the user equilibrium
    is the tolled one.
    """
    tolls = load_results(project, toll_table)[TOLL] if toll_table else None
    model = assignment_model(load_links(project), load_demand(project, matrix_name), objective, tolls)
    set_parameters(model, project)
    run_model(project, model, table_name)


# ---------------------------------------------------------------------------
# Route flows from link flows
# ---------------------------------------------------------------------------


def origin_flows(links: pd.DataFrame, demand: pd.Series, flow: pd.Series) -> pd.DataFrame:
    """Split link flows by origin. Returns a table with links as rows and origins as columns.

    Linear programme: carry every origin's trips at least total travel time while
    using, on each link, no more than the flow it has. For equilibrium or optimal
    flows the origin flows add up to the link flows again, and since travel times are
    positive no origin flow contains a cycle. The split is in general not unique.
    """
    model = new_model("origin_flows")
    model.Params.TimeLimit = TIME_LIMIT
    x = add_origin_flows(model, links, demand)
    slack = 1e-9 * demand.sum()
    model.addConstrs((x.sum("*", a) <= max(flow[a], 0.0) + slack for a in links.index), name="link")
    time = bpr_time(links, flow)
    model.setObjective(gp.quicksum(time[a] * x[o, a] for o, a in x))
    model.optimize()
    solution_report(model)
    origins = sorted(demand.index.unique("origin"))
    return pd.DataFrame({o: [max(x[o, a].X, 0.0) for a in links.index] for o in origins}, index=links.index)


def peel_paths(flow: dict, tail: dict, in_links: dict, origin, trips: dict, tolerance: float) -> list[tuple]:
    """Decompose the acyclic link flow of one origin into route flows.

    For each destination walk back to the origin, always taking the entering link
    that still carries most flow. Send as much as the route and the remaining trips
    allow, take it off the links and repeat until the trips are used up. Every pass
    empties a link or a destination, so the loop is short. ``flow`` is consumed.

    Returns a list of (destination, route as a tuple of link ids, flow).
    """
    paths = []
    for destination, remaining in trips.items():
        while remaining > tolerance:
            path, node = [], destination
            while node != origin and len(path) <= len(flow):
                link = max(in_links.get(node, []), key=flow.get, default=None)
                if link is None:
                    break
                path.append(link)
                node = tail[link]
            sent = min([remaining] + [flow[a] for a in path]) if node == origin else 0.0
            if sent <= tolerance:
                break
            for a in path:
                flow[a] -= sent
            remaining -= sent
            paths.append((destination, tuple(reversed(path)), sent))
    return paths


def decompose_flows(links: pd.DataFrame, demand: pd.Series, flow: pd.Series) -> pd.DataFrame:
    """Route flows that reproduce the link flows.

    Returns one row per used route with the columns ``origin``, ``destination``,
    ``path`` (tuple of link ids), ``flow`` and ``time`` (travel time of the route at
    the given link flows).
    """
    by_origin = origin_flows(links, demand, flow)
    tail = links.a_node.to_dict()
    in_links = {node: list(ids) for node, ids in links.groupby("b_node").groups.items()}
    tolerance = 1e-6 * demand.sum()
    time = bpr_time(links, flow)

    rows = []
    for origin in by_origin.columns:
        trips = demand.xs(origin, level="origin").to_dict()
        for destination, path, sent in peel_paths(by_origin[origin].to_dict(), tail, in_links, origin, trips, tolerance):
            rows.append((origin, destination, path, sent, time[list(path)].sum()))
    return pd.DataFrame(rows, columns=["origin", "destination", "path", "flow", "time"])


def path_flows(project: Project, matrix_name: str, table_name: str) -> pd.DataFrame:
    """Route flows for the link flows saved in a results table. See ``decompose_flows``."""
    return decompose_flows(load_links(project), load_demand(project, matrix_name), link_flows(project, table_name))


def report_of(project: Project, table_name: str) -> dict:
    """The report stored with a results table (status, objective, total time, ...)."""
    report = project.results.get_record(table_name).procedure_report
    while isinstance(report, str):  # AequilibraE stores the report as JSON text
        report = json.loads(report)
    return report
