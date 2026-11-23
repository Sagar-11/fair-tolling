"""Tests for the assignment, toll pricing and fairness functions.

The model functions are tested on small tables, without a project. One test at the end
runs the whole chain on the ``nine-node`` AequilibraE project (in a temporary copy).
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fair_tolling import assignment as fa
from fair_tolling import fairness as ff
from fair_tolling import tolling as ft

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_links(rows):
    columns = ["a_node", "b_node", "fftt", "capacity", "alpha", "beta"]
    return pd.DataFrame(rows, columns=columns, index=pd.RangeIndex(1, len(rows) + 1, name="link_id"))


def make_demand(trips):
    index = pd.MultiIndex.from_tuples(trips.keys(), names=["origin", "destination"])
    return pd.Series(list(trips.values()), index=index, name="demand", dtype=float)


def solve(links, demand, objective="UE", tolls=None):
    model = fa.assignment_model(links, demand, objective, tolls)
    model.optimize()
    return pd.Series({a: model._flow[a].X for a in links.index})


@pytest.fixture
def braess():
    """The Braess network of gmns-data/hearn_example: 4000 trips from node 1 to node 4."""
    links = make_links(
        [
            (1, 2, 1.0, 4000.0, 40.0, 1.0),
            (1, 3, 45.0, 4000.0, 0.0, 1.0),
            (2, 4, 45.0, 4000.0, 0.0, 1.0),
            (3, 4, 1.0, 4000.0, 40.0, 1.0),
            (2, 3, 0.01, 4000.0, 0.0, 1.0),
        ]
    )
    return links, make_demand({(1, 4): 4000.0})


@pytest.fixture
def two_od():
    """Two origins that share a congested link (3 -> 4) and each have a private bypass."""
    links = make_links(
        [
            (1, 3, 1.0, 10.0, 0.15, 4.0),
            (2, 3, 1.0, 10.0, 0.15, 4.0),
            (3, 4, 2.0, 10.0, 0.15, 4.0),
            (1, 4, 6.0, 10.0, 0.15, 4.0),
            (2, 4, 8.0, 10.0, 0.15, 4.0),
        ]
    )
    return links, make_demand({(1, 4): 10.0, (2, 4): 10.0})


# --- assignment -----------------------------------------------------------


def test_bpr_time():
    links = make_links([(1, 2, 10.0, 100.0, 0.15, 4.0)])
    assert fa.bpr_time(links, pd.Series({1: 0.0}))[1] == pytest.approx(10.0)
    assert fa.bpr_time(links, pd.Series({1: 100.0}))[1] == pytest.approx(11.5)
    assert fa.total_time(links, pd.Series({1: 100.0})) == pytest.approx(1150.0)


def test_user_equilibrium_on_braess(braess):
    links, demand = braess
    flow = solve(links, demand, "UE")
    # everyone takes the shortcut 1 -> 2 -> 3 -> 4 and needs 41 + 0.01 + 41
    assert flow.to_list() == pytest.approx([4000, 0, 0, 4000, 4000], abs=1)
    assert fa.total_time(links, flow) == pytest.approx(4000 * 82.01, rel=1e-4)


def test_system_optimum_on_braess(braess):
    links, demand = braess
    flow = solve(links, demand, "SO")
    assert flow.to_list() == pytest.approx([2199.5, 1800.5, 1800.5, 2199.5, 399.0], abs=2)
    assert fa.total_time(links, flow) < fa.total_time(links, solve(links, demand, "UE"))


def test_od_costs_allow_subsidies(braess):
    links, demand = braess
    cost = pd.Series([5.0, 1.0, 1.0, 5.0, -3.0], index=links.index)
    assert fa.od_costs(links, demand, cost)[1, 4] == pytest.approx(6.0)  # 1 -> 2 -> 4 or 1 -> 3 -> 4
    cost[5] = -9.0
    assert fa.od_costs(links, demand, cost)[1, 4] == pytest.approx(1.0)  # 1 -> 2 -> 3 -> 4


# --- route flows ----------------------------------------------------------


def test_peel_paths_single_origin():
    #   1 -> 2 -> 4 carries 3,  1 -> 3 -> 4 carries 2,  1 -> 3 carries 1 more that stops at 3
    flow = {1: 3.0, 2: 3.0, 3: 3.0, 4: 2.0}
    tail = {1: 1, 2: 1, 3: 2, 4: 3}
    in_links = {2: [1], 3: [2], 4: [3, 4]}
    paths = fa.peel_paths(flow, tail, in_links, origin=1, trips={4: 5.0, 3: 1.0}, tolerance=1e-9)
    assert sorted(paths) == [(3, (2,), 1.0), (4, (1, 3), 3.0), (4, (2, 4), 2.0)]
    assert all(abs(left) < 1e-9 for left in flow.values())


@pytest.mark.parametrize("objective", ["UE", "SO"])
def test_decomposition_reproduces_link_flows(two_od, objective):
    links, demand = two_od
    flow = solve(links, demand, objective)
    paths = fa.decompose_flows(links, demand, flow)

    carried = paths.groupby(["origin", "destination"]).flow.sum()
    assert carried.to_list() == pytest.approx(demand.to_list(), abs=1e-3)
    rebuilt = pd.Series(0.0, index=links.index)
    for path, sent in zip(paths.path, paths.flow):
        rebuilt[list(path)] += sent
    assert rebuilt.to_list() == pytest.approx(flow.to_list(), abs=1e-3)
    # every route really runs from its origin to its destination
    for row in paths.itertuples():
        assert links.a_node[row.path[0]] == row.origin and links.b_node[row.path[-1]] == row.destination


def test_used_routes_are_equally_fast_at_equilibrium(two_od):
    links, demand = two_od
    paths = fa.decompose_flows(links, demand, solve(links, demand, "UE"))
    assert ff.od_times(paths).spread.max() == pytest.approx(1.0, abs=1e-2)


# --- tolls ----------------------------------------------------------------


def test_mscp_tolls_formula():
    links = make_links([(1, 2, 10.0, 100.0, 0.15, 4.0), (1, 2, 5.0, 100.0, 0.0, 4.0)])
    tolls = ft.mscp_tolls(links, pd.Series({1: 100.0, 2: 50.0}))
    assert tolls.to_list() == pytest.approx([10 * 0.15 * 4, 0.0])


@pytest.mark.parametrize("method", [m for m in ft.METHODS if m not in ("MSCP", "MINHARM")])
def test_tolled_equilibrium_is_the_system_optimum(two_od, method):
    links, demand = two_od
    so = solve(links, demand, "SO")
    model = ft.toll_model(links, demand, so, method)
    model.optimize()
    tolls = pd.Series({a: model._toll[a].X for a in links.index})

    tolled = solve(links, demand, "UE", tolls)
    assert tolled.to_list() == pytest.approx(so.to_list(), abs=0.2)
    if method in ft.SUBSIDY_METHODS:
        assert float((tolls * so).sum()) == pytest.approx(0.0, abs=1e-6)
    else:
        assert tolls.min() >= -1e-9


def test_minsys_collects_least_and_mintb_has_one_booth(braess):
    links, demand = braess
    so = solve(links, demand, "SO")

    def tolls_of(method):
        model = ft.toll_model(links, demand, so, method)
        model.optimize()
        return pd.Series({a: model._toll[a].X for a in links.index})

    revenue = {m: float((tolls_of(m) * so).sum()) for m in ("MINSYS", "MINMAX")}
    revenue["MSCP"] = float((ft.mscp_tolls(links, so) * so).sum())
    assert revenue["MINSYS"] <= revenue["MINMAX"] + 1e-6
    assert revenue["MINSYS"] < 0.2 * revenue["MSCP"]  # a toll on the shortcut alone is enough
    assert ft.toll_summary(links, so, tolls_of("MINTB"))["toll_booths"] == 1
    assert tolls_of("MINMAX").max() <= tolls_of("MINSYS").max() + 1e-6


def test_unknown_method_is_rejected(braess):
    links, demand = braess
    with pytest.raises(ValueError):
        ft.toll_model(links, demand, pd.Series(1.0, index=links.index), "MSCP")


# --- fairness -------------------------------------------------------------


def test_gini():
    assert ff.gini([5, 5, 5]) == pytest.approx(0.0)
    assert ff.gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert ff.gini([1, 3], weights=[3, 1]) == pytest.approx(ff.gini([1, 1, 1, 3]))


def test_refund_returns_the_revenue(two_od):
    links, demand = two_od
    ue, so = solve(links, demand, "UE"), solve(links, demand, "SO")
    tolls = ft.mscp_tolls(links, so)
    kept = ff.od_outcomes(links, demand, ue, so, tolls, refund=False)
    refunded = ff.od_outcomes(links, demand, ue, so, tolls, refund=True)

    revenue = float((tolls * so).sum())
    assert (kept.cost >= kept.cost_before - 1e-6).all()  # tolls kept: nobody gains here
    assert float((refunded.refund * demand).sum()) == pytest.approx(revenue)
    # with the refund travellers pay, in total, the travel time of the optimum
    assert float((refunded.net_cost * demand).sum()) == pytest.approx(fa.total_time(links, so), rel=1e-2)
    summary = ff.fairness_summary(refunded)
    assert summary.mean_net_cost < summary.mean_cost_before


# --- the whole chain on an AequilibraE project ------------------------------


@pytest.fixture
def nine_node(tmp_path):
    source = PROJECT_ROOT / "ae-projects" / "nine-node"
    if not source.is_dir():
        pytest.skip("run `uv run create-ae-projects` first")
    from aequilibrae import Project

    shutil.copytree(source, tmp_path / "nine-node")
    project = Project()
    project.open(str(tmp_path / "nine-node"))
    yield project
    project.close()


def test_nine_node_project(nine_node):
    """Table 1 of Hearn and Ramana (1998), to the accuracy of the default gap."""
    matrix = "nine-node_demand"
    links, demand = fa.load_links(nine_node), fa.load_demand(nine_node, matrix)
    assert len(links) == 18 and demand.to_dict() == {(1, 3): 10, (1, 4): 20, (2, 3): 30, (2, 4): 40}

    fa.solve_assignment(nine_node, matrix, "ue", "UE")
    fa.solve_assignment(nine_node, matrix, "so", "SO")
    so = fa.link_flows(nine_node, "so")
    assert fa.total_time(links, fa.link_flows(nine_node, "ue")) == pytest.approx(2455.84, rel=1e-3)
    assert fa.total_time(links, so) == pytest.approx(2253.92, rel=1e-3)

    ft.solve_tolls(nine_node, matrix, "MSCP")
    ft.solve_tolls(nine_node, matrix, "MINSYS")
    assert fa.report_of(nine_node, "toll_mscp")["total_tolls"] == pytest.approx(1493.5, rel=0.01)
    assert fa.report_of(nine_node, "toll_minsys")["total_tolls"] == pytest.approx(887.6, rel=0.01)

    # results are replaced, not duplicated, when a step is repeated
    ft.solve_tolls(nine_node, matrix, "MINSYS")
    assert list(nine_node.results.list().table_name).count("toll_minsys") == 1

    # AequilibraE's own assignment with the MINSYS tolls gives the system optimum
    graph = fa.build_graph(nine_node, tolls=ft.load_tolls(nine_node, "toll_minsys"))
    assignment = fa.traffic_assignment(fa.traffic_class(nine_node, graph, matrix), "UE")
    fa.run_assignment(nine_node, assignment, "ae_tolled")
    assert np.abs(fa.link_flows(nine_node, "ae_tolled") - so).max() < 1.0

    outcome = ff.evaluate(nine_node, matrix, "so", "toll_minsys", refund=True)
    assert ff.fairness_summary(outcome).share_worse_off == 0.0
