"""Create the example AequilibraE projects.

Projects are written to ``<repo>/ae-projects``:

* ``hearn-example`` - Simple Braess Network example with 4 nodes.
* ``sioux-falls``   - Sioux Falls imported from the GMNS files in
  ``<repo>/gmns-data/17_Sioux_Falls``, with centroids and an OD demand matrix.

Run with ``uv run create-ae-projects`` to create all examples, or name the ones
you want, e.g. ``uv run create-ae-projects sioux-falls``. Add ``--overwrite``
to rebuild existing projects.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from shapely.geometry import LineString

# ROOT = Path(__file__).resolve().parents[2]
# Paths are relative to the current working directory (run from the repo root)
AE_DIR = Path("ae-projects").resolve()
GMNS_DIR = Path("gmns-data").resolve()
SIOUX_FALLS_DIR = GMNS_DIR / "17_Sioux_Falls"

HEARN_PROJECT = AE_DIR / "hearn-example"
SIOUX_FALLS_PROJECT = AE_DIR / "sioux-falls"

# --------------------------------------------------------------------------
# Braess network data simple 
#
# Link cost uses BPR: t = t0 * (1 + alpha * (v / capacity) ** beta)
#   1->2 and 3->4: t = 1 + v/100   (t0=1, capacity=4000, alpha=40, beta=1)
#   1->3 and 2->4: t = 45          (alpha=0, constant)
#   2->3         : shortcut, t = 0.01
# Demand 1->4 = 4000. User equilibrium travel time increases with the shortcut
# --------------------------------------------------------------------------

# node_id -> (lon, lat), diamond layout: 1 = origin (west), 4 = destination (east)
BRAESS_NODES = {
    1: (0.000, 0.000),
    2: (0.010, 0.010),
    3: (0.010, -0.010),
    4: (0.020, 0.000),
}

# (a_node, b_node, free_flow_time, capacity, alpha, beta)
BRAESS_LINKS = [
    (1, 2, 1.0, 4000.0, 40.0, 1.0),
    (1, 3, 45.0, 4000.0, 0.0, 1.0),
    (2, 4, 45.0, 4000.0, 0.0, 1.0),
    (3, 4, 1.0, 4000.0, 40.0, 1.0),
    (2, 3, 0.01, 4000.0, 0.0, 1.0),
]

BRAESS_CENTROIDS = [1, 4]
BRAESS_DEMAND = {(1, 4): 4000.0}

# Sioux Falls project data is imported from GMNS files in ``gmns-data/17_Sioux_Falls``. The GMNS files are missing some fields AequilibraE expects, so we add them here.
# Extra GMNS link columns to keep as link fields in the Sioux Falls project
SIOUX_FALLS_EXTRA_LINK_FIELDS = {
    "vdf_fftt": "Free-flow travel time (min)",
    "vdf_alpha": "BPR alpha",
    "vdf_beta": "BPR beta",
    "vdf_toll": "Link toll",
    "ref_volume": "Reference volume",
}


def _new_project(path: Path, overwrite: bool) -> Project | None:
    """Create an empty project at ``path``; skip or replace it if it exists."""
    if path.exists():
        if not overwrite:
            print(f"{path} already exists, skipping (use --overwrite to rebuild)")
            return None
        shutil.rmtree(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    project = Project()
    project.new(str(path))
    return project


def _save_demand_matrix(project: Project, name: str, zone_ids, demand: np.ndarray) -> None:
    """Store a square OD matrix in the project's matrices folder in OMX format."""
    mat = AequilibraeMatrix()
    mat.create_empty(zones=len(zone_ids), matrix_names=["volume"], memory_only=True)
    mat.index[:] = np.asarray(zone_ids)
    mat.matrix["volume"][:, :] = demand
    record = project.matrices.new_record(name, f"{name}.omx", mat)
    record.description = f"OD demand for {name}"
    record.save()


def create_hearn_example(overwrite: bool = False) -> None:
    project = _new_project(HEARN_PROJECT, overwrite)
    if project is None:
        return

    try:
        links = project.network.links
        for field, description in [
            ("free_flow_time", "Free-flow travel time"),
            ("alpha", "BPR alpha"),
            ("beta", "BPR beta"),
        ]:
            links.fields.add(field, description, "NUMERIC")
        links.refresh_fields()

        for a, b, fftt, capacity, alpha, beta in BRAESS_LINKS:
            link = links.new()
            link.geometry = LineString([BRAESS_NODES[a], BRAESS_NODES[b]])
            link.direction = 1
            link.modes = "c"
            link.link_type = "default"
            link.name = f"{a}-{b}"
            link.capacity_ab = capacity
            link.free_flow_time = fftt
            link.alpha = alpha
            link.beta = beta
            link.save()

        nodes = project.network.nodes
        for node_id in BRAESS_CENTROIDS:
            node = nodes.get(node_id)
            node.is_centroid = 1
            node.save()

        demand = np.zeros((len(BRAESS_CENTROIDS), len(BRAESS_CENTROIDS)))
        position = {node_id: i for i, node_id in enumerate(BRAESS_CENTROIDS)}
        for (o, d), volume in BRAESS_DEMAND.items():
            demand[position[o], position[d]] = volume
        _save_demand_matrix(project, "braess_demand", BRAESS_CENTROIDS, demand)
    finally:
        project.close()

    print(f"Created {HEARN_PROJECT}")


def create_sioux_falls(overwrite: bool = False) -> None:
    project = _new_project(SIOUX_FALLS_PROJECT, overwrite)
    if project is None:
        return

    try:
        # Declare the extra GMNS columns so create_from_gmns imports them.
        par = Parameters()
        par.parameters["network"]["gmns"]["link"]["fields"].update(
            {
                name: {"description": description, "type": "numeric", "required": False}
                for name, description in SIOUX_FALLS_EXTRA_LINK_FIELDS.items()
            }
        )
        par.write_back()

        # The GMNS file lacks two columns AequilibraE relies on:
        #   directed      - required; every Sioux Falls link is one-way
        #   allowed_uses  - maps links to the built-in "car" mode (c)
        link_df = pd.read_csv(SIOUX_FALLS_DIR / "link.csv")
        link_df["directed"] = 1
        link_df["allowed_uses"] = "car"

        with tempfile.TemporaryDirectory() as tmp:
            link_file = Path(tmp) / "link.csv"
            link_df.to_csv(link_file, index=False)
            project.network.create_from_gmns(
                link_file_path=str(link_file),
                node_file_path=str(SIOUX_FALLS_DIR / "node.csv"),
                srid=4326,
            )

        # Every Sioux Falls node is a zone centroid.
        node_df = pd.read_csv(SIOUX_FALLS_DIR / "node.csv", encoding="utf-8-sig")
        zone_ids = sorted(node_df.loc[node_df["zone_id"].notna(), "node_id"].astype(int))
        with project.db_connection as conn:
            conn.executemany(
                "UPDATE nodes SET is_centroid = 1 WHERE node_id = ?",
                [(z,) for z in zone_ids],
            )
            conn.commit()

        demand_df = pd.read_csv(SIOUX_FALLS_DIR / "demand.csv")
        demand = (
            demand_df.pivot_table(index="o_zone_id", columns="d_zone_id", values="volume", aggfunc="sum")
            .reindex(index=zone_ids, columns=zone_ids, fill_value=0)
            .fillna(0)
            .to_numpy()
        )
        _save_demand_matrix(project, "sioux_falls_demand", zone_ids, demand)
    finally:
        project.close()

    print(f"Created {SIOUX_FALLS_PROJECT}")


EXAMPLES = {
    "hearn-example": create_hearn_example,
    "sioux-falls": create_sioux_falls,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the example AequilibraE projects.")
    parser.add_argument(
        "examples",
        nargs="*",
        choices=list(EXAMPLES),
        metavar="EXAMPLE",
        help=f"examples to create: {', '.join(EXAMPLES)} (default: all)",
    )
    parser.add_argument("--overwrite", action="store_true", help="delete and rebuild existing projects")
    args = parser.parse_args()

    for name in args.examples or EXAMPLES:
        EXAMPLES[name](overwrite=args.overwrite)


if __name__ == "__main__":
    main()
