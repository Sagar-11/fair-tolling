"""Create the example AequilibraE projects from GMNS files.

Each example is a folder in ``<repo>/gmns-data`` holding ``node.csv``, ``link.csv``
and ``demand.csv``. Projects are written to ``<repo>/ae-projects`` under the folder
name, with ``_`` replaced by ``-``:

* ``hearn-example`` - Simple Braess network example with 4 nodes.
* ``nine-node``     - Nine node network of Hearn and Ramana (1998), Figure 1.
* ``sioux-falls``   - Sioux Falls network with centroids and an OD demand matrix.

The links and nodes tables are filled directly instead of through AequilibraE's own
GMNS importer, which merges opposite one-way links into two-way links. GMNS columns
that match an AequilibraE field are stored as the GMNS equivalency in the project's
``parameters.yml``; every other column is added as an extra field and described there.

Minimum GMNS columns:

* ``node.csv``:   ``node_id``, ``x_coord``, ``y_coord``, plus ``zone_id`` on zone nodes
* ``link.csv``:   ``link_id``, ``from_node_id``, ``to_node_id``
* ``demand.csv``: ``o_zone_id``, ``d_zone_id``, ``volume``

Run with ``uv run create-ae-projects`` to create all examples, or name the ones
you want, e.g. ``uv run create-ae-projects sioux-falls``. Add ``--overwrite``
to rebuild existing projects.
"""

import argparse
import shutil
import string
from pathlib import Path

import numpy as np
import pandas as pd
from aequilibrae import Project
from aequilibrae.matrix import AequilibraeMatrix
from aequilibrae.parameters import Parameters
from shapely import wkt
from shapely.geometry import LineString

# Paths are relative to the current working directory (run from the repo root)
AE_DIR = Path("ae-projects").resolve()
GMNS_DIR = Path("gmns-data").resolve()
HEARN_DIR = GMNS_DIR / "hearn_example"
NINE_NODE_DIR = GMNS_DIR / "nine_node"
SIOUX_FALLS_DIR = GMNS_DIR / "sioux_falls"

# AequilibraE link field -> GMNS link columns that can supply it; the first one found
# in link.csv is used. ``capacity`` and ``speed`` fill the ``_ab``/``_ba`` fields.
LINK_EQUIVALENCY = {
    "link_id": ("link_id",),
    "a_node": ("from_node_id",),
    "b_node": ("to_node_id",),
    "direction": ("directed", "dir_flag"),
    "modes": ("allowed_uses",),
    "link_type": ("facility_type", "link_type"),
    "name": ("name",),
    "capacity": ("capacity",),
    "speed": ("free_speed",),
    "geometry": ("geometry",),
}
REQUIRED_LINK_FIELDS = ("link_id", "a_node", "b_node")

# AequilibraE node field -> GMNS node columns. Nodes with a zone_id are centroids.
NODE_EQUIVALENCY = {
    "node_id": ("node_id",),
    "is_centroid": ("zone_id",),
}
NODE_COORDINATES = ("x_coord", "y_coord")

# Values for AequilibraE link fields that link.csv leaves out
DEFAULT_DIRECTION = 1  # one-way, from a_node to b_node
DEFAULT_MODES = "c"  # car
DEFAULT_LINK_TYPE = "default"

# Descriptions for extra GMNS fields. Columns not listed here use the description in
# the GMNS defaults of parameters.yml, if any.
FIELD_DESCRIPTIONS = {
    "vdf_fftt": "Free-flow travel time (min)",
    "vdf_alpha": "BPR alpha",
    "vdf_beta": "BPR beta",
    "vdf_toll": "Link toll",
    "vdf_length_mi": "Length (mi)",
    "vdf_free_speed_mph": "Free speed (mph)",
    "ref_volume": "Reference volume",
    "obs_volume": "Observed volume",
    "district_id": "District id",
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


def _match_columns(equivalency: dict, columns) -> dict:
    """Map each AequilibraE field to the first of its GMNS columns found in ``columns``."""
    matched = {}
    for ae_field, gmns_columns in equivalency.items():
        found = [col for col in gmns_columns if col in columns]
        if found:
            matched[ae_field] = found[0]
    return matched


def _field_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series) or pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    return "text"


def _extra_fields(df: pd.DataFrame, used_columns, gmns_defaults: dict, table: str) -> dict:
    """Parameter entries for every GMNS column not used by the equivalency."""
    fields = {}
    for col in df.columns:
        if col in used_columns:
            continue
        description = FIELD_DESCRIPTIONS.get(col) or gmns_defaults.get(col, {}).get("description")
        fields[col] = {
            "description": description or f"GMNS {table} field {col}",
            "type": _field_type(df[col]),
            "required": False,
        }
    return fields


def _add_table_fields(table, fields: dict) -> None:
    """Add the extra fields as columns of an AequilibraE links or nodes table."""
    existing = set(table.fields.all_fields())
    for field, spec in fields.items():
        if field in existing:
            raise ValueError(f"GMNS column '{field}' clashes with an existing AequilibraE field")
        table.fields.add(field, spec["description"], spec["type"].upper())


def _sql_rows(df: pd.DataFrame) -> list[list]:
    """Rows of ``df`` as plain Python values, with missing values as None."""
    return df.astype(object).where(df.notna(), None).to_numpy().tolist()


def _insert(project: Project, table: str, df: pd.DataFrame, geometry_sql: str) -> None:
    """Insert ``df`` into ``table``; its last columns feed the ``geometry_sql`` expression."""
    n_geometry = geometry_sql.count("?")
    columns = list(df.columns[: len(df.columns) - n_geometry])
    placeholders = ", ".join("?" * len(columns))
    sql = f"INSERT INTO {table} ({', '.join(columns)}, geometry) VALUES ({placeholders}, {geometry_sql})"
    with project.db_connection as conn:
        conn.executemany(sql, _sql_rows(df))


def _to_direction(value) -> int:
    """GMNS ``directed`` (true/false) or ``dir_flag`` (1/0/-1) as an AequilibraE direction."""
    if isinstance(value, str):
        value = {"true": 1, "false": 0}.get(value.strip().lower(), value)
    return int(value)


def _to_modes(project: Project, uses: pd.Series) -> pd.Series:
    """GMNS ``allowed_uses`` (comma separated mode names or ids) as AequilibraE mode ids."""
    with project.db_connection as conn:
        mode_ids = dict(conn.execute("SELECT mode_name, mode_id FROM modes").fetchall())
    mode_ids.update({mode_id: mode_id for mode_id in list(mode_ids.values())})

    def convert(value) -> str:
        if pd.isna(value) or str(value).strip() == "":
            return DEFAULT_MODES
        names = [name.strip() for name in str(value).split(",")]
        unknown = [name for name in names if name not in mode_ids]
        if unknown:
            raise ValueError(f"Unknown modes in allowed_uses: {', '.join(unknown)}")
        return "".join(sorted({mode_ids[name] for name in names}))

    return uses.map(convert)


def _to_link_types(project: Project, types: pd.Series) -> pd.Series:
    """GMNS link types as AequilibraE link types, adding any that are new."""
    types = types.map(lambda t: DEFAULT_LINK_TYPE if pd.isna(t) else str(t))
    with project.db_connection as conn:
        existing = dict(conn.execute("SELECT link_type, link_type_id FROM link_types").fetchall())
        free_ids = [c for c in string.ascii_letters if c not in existing.values()]
        for link_type in types.unique():
            if link_type not in existing:
                conn.execute(
                    "INSERT INTO link_types (link_type, link_type_id, description) VALUES (?, ?, ?)",
                    (link_type, free_ids.pop(0), "Link type from GMNS link.csv"),
                )
    return types


def _link_geometry(a_xy, b_xy, gmns_wkt) -> bytes:
    """Link line as WKB, with its ends on the node points so AequilibraE attaches them."""
    coords = [a_xy, b_xy]
    if isinstance(gmns_wkt, str) and gmns_wkt.strip():
        coords = list(wkt.loads(gmns_wkt).coords)
        coords[0], coords[-1] = a_xy, b_xy
    return LineString(coords).wkb


def _add_nodes(project: Project, node_df: pd.DataFrame, equivalency: dict, extra_fields: dict) -> dict:
    """Fill the nodes table from GMNS nodes; return node_id -> (x, y).

    Nodes go in before links so links attach to them. Otherwise AequilibraE creates
    the nodes itself and numbers them in its own order.
    """
    _add_table_fields(project.network.nodes, extra_fields)

    out = pd.DataFrame({"node_id": node_df[equivalency["node_id"]].astype(int)})
    if "is_centroid" in equivalency:
        out["is_centroid"] = node_df[equivalency["is_centroid"]].notna().astype(int)
    else:
        out["is_centroid"] = 0
    for field in extra_fields:
        out[field] = node_df[field]
    out["x"] = node_df["x_coord"].astype(float)
    out["y"] = node_df["y_coord"].astype(float)

    _insert(project, "nodes", out, "MakePoint(?, ?, 4326)")
    return dict(zip(out["node_id"], zip(out["x"], out["y"])))


def _add_links(project: Project, link_df: pd.DataFrame, equivalency: dict, extra_fields: dict, nodes: dict) -> None:
    """Fill the links table from GMNS links, one AequilibraE link per GMNS link."""
    _add_table_fields(project.network.links, extra_fields)

    def gmns(field: str) -> pd.Series:
        return link_df[equivalency[field]]

    out = pd.DataFrame(
        {
            "link_id": gmns("link_id").astype(int),
            "a_node": gmns("a_node").astype(int),
            "b_node": gmns("b_node").astype(int),
        }
    )
    unknown = set(out["a_node"]).union(out["b_node"]).difference(nodes)
    if unknown:
        raise ValueError(f"Links use nodes missing from node.csv: {sorted(unknown)}")

    if "direction" in equivalency:
        out["direction"] = gmns("direction").fillna(DEFAULT_DIRECTION).map(_to_direction)
    else:
        out["direction"] = DEFAULT_DIRECTION
    out["modes"] = _to_modes(project, gmns("modes")) if "modes" in equivalency else DEFAULT_MODES
    if "link_type" in equivalency:
        out["link_type"] = _to_link_types(project, gmns("link_type"))
    else:
        out["link_type"] = DEFAULT_LINK_TYPE
    if "name" in equivalency:
        out["name"] = gmns("name")

    # Direction 1 is a_node -> b_node only, -1 is b_node -> a_node only, 0 is both ways
    for field in ("capacity", "speed"):
        if field in equivalency:
            out[f"{field}_ab"] = gmns(field).where(out["direction"] >= 0)
            out[f"{field}_ba"] = gmns(field).where(out["direction"] <= 0)

    for field in extra_fields:
        out[field] = link_df[field]

    gmns_wkt = gmns("geometry") if "geometry" in equivalency else [None] * len(out)
    out["geometry"] = [
        _link_geometry(nodes[a], nodes[b], line) for a, b, line in zip(out["a_node"], out["b_node"], gmns_wkt)
    ]

    _insert(project, "links", out, "GeomFromWKB(?, 4326)")


def _save_demand_matrix(project: Project, name: str, zone_ids, demand: np.ndarray) -> None:
    """Store a square OD matrix in the project's matrices folder in OMX format."""
    mat = AequilibraeMatrix()
    mat.create_empty(zones=len(zone_ids), matrix_names=["volume"], memory_only=True)
    mat.index[:] = np.asarray(zone_ids)
    mat.matrix["volume"][:, :] = demand
    record = project.matrices.new_record(name, f"{name}.omx", mat)
    record.description = f"OD demand for {name}"
    record.save()


def create_from_gmns(gmns_dir: Path, overwrite: bool = False) -> None:
    """Create a new AequilibraE project from the GMNS files in ``gmns_dir``.

    The project is written to ``ae-projects/<gmns_dir name, with "_" replaced by "-">``.
    """
    name = gmns_dir.name.replace("_", "-")
    project_path = AE_DIR / name

    node_df = pd.read_csv(gmns_dir / "node.csv", encoding="utf-8-sig")
    link_df = pd.read_csv(gmns_dir / "link.csv", encoding="utf-8-sig")
    demand_df = pd.read_csv(gmns_dir / "demand.csv", encoding="utf-8-sig")

    link_equivalency = _match_columns(LINK_EQUIVALENCY, link_df.columns)
    node_equivalency = _match_columns(NODE_EQUIVALENCY, node_df.columns)
    missing = [LINK_EQUIVALENCY[f][0] for f in REQUIRED_LINK_FIELDS if f not in link_equivalency]
    missing += [col for col in ("node_id", *NODE_COORDINATES) if col not in node_df.columns]
    if "is_centroid" not in node_equivalency:
        missing.append("zone_id")
    if missing:
        raise ValueError(f"GMNS files in {gmns_dir} are missing required columns: {', '.join(missing)}")

    project = _new_project(project_path, overwrite)
    if project is None:
        return

    try:
        # Store the GMNS equivalency (only the columns found) and the extra fields in
        # parameters.yml
        par = Parameters()
        gmns = par.parameters["network"]["gmns"]
        extra_link_fields = _extra_fields(link_df, set(link_equivalency.values()), gmns["link"]["fields"], "link")
        extra_node_fields = _extra_fields(
            node_df, {*node_equivalency.values(), *NODE_COORDINATES}, gmns["node"]["fields"], "node"
        )
        gmns["link"]["equivalency"] = link_equivalency
        gmns["link"]["fields"].update(extra_link_fields)
        gmns["node"]["equivalency"] = node_equivalency
        gmns["node"]["fields"].update(extra_node_fields)
        par.write_back()

        nodes = _add_nodes(project, node_df, node_equivalency, extra_node_fields)
        _add_links(project, link_df, link_equivalency, extra_link_fields, nodes)

        # OD demand matrix indexed by the centroid node ids
        zone_col = node_equivalency["is_centroid"]
        zones = node_df[node_df[zone_col].notna()]
        zone_to_node = dict(zip(zones[zone_col].astype(int), zones["node_id"].astype(int)))
        zone_ids = sorted(zone_to_node.values())
        demand = (
            demand_df.assign(
                o_node=demand_df["o_zone_id"].map(zone_to_node),
                d_node=demand_df["d_zone_id"].map(zone_to_node),
            )
            .pivot_table(index="o_node", columns="d_node", values="volume", aggfunc="sum")
            .reindex(index=zone_ids, columns=zone_ids, fill_value=0)
            .fillna(0)
            .to_numpy()
        )
        _save_demand_matrix(project, f"{name}_demand", zone_ids, demand)

        project.logger.info(f"Created project {project_path} from GMNS files in {gmns_dir}")
    finally:
        project.close()

    print(f"Created {project_path}")


EXAMPLES = {
    "hearn-example": HEARN_DIR,
    "nine-node": NINE_NODE_DIR,
    "sioux-falls": SIOUX_FALLS_DIR,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the example AequilibraE projects from GMNS files.")
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
        create_from_gmns(EXAMPLES[name], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
