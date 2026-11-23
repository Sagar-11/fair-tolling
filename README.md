# fair-tolling 

Experiments on fair tolling algorithms based on the work of Hearn and Ramana Congestion Tolling

## Dependencies
### AequilibraE
Follow the installation instructions on the AequilibraE website:
<https://www.aequilibrae.com/latest/home.html>
The Project uses a local editable install of AE at aequilibrae-src modify project toml to use prebuilt wheels.
### Gurobi
The project uses [Gurobi](https://www.gurobi.com/) through its Python API (`gurobipy`), which `uv sync` installs.
A valid Gurobi license is required, and the `gurobi_cl` command must be on your `PATH` for the license test.
Check the license with:
```bash
gurobi_cl --license
```
### uv
This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.
Install uv on macOS or Linux;
Then install the project and its dev dependencies:
```bash
uv sync
```

### Running tests

```bash
uv run pytest
```
Run a specific test file:
```bash
uv run pytest tests/test_project.py
```
## Directory tree
.
├── ae-projects : Aequilibrae projects, each folder is structured as recommended
├    by the Aequilibrae documentation
├── docs : Documentation files
├── experiments : Notebooks and scripts for exploration and analysis
├── gmns-data : Example networks in General Modeling Network Specification (GMNS) format
├── src : source 
├── tests : test suites 

## Scripts
Create the example ae-projects (Braess `hearn-example`, `nine-node` and `sioux-falls`)
```bash
uv run create-ae-projects --overwrite
```

## Source
`src/fair_tolling` works on an open AequilibraE project and saves into its results database:
- `assignment.py` : UE / SO assignment in link flows (AequilibraE or Gurobi), route flows from link flows
- `tolling.py` : first-best tolls of Hearn and Ramana, `solve_tolls(project, matrix_name, method)`
- `fairness.py` : cost of each O-D pair before and after tolls, and fairness measures

## Experiments
Run the notebooks in `experiments/` in this order (the first saves the AequilibraE runs the second reads):
1. `explore_ae_projects.ipynb` : the projects, AequilibraE assignment and tolls, route flows from link flows
2. `gurobi_expts.ipynb` : the assignment and toll models in Gurobi, compared with AequilibraE
3. `nine_node_tolls.ipynb` : Table 1 of Hearn and Ramana on the nine node network
4. `fairness_expts.ipynb` : fairness of the toll methods on the three networks (figures in `experiments/figures`)

The findings are summarised in `docs/fairness_report.pdf`.