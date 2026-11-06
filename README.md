# fair-tolling 

Experiments on fair tolling algorithms based on the work of Hearn and Ramana Congestion Tolling

## Dependencies
### AequilibraE
Follow the installation instructions on the AequilibraE website:
<https://www.aequilibrae.com/latest/home.html>
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
Create example ae-projects for testing (simple example and Sioux Falls)
```bash
uv run --overwrite create-ae-projects 
```