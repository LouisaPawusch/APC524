# Changelog

All notable changes to this project will be documented in this file.

## [0.0.7] - 12-09-2025
Added an example of how the disease spread example could be implemented as an ensemble

### Added
**[MAJOR]** `examples/disease_parallel_example.py` : creates an ensemble version of the simulation and benchmarks it in parallel vs ensemble

### Changed
**[MAJOR]** `examples/display_3d_cgol.py` : refactored to be consistent with repo structure and use the CA class.
**[MINOR]** `src/APC524/visualization/visualization.py` : added function to do ensemble animation of disease

## [0.0.6] - 12-05-2025
Completed the implementation of the 3D example of CGOL and added lingering documentation to files in the examples folder.

### Added
- **[MAJOR]** `examples/display_3d_cgol.py`: creates random 3x3x3 grid and uses the rules of CGOL to evolve the game
- **[MAJOR]** `src/APC524/visualization/vizualization_3d`: creates the 3D animation based off of `visualization.py`

### Changed
- **[MINOR]** `src/APC524/solver/kernels.py`: added the 3D Moore kernel
- **[MINOR]** `src/APC524/visualization/__init__.py`: added the animte_automaton_3D function
- **[MINOR]** `examples/GCOL_example.py`: added documentation
- **[MINOR]** `examples/disease_example.py`: added documentation

## [0.0.5] - 11-03-2025
Added new tests, added a new rules set for disease spread.

### Added
* **[MAJOR]** `src/APC524/rules.py` : adjusted init function for CGOL to improve the usability, created a new init and rules function for a disease spread simulation
* **[MINOR]** `tests/test_rules.py` : added new tests for new rules, including test for stochastic behavior

## [0.0.4] - 11-01-2025
Made some substantial changes to the cellular automaton object to allow
for more flexibility in specifying rules functions.

### Added
* **[MAJOR]** `src/APC524/automaton.py` : added new class method `from_rules` so that a rules function can be used to specify and initialize grids
* **[MAJOR]** `src/APC524/automaton.py` : added new attr `states_dict` which allows more flexibility and transparency defining rules
* **[MAJOR]** `src/APC524/rules.py`: changed `CGOL_rules` to allow it to initialize the grid via a random number generator and a rules dictionary
as well as stepping
* **[MINOR]** `test/test_automaton.py` : updated to use the new initialization method

## [0.0.3] - 11-01-2025
This update resolves a conflict between the visualization branch and the
changes on the `erin_add` branch.

### Changed
* **[MINOR]** `src/APC524/solver/__init__.py` : all solver modules can now be imported correctly
* **[MINOR]** `src/APC524/visualization/__init__.py` : fixed a typo so that animation function can be imported correctly
* **[MINOR]** `src/APC524/__init__.py` : methods in all modules now import like a package would
* **[MINOR]** `examples/GCOL_example.py` : changed the path where the gif is saved, used random number generator to init grid
for Ruff compliance, removed link to `src` modules by fixing the `__init__.py` files

## [0.0.2] - 10-19-2025
This update adds CI to the project to ensure test compatibility going forward.
CI is run using pre-commit, nox and GitHub Actions and was tested locally with act.

### Added
* **[MAJOR]** `noxfile.py` : noxfile running our testing suite through pytest
* **[MAJOR]** `.github/workflows/ci.yml` : CI structure running a basic set of GitHub actions to ensure we are passing all tests upon push to all branches
* **[MAJOR]** `.pre-commit-config.yaml` : config file for the pre-commit tool. This is set up to autoformat all code with ruff, so this should be run before adding files to a commit to ensure compliance with PEP-8 standard

### Changed
* **[MAJOR]** `pyproject.toml` : changed to include parameters essential for CI and PEP-8 compliant formatting
* **[MINOR]** `/test/*` : removed shebangs from non-executable files and cleaned up for PEP-8 compliance
* **[MINOR]** `utils.py` : cleaned up implementation

## [0.0.2] - 10-18-2025

### Changed
- **[MINOR]** Made solver.py more modular

### Added
- **[MINOR]** `test_convolution.py` to test neighbor counts for a simple 2D CA grid

## [0.0.1] - 10-16-2025

### Changed

- **[MINOR]** Deleted unnecessary `__init__.py` in the top level directory
- **[MINOR]** Fixed the src file to match directory structure expected by hatchling

### Added

- **[MINOR]** A `uv.lock` file for uv developers
- **[MINOR]** A  modular, dataclass based structure for a 2D Cellular Automaton to the project in `src/APC524/solver.py`
- **[MINOR]** A rule set to simulate a basic version of Conway's Game of Life
- **[MINOR]** A simple tests to confirm everything is working as expected in test/`test_solver.py` - all tests run as expected locally
- **[MINOR]** Content to pyproject.toml so that softtware could be built with hatchling as a backend

## [0.0.2] - 10-23-2025

### Added

- **[MINOR]** Example as a randomly initialized GCOL in examples/
- **[MINOR]** Visualization as animation in src/APC524/visualization.py
