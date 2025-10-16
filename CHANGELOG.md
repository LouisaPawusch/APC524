# Changelog

All notable changes to this project will be documented in this file.

## [0.0.1] - 10-16-2025

### Changed

- Deleted unnecessary __init__.py in the top level directory
- Fixed the src file to match directory structure expected by hatchling

### Added

- A uv.lock file for uv developers
- A  modular, dataclass based structure for a 2D Cellular Automaton to the project in src/APC524/solver.py
- A rule set to simulate a basic version of Conway's Game of Life
- A simple tests to confirm everything is working as expected in test/test_solver.py - all tests run as expected locally
- Content to pyproject.toml so that softtware could be built with hatchling as a backend
