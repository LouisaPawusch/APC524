# Cellular Automaton Library
Authors: Lara Tobias-Tarsh, Louisa Pawusch, Erin O'Neil
Course: APC524 - Fall 2025

## Overview
This repository contains a small Python Library for building and experimenting with **cellular automata (CA)**.
We provide a reusable `CellularAutomaton` class and a set of tools to define

- the geometry of the grid (2D or 3D)
- the neighborhood kernel (Moore or von Neumann)
- rule functions
- simple visualizations
- exemplary applications

Our example models are:
- **Conway's Game of Life (CGOL)** - baseline implementation
- **3D Game of Life** - extension to three spatial dimensions
- **SVIR Epidemic Model** - stochastic epidemiological CA with ensemble runs
- **Generating Music** - mapping CGOL dynamics to a simple musical sequence to generate a melody

They live in the examples/ directory.

## Installation

This project is managed with `uv` and installed as an editable package.

```bash
# From the repository root
uv pip install -e .
```

Python >=3.11 is recommended.


## 🎧 Optional: Audio and Video Support

The sonification module (`src/APC524/sonification/`) allows you to generate and
save audio-visual representations of the Game of Life evolution.
These features require additional dependencies (e.g. moviepy) that are **not installed by default**.

To install them, use the audio group:

```bash
uv pip install -e . --group audio
```

## Testing
To run the test suite:
```bash
pytest
```
