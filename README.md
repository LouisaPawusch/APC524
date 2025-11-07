# APC524 Final Project
Authors: Lara Tobias-Tarsh, Louisa Pawusch, Erin O'Neil

## Description
This repository contains the code for our APC524 Project (Fall 2025), creating a universal
cellular automata library for use in scientific modelling. The most basic implementation
of this is Conway's Game of Life, which is implemented first.

## 🎧 Optional: Audio and Video Support

The sonification module (`src/APC524/sonification/`) allows you to generate and
save audio-visual representations of the Game of Life evolution.
These features require additional dependencies that are **not installed by default**.

To install them, run:

```bash
uv pip install -e . --group audio