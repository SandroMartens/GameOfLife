# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two independent, self-contained Game of Life implementations. They share no code or imports — each is a standalone script you run directly. There is no shared package, no `src/` layout, no tests.

- **GameOfLife.py** — Classic Conway's GoL. Discrete 0/1 grid, `scipy.signal.convolve2d` (wrap boundary) for neighbor counts, pygame for rendering. Configurable survive/birth rule sets (default `([2,3],[3])`).
- **intetactive_gol.py** — SmoothLife (continuous generalization of GoL, Rafler 2011). The simulation (`SmoothGameOfLife`) is decoupled from rendering. PySide6/Qt UI: `AnimatedLabel` drives the sim on a `QTimer` and renders via `QImage`/`QPixmap`; `AnimationWidget` assembles parameter sliders (from `SLIDER_CONFIGS`), a mode selector, and a reset button.

`1111.1567v2.pdf` is Rafler's SmoothLife paper — the source for the math in `intetactive_gol.py` (inner/outer "filling" convolutions `m`/`n`, sigmoid interval functions, the `dt`-stepped update in eq. 8). The docstrings in `intetactive_gol.py` reference the paper's equation numbers directly.

`sdg.py`, `test.ipynb`, and `smoothlife` exist on disk but are gitignored — `smoothlife` is a third-party reference port (see its own header comment), not maintained code. Don't treat them as part of the active codebase.

## Commands

Dependency management is via `uv` (`uv.lock` present, requires Python >=3.14).

```
uv sync                          # install dependencies
uv run python intetactive_gol.py # run the Qt SmoothLife UI
uv run python GameOfLife.py      # run classic Conway's GoL
```

No test suite, no lint/format config exists in this repo (no ruff/black/mypy config despite a stray `.mypy_cache`) — don't assume tooling that isn't configured.

## Architecture notes

- **Discrete GoL** (`GameOfLife.py`): integer neighbor-sum convolution, then `np.isin` against the rule lists to decide survival/birth.
- **SmoothLife** (`intetactive_gol.py`): state is a continuous array in `[0, 1]`. Each step:
  1. Two Gaussian convolutions approximate the disk integral (`cell_sums`, radius = `cell_size`) and the annulus integral (`neighbor_sums`, radius = `3*cell_size`) from the paper.
  2. Sigmoid-based interval functions (`birth`/`survival`) give smooth 0–1 membership instead of hard thresholds.
  3. An "aliveness" sigmoid on `cell_sums` mixes birth and survival into `s(n, m)`.
  4. `mode` (1/2/3) selects which delta formula integrates `s(n, m)` into the next state via `dt` — mode 1 (exponential relaxation toward `s`) is the default and most stable; mode 2 is the paper's literal formula and can diverge.
- Slider parameter names in `SLIDER_CONFIGS` map 1:1 to `SmoothGameOfLife` attribute names via `setattr` — adding a new tunable parameter means adding both the `__init__` arg and a `SliderConfig` entry.
