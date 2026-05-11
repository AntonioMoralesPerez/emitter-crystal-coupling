# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

Meep/MPB must be installed via conda before the package (not available on PyPI):
```bash
conda install -c conda-forge pymeep=1.30
pip install -e .          # editable install for development
pip install -e ".[dev]"   # includes pytest and ipykernel
```

## Running the example

```bash
python kimble.py
```

Requires precomputed MPB field data in `data/kimble/fields/`. If absent, run first:
```bash
cd miscellaneous && python mpb_field_computation.py
```

## Tests

No test suite exists yet (`pytest` is listed as a dev dependency but there are no test files).

## Architecture

The physics pipeline flows through three stages:

**1. Field computation** (`miscellaneous/mpb_field_computation.py`)  
MPB computes the photonic crystal eigenmodes on a Monkhorst–Pack k-grid at specified emitter positions. Results are serialized to `data/kimble/fields/{N_K}_meshgrid_at_{r_0}.pkl` — one file per (grid size, position) pair. All positions are computed in a single MPB run to share a consistent k-dependent gauge.

**2. Core library** (`emitter_crystal_coupling/`)  
- `coupling.py` — Computes `g_{n,k}(r_0)` (reciprocal-space) and `g_{n,R}(r_0)` (real-space via DFT over k-points), and LDOS via Lorentzian broadening.
- `hamiltonian.py` — Assembles the block-structured sparse Hamiltonian `H = [[H_e, H_int], [H_int†, H_b]]`, where `H_b = sqrt(Θ_TB)` (TB energies are frequencies², so the square root gives mode frequencies). Supports multiple emitters and a quadratic PML absorbing layer (makes H non-Hermitian).
- `mpb_fields.py` — Discrete inverse Bloch transform to get Wannier-like lattice amplitudes.
- `monkhorst_pack.py` — Monkhorst–Pack k-point generation.
- `entropy.py` — Binary entropy and quadrant-resolved Shannon entropy for directional emission.
- `utils.py` — I/O (pickle), geometry helpers, unit conversions, plotting utilities.
- `constants.py` — Physical constants (notably `K_CONST` for coupling prefactor, `LORENTZIAN_BROADENING`).

**3. Simulation script** (`kimble.py`)  
Top-level script using the library. All physical parameters are module-level constants at the top. Sections in order: dielectric function → band structure (TB vs MPB) → real-space coupling → LDOS → isofrequency contour → finite-lattice time evolution (`expm_multiply`) → PML absorbing layer → quadrant entropy → two-emitter probe simulation with concurrence. Figures saved to `figures/kimble/`.

**Units convention**: TB Hamiltonian stores energies as `ω²` in `(c/a)²`, so mode frequencies are `ω = sqrt(Θ)` in `c/a`. Emitter detuning `ω_0` is extracted at the X-point (saddle point with high LDOS). Time evolution uses `a/c` internally; inputs/outputs use `1/Γ₀`.

**Key design note**: `make_light_matter_Hamiltonian` accesses private PythTB attributes `_site_energies`, `_hoppings`, `_orb`, and `_lat`. Pin the PythTB version if these break across upgrades.

## Miscellaneous scripts

`miscellaneous/` contains standalone sweep scripts (PML parameter sweeps, probe orientation sweeps, polarization sweeps, etc.) that explore parameter space and save figures to `figures/miscellaneous/`. These are independent of each other and of `kimble.py`.
