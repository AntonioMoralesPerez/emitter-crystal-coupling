# Atom–light coupling in a 2D photonic crystal

Companion code for the paper:

> *Two-dimensional photonic crystals for engineering atom–light interactions*
> Based on the framework of Coome et al. & Kimble, H. J., PNAS 116(26), 12743–12751 (2019).

## Contents

### `emitter_crystal_coupling/`
Core physics library imported as `mf` throughout the scripts. Provides:
- **`coupling.py`** — reciprocal- and real-space emitter–bath coupling $g_{n,k}(\mathbf{r}_0)$, $g_{n,\mathbf{R}}(\mathbf{r}_0)$, and LDOS computation.
- **`hamiltonian.py`** — construction of the full light–matter Hamiltonian.
- **`entropy.py`** — binary entropy and quadrant-resolved Shannon entropy.
- **`monkhorst_pack.py`** — Monkhorst–Pack $k$-point sampling.
- **`mpb_fields.py`** — discrete inverse Bloch transform of MPB fields.
- **`utils.py`** — I/O, geometry, unit conversions, and plotting helpers.
- **`constants.py`** — physical constants.

### `kimble.py`
Main demonstration script. Reproduces the key results of the paper:
1. Dielectric function of the photonic crystal.
2. Tight-binding vs. MPB band structure and isofrequency contour.
3. Real-space coupling $g_\mathbf{R}(\mathbf{r}_0)$ and LDOS at the emitter position.
4. Time evolution of emitter population and entanglement entropy.
5. Bath-mediated directional emission and concurrence for a two-emitter system.

Run with:
```bash
python kimble.py
```

Figures are saved to `figures/kimble/`. Precomputed MPB fields in `data/kimble/` are required — if absent, run `miscellaneous/mpb_field_computation.py` first.

### `miscellaneous/`
Auxiliary scripts for exploratory calculations:
- `mpb_field_computation.py` — computes and saves MPB fields at emitter positions.
- `mpb_fields_plot.py` — visualises the computed fields.
- `coupling_plot.py` — plots the coupling strength for different emitter positions.
- `polarization_sweep.py` — sweeps over emitter polarisation angles.
- `position_sweep.py` — sweeps over emitter positions in the unit cell.
- `time_evolution.py` — standalone time-evolution calculations.

## Installation

### 1. Install Meep (via conda)
Meep and MPB are not available on PyPI and must be installed through conda:
```bash
conda install -c conda-forge pymeep=1.30
```

### 2. Install the package
```bash
pip install .
```

If you plan to modify the source code, use the editable install instead:
```bash
pip install -e .
```
