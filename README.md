# Atom–light coupling in a 2D photonic crystal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20592589.svg)](https://doi.org/10.5281/zenodo.20592589)

A Python library for computing emitter–bath coupling, time evolution, and entanglement in 2D photonic crystal systems. `kimble.py` provides a worked example based on the framework of Kimble, H. J., PNAS 116(26), 12743–12751 (2019).

## Contents

### `emitter_crystal_coupling/`
Core physics library. Provides:
- **`coupling.py`** — reciprocal- and real-space emitter–bath coupling $g_{n,k}(\mathbf{r}_0)$, $g_{n,\mathbf{R}}(\mathbf{r}_0)$, and decay rate $\Gamma(\omega)$ computation.
- **`hamiltonian.py`** — construction of the full light–matter Hamiltonian.
- **`entropy.py`** — binary entropy and quadrant-resolved Shannon entropy.
- **`monkhorst_pack.py`** — Monkhorst–Pack $k$-point sampling.
- **`mpb_fields.py`** — discrete inverse Bloch transform of MPB fields.
- **`utils.py`** — I/O, geometry, unit conversions, and plotting helpers.
- **`constants.py`** — physical constants.

### `kimble.py`
Example script demonstrating the library on a 2D photonic crystal:
1. Dielectric function of the photonic crystal.
2. Tight-binding vs. MPB band structure and isofrequency contour.
3. Real-space coupling $g_\mathbf{R}(\mathbf{r}_0)$ and decay rate $\Gamma(\omega)$ at the emitter position.
4. Time evolution of emitter population and entanglement entropy.
5. Bath-mediated directional emission and concurrence for a two-emitter system.

Run with:
```bash
python kimble.py
```

Figures are saved to `figures/kimble/`. Precomputed MPB fields in `data/kimble/` are required — if absent, run `miscellaneous/mpb_field_computation.py` first.

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

## References
