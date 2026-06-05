# Getting Started

## Installation

Install `synth-afm` from PyPI:

```bash
pip install synth-afm
```

For development (with test and linting tools):

```bash
git clone https://github.com/elkins/synth-afm
cd synth-afm
pip install -e ".[dev]"
```

### Requirements

- Python 3.10, 3.11, or 3.12
- JAX + jaxlib
- NumPy ≥ 2.0
- biotite ≥ 0.35 (for PDB/mmCIF I/O)
- scipy ≥ 1.10

!!! tip "GPU / TPU Acceleration"
    `synth-afm` runs on CPU by default. To enable GPU/TPU support, install the
    appropriate `jaxlib` build for your hardware. See the
    [JAX installation guide](https://github.com/google/jax#installation).

---

## Your First Simulation

### From a PDB file

```python
from synth_afm.simulator import AFMSimulator
from synth_afm.io import load_coords_and_radii
import matplotlib.pyplot as plt

# Load a structure
coords, radii = load_coords_and_radii("1UBQ.pdb")

# Create simulator: 64x64 image, 1Å pixels, 2nm tip radius
sim = AFMSimulator(
    pixel_size=1.0,
    grid_size=(64, 64),
    tip_radius=20.0,
    noise_level=0.3,
)

# Generate height map
height_map = sim.scan(coords, radii)

# Visualise
plt.imshow(height_map, cmap="afmhot", origin="lower")
plt.colorbar(label="Height (Å)")
plt.title("Synthetic HS-AFM — Ubiquitin")
plt.show()
```

### Simulating an HS-AFM Movie

```python
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator

# trajectory shape: (T, N, 3) — T frames, N atoms, XYZ
sim = AFMSimulator(pixel_size=1.0, grid_size=(64, 64), tip_radius=20.0)

movie = sim.scan_movie(
    trajectory,          # (T, N, 3) array
    radii=radii,         # (N,) atom radii
    frames_per_second=10.0,
)
# movie.shape == (T, 64, 64)
```

### Without Tip Dilation (Faster, Less Physical)

For quick prototyping, you can disable tip dilation and use the bare height-map kernel:

```python
height_map = sim.scan(coords, radii, use_tip_dilation=False)
```

---

## Examples

See the [`examples/`](https://github.com/elkins/synth-afm/tree/master/examples) directory in the repository:

- **`quickstart_afm.ipynb`**: Quick start notebook — height maps with tip dilation and scanning lag.
- **`simulate_ubiquitin_afm.py`**: End-to-end script simulating ubiquitin from a PDB file.
