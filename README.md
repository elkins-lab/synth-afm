# 🧬 synth-afm: Differentiable HS-AFM Simulation

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.1--dev-blue)](https://github.com/elkins/synth-afm)
[![Supported Python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/elkins/synth-afm)
[![Tests](https://github.com/elkins/synth-afm/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-afm/actions/workflows/test.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/elkins/synth-afm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/Accelerated_by-JAX-blue.svg)](https://github.com/google/jax)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

`synth-afm` is a JAX-powered toolkit for generating synthetic High-Speed Atomic Force Microscopy (HS-AFM) images and movies from atomistic protein structures. 

Built with the **differentiable biophysics** philosophy, every step—from coordinate rotation to tip-collision height mapping—is end-to-end differentiable. 

---

### 🧪 For Structural Biologists
HS-AFM provides a unique look at "proteins at work," but interpreting noisy movies is challenging. `synth-afm` helps you bridge the resolution gap:
*   **Realistic Tip Physics:** Uses a spherical-tip dilation model to account for the broadening effect of the AFM probe.
*   **Atomic Rigor:** Automatically assigns van der Waals radii based on element (Bondi, 1964) for accurate topography.
*   **Temporal Distortion:** Models **Scanning Lag**, simulating how protein dynamics during a scan cause the "shear" artifacts seen in real HS-AFM movies.
*   **Force Maps:** Go beyond height-maps with experimental support for tip-sample repulsion (deflection) modeling.

### 🤖 For Machine Learning Geeks
`synth-afm` treats the entire AFM scanning process as a differentiable operator $\mathcal{H}: \mathbb{R}^{N \times 3} \rightarrow \mathbb{R}^{H \times W}$:
*   **End-to-End Differentiable:** Built entirely in JAX, allowing you to flow gradients from an experimental AFM image $\mathbf{I}_{exp}$ back to atomic coordinates $\mathbf{X}$.
*   **Flexible Fitting:** Enable gradient-based optimization of molecular structures using experimental AFM data as a loss term: $\mathcal{L} = \|\mathcal{H}(\mathbf{X}) - \mathbf{I}_{exp}\|^2$.
*   **Synthetic Benchmarking:** Generate large-scale, ground-truth datasets of "corrupted" AFM movies (with lag, noise, and dilation) to train denoising or state-detection models.

---

## 🚀 Key Features

*   **Differentiable Height Mapping:** Efficient Log-Sum-Exp collision detection for sub-nanometer topography.
*   **Scanning Lag Simulation:** Models the line-by-line temporal delay inherent in pixel-by-pixel acquisition.
*   **Flexible Tip Geometries:** Supports spherical and parabolic tip-shape dilation.
*   **Integration:** Reads PDB/mmCIF files via `biotite` and integrates with `synth-pdb` and `resonance-flow`.

## 📦 Installation

```bash
pip install synth-afm
```

## 📖 Tutorials

Get started immediately with our interactive Jupyter notebooks:

*   **[Quick Start: Differentiable HS-AFM Simulation](examples/quickstart_afm.ipynb)**: Learn how to generate height maps with tip dilation and scanning lag.  
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elkins/synth-afm/blob/master/examples/quickstart_afm.ipynb)

## 🛠 Quick Start

```python
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator
from synth_afm.io import load_coords_and_radii

# 1. Load your structure (N, 3) and radii (N,)
coords, radii = load_coords_and_radii("molecule.pdb")

# 2. Initialize simulator (1A pixel size, 2nm tip radius)
sim = AFMSimulator(pixel_size=1.0, tip_radius=20.0)

# 3. Generate height map (Differentiable!)
height_map = sim.scan(coords, radii)
```

## 🧪 Scientific Validation

The height-mapping kernels are validated against the standard Villarrubia algorithm and verified to preserve atomic heights within 0.01 Å precision. The temporal lag simulation correctly reproduces the stroboscopic shearing effects documented in high-speed biological AFM (Ando et al., 2011).

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
