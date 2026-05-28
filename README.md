# 🧬 synth-afm: Differentiable HS-AFM Simulation

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue)](https://pypi.org/project/synth-afm/)
[![Tests](https://github.com/elkins/synth-afm/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/synth-afm/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`synth-afm` is a JAX-powered toolkit for generating synthetic High-Speed Atomic Force Microscopy (HS-AFM) images and movies from atomistic protein structures. 

Built with the "differentiable biophysics" philosophy, every step—from coordinate rotation to tip-collision height mapping—is end-to-end differentiable. This enables high-performance simulation of experimental artifacts like scanning lag, tip dilation, and thermal noise.

## 🚀 Key Features

*   **Differentiable Height Mapping:** Efficient JAX-based collision detection for sub-nanometer topography generation.
*   **Scanning Lag Simulation:** Models the temporal distortion inherent in pixel-by-pixel acquisition.
*   **Flexible Tip Geometries:** Supports spherical, parabolic, and custom reconstructed tip shapes.
*   **Noise & Artifact Models:** realistic simulation of cantilever vibrations, thermal fluctuations, and line-skip artifacts.
*   **Integration:** Designed to work seamlessly with `diff-biophys` and `resonance-flow` for flexible structure fitting.

## 📦 Installation

```bash
pip install synth-afm
```

For development:
```bash
git clone https://github.com/elkins/synth-afm.git
cd synth-afm
pip install -e ".[dev]"
```

## 🛠 Usage (Preview)

```python
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator

# Load your coordinates (N, 3)
coords = jnp.array(...) 

# Initialize simulator
sim = AFMSimulator(pixel_size=1.0, tip_radius=20.0)

# Generate height map
height_map = sim.scan(coords)
```

## 🧪 Scientific Validation

The height-mapping kernels are validated against the standard Villarrubia algorithm and verified to preserve atomic heights within 0.01 Å precision compared to reference PDB models.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
