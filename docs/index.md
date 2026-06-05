# 🔬 synth-afm

**Differentiable HS-AFM Simulation for Protein Structures**

`synth-afm` is a JAX-powered toolkit for generating synthetic **High-Speed Atomic Force Microscopy (HS-AFM)** images and movies from atomistic protein structures.

Built with the **differentiable biophysics** philosophy, every step — from coordinate rotation to tip-collision height mapping — is fully end-to-end differentiable. This means you can flow gradients from experimental AFM images back to atomic coordinates for structure refinement.

---

## Why synth-afm?

HS-AFM provides a unique, real-time window into "proteins at work." Interpreting noisy movies requires understanding the forward physics of the imaging process. `synth-afm` gives you that physics model — and makes it differentiable.

### For Structural Biologists

- **Realistic Tip Physics**: Spherical-tip dilation model accounts for probe broadening.
- **Atomic Rigor**: Assigns van der Waals radii per-element (Bondi, 1964) for accurate topography.
- **Temporal Distortion**: Models **Scanning Lag**, simulating how protein dynamics during a scan cause the characteristic "shear" artifacts seen in real HS-AFM movies.
- **Force Maps**: Experimental tip-sample repulsion (deflection) modeling beyond height-maps.

### For Machine Learning Researchers

`synth-afm` treats the entire AFM scanning process as a differentiable operator $\mathcal{H}: \mathbb{R}^{N \times 3} \rightarrow \mathbb{R}^{H \times W}$:

- **End-to-End Differentiable**: Built entirely in JAX — flow gradients from an experimental image $\mathbf{I}_{exp}$ back to atomic coordinates $\mathbf{X}$.
- **Flexible Fitting**: Enable gradient-based optimization using experimental AFM data as a loss: $\mathcal{L} = \|\mathcal{H}(\mathbf{X}) - \mathbf{I}_{exp}\|^2$
- **Synthetic Benchmarking**: Generate large-scale, ground-truth datasets of corrupted AFM movies to train denoising or state-detection models.

---

## Key Features

- **Differentiable Height Mapping**: Log-Sum-Exp collision detection for sub-nanometer topography
- **Physical Realism**: Cantilever noise and substrate tilt simulation
- **Scanning Lag Simulation**: Line-by-line temporal delay modelling
- **Memory Efficiency**: `jax.lax.scan` for constant-memory trajectory simulation
- **Flexible Tip Geometries**: Spherical tip-shape dilation
- **Integration**: Reads PDB/mmCIF via `biotite`; integrates with `synth-pdb` and `synth-dynamics`

---

## Quick Start

```python
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator
from synth_afm.io import load_coords_and_radii

# 1. Load your structure (N, 3) and radii (N,)
coords, radii = load_coords_and_radii("molecule.pdb")

# 2. Initialize simulator (1Å pixel size, 2nm tip radius, 0.5Å noise, slight tilt)
sim = AFMSimulator(
    pixel_size=1.0,
    tip_radius=20.0,
    noise_level=0.5,
    substrate_tilt=(0.01, 0.0)
)

# 3. Generate height map (fully differentiable!)
height_map = sim.scan(coords, radii)
```

---

## Next Steps

- **[Getting Started](getting_started.md)**: Installation and your first simulation
- **[Science & Theory](theory.md)**: The physics behind AFM imaging and tip dilation
- **[API Reference](api/index.md)**: Full Python API documentation

---

## Related Projects

`synth-afm` is part of a broader ecosystem for differentiable biophysics data generation:

| Project | Purpose |
|---|---|
| [synth-pdb](https://elkins.github.io/synth-pdb) | Foundation: Realistic protein structure generation |
| [synth-nmr](https://elkins.github.io/synth-nmr) | NMR observables (NOE, RDC, chemical shifts) |
| [synth-saxs](https://elkins.github.io/synth-saxs) | SAXS profile simulation |
| [synth-cryo-em](https://elkins.github.io/synth-cryo-em) | Cryo-EM density map generation |
| [synth-dynamics](https://elkins.github.io/synth-dynamics) | ANM/Langevin dynamics for conformational ensembles |
| [diff-biophys](https://elkins.github.io/diff-biophys) | Differentiable JAX kernels for all biophysics |
