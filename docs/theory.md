# Science & Theory

## High-Speed Atomic Force Microscopy

High-Speed Atomic Force Microscopy (HS-AFM) is a scanning probe technique capable of imaging
individual protein molecules at near-physiological conditions, in real time. Unlike cryo-EM or
X-ray crystallography — which capture static or ensemble-averaged snapshots — HS-AFM movies can
directly observe **protein conformational dynamics** on timescales of milliseconds.

!!! note "Key Reference"
    Ando T. et al. (2011). *A high-speed atomic force microscope for studying biological macromolecules.* PNAS, 98(22):12468–72.

---

## The Physical Imaging Model

### Height Mapping

The fundamental observable in AFM is the **topographic height** $H(x, y)$ at each lateral position
$(x, y)$ of the sample:

$$H(x, y) = \max_{i} \left( z_i + \sqrt{r_i^2 - \left[(x - x_i)^2 + (y - y_i)^2\right]} \right)$$

where $i$ indexes over all atoms with position $(x_i, y_i, z_i)$ and van der Waals radius $r_i$.

This max-over-atoms is a **non-differentiable** operation. `synth-afm` approximates it with the
**Log-Sum-Exp** (LSE) soft maximum:

$$\text{logsumexp}_\beta(h_1, \ldots, h_N) = \beta \log \sum_i \exp\!\left(\frac{h_i}{\beta}\right)$$

As the smoothness parameter $\beta \to 0$, this converges to the exact max. This approximation is
smooth and fully differentiable through JAX's autograd system.

---

### Tip Dilation

A real AFM tip is not an ideal point probe — it has a finite radius $r_\text{tip}$, typically
10–30 nm for HS-AFM cantilevers. The image recorded is actually the **mathematical dilation** of the
sample surface by the tip shape (Villarrubia, 1997).

For a spherical tip of radius $r_\text{tip}$, the image height is:

$$H_\text{tip}(x, y) = \max_{i} \left( z_i + \sqrt{(r_i + r_\text{tip})^2 - d_{xy,i}^2} \right) - r_\text{tip}$$

where $d_{xy,i}^2 = (x - x_i)^2 + (y - y_i)^2$.

This correctly accounts for the broadening of surface features and is the default kernel in `synth-afm`.

!!! tip "Physical Interpretation"
    Tip dilation makes narrow features appear wider and rounded. For a protein with features on the
    1–5 nm scale, a 10 nm tip radius produces substantial broadening. Always report the tip radius
    used when publishing synthetic HS-AFM data.

---

### Scanning Lag

HS-AFM acquires images by raster-scanning a cantilever line-by-line. Each horizontal line is
recorded at a slightly different moment in time. If the protein is moving during a scan, the result
is a **spatially-sheared** image — different columns reflect different instantaneous conformations.

The time delay for column $y$ in a frame starting at time $t_0$ is:

$$\Delta t(y) = \frac{y}{f_\text{scan} \cdot W}$$

where $f_\text{scan}$ is the frame rate and $W$ is the image width in pixels. `synth-afm` models
this by indexing into a molecular dynamics trajectory using the correct time-offset for each column.

---

## van der Waals Radii

Atomic radii used for the height mapping are from the Bondi (1964) parameter set:

| Element | Radius (Å) |
|---------|-----------|
| H | 1.20 |
| C | 1.70 |
| N | 1.55 |
| O | 1.52 |
| S | 1.80 |
| P | 1.80 |

Atoms of unknown element fall back to a default radius of 1.70 Å (Carbon).

---

## Differentiability

Every operation in `synth-afm` — the LSE height map, tip dilation, scanning lag interpolation, and
noise addition — is written in pure JAX and supports `jax.grad`, `jax.jit`, and `jax.vmap`.

This makes `synth-afm` suitable as the **forward model** inside a gradient-based structure
determination loop:

$$\mathcal{L}(\mathbf{X}) = \|\mathcal{H}(\mathbf{X}) - \mathbf{I}_\text{exp}\|^2$$

$$\mathbf{X}^* = \arg\min_{\mathbf{X}} \mathcal{L}(\mathbf{X})$$

where $\mathcal{H}$ is the `AFMSimulator.scan` operator and $\mathbf{I}_\text{exp}$ is an
experimentally acquired HS-AFM frame.
