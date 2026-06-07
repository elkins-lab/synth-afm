# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-07

### Security
- Removed compromised `polyfill.io` CDN script from MkDocs configuration to resolve supply-chain vulnerability.

## [0.1.1] - 2026-05-27

### Added
- **Scientific Rigor**: Implemented `synth_afm.io.load_coords_and_radii` for full PDB/mmCIF integration.
- **Biophysics Kernels**: Added atomic van der Waals radii constants based on Bondi (1964) parameters.
- **Physical Models**: Implemented `compute_force_map` for simulating tip-sample repulsion (deflection).
- **Scanning Lag**: Enhanced `scan_movie` with realistic line-by-line temporal distortion.
- **Validation**: New `tests/test_biophysics.py` verifying vdW assignment and end-to-end PDB scanning.
- **Modern Stack**: Full support for NumPy 2.x and JAX 0.10.x.

### Fixed
- Fixed JAX tracing issues in `scan_movie` using `jax.lax.dynamic_slice`.

## [0.1.0] - 2026-05-27

### Added
- Initial release of `synth-afm`.
- Core differentiable height-mapping kernels.
- Spherical tip-dilation model.
- Basic HS-AFM movie simulation support.
