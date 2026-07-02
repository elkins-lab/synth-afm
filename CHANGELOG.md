# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-07-02

### Added
- **Tutorials**: Added a comprehensive Jupyter Notebook tutorial (`tip_geometry.ipynb`).
- **Docs Integration**: Added `docs` optional dependency group to `pyproject.toml` for standardized documentation builds.

### Changed
- **Toolchain**: Replaced `black` with `ruff-format` in `.pre-commit-config.yaml` for faster, unified linting and formatting.
- **Documentation**: Updated GitHub Pages links to the new `elkins-lab` organization and standardized repository badges.
- **Documentation**: Clarified physical assumptions and mathematical boundaries in `kernels.py` and `compute_force_map`.

### Fixed
- **CI Pipelines**: Forced JavaScript actions to Node.js 24 to suppress deprecation warnings.
- **CI Pipelines**: Fixed Codecov reporting by enforcing XML coverage output in `pytest`.
- **Test Coverage**: Achieved 100% test coverage by adding tests for default PRNGKey generation and protein filtering.
- **Biophysics Loading**: Fixed the `biotite` amino acid filter logic and imported it properly as an optional argument in `io.py`.

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
