import jax.numpy as jnp
import os
import pathlib
from synth_afm.io import load_coords_and_radii
from synth_afm.simulator import AFMSimulator
from synth_afm.constants import VDW_RADII


def test_vdw_radii_assignment() -> None:
    """Assigning radii to elements should match Bondi constants."""
    # Mock some atoms
    assert VDW_RADII["H"] == 1.20
    assert VDW_RADII["C"] == 1.70
    assert VDW_RADII["N"] == 1.55


def test_io_and_scan_consistency(tmp_path: pathlib.Path) -> None:
    """Loading a PDB and scanning should work end-to-end."""
    # Create a minimal mock PDB file
    pdb_content = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1       2.009   1.362   0.000  1.00  0.00           C\n"
        "END\n"
    )
    pdb_path = os.path.join(tmp_path, "test.pdb")
    with open(pdb_path, "w") as f:
        f.write(pdb_content)
    
    coords, radii = load_coords_and_radii(pdb_path)
    
    assert coords.shape == (3, 3)
    assert radii.shape == (3,)
    assert jnp.isclose(radii[0], VDW_RADII["N"])
    
    sim = AFMSimulator(grid_size=(10, 10))
    img = sim.scan(coords, radii)
    assert img.shape == (10, 10)
    assert jnp.max(img) > 0.0


def test_force_map_repulsion() -> None:
    """Force map should be zero if tip is above sample, positive if compressed."""
    sim = AFMSimulator(grid_size=(5, 5))
    coords = jnp.array([[2.5, 2.5, 5.0]])
    radii = jnp.array([2.0])
    
    # Grid points are at 0, 1, 2, 3, 4. Center is (2, 2)
    # Peak height is 5 + 2 = 7.0
    
    # 1. Tip high above sample (z=10.0)
    z_high = jnp.full((5, 5), 10.0)
    force_none = sim.compute_force_map(coords, radii, z_high)
    assert jnp.all(force_none == 0.0)
    
    # 2. Tip compressing sample (z=6.0 at peak)
    z_low = jnp.full((5, 5), 6.0)
    force_push = sim.compute_force_map(coords, radii, z_low)
    assert force_push[2, 2] > 0.0
