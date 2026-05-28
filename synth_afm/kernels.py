import jax
import jax.numpy as jnp
from typing import cast


def simple_height_map_kernel(
    positions: jax.Array,
    radii: jax.Array,
    grid_x: jax.Array,
    grid_y: jax.Array,
    smoothness: float = 0.1,
) -> jax.Array:
    """
    A differentiable forward kernel to compute a molecular height map.
    Uses a Log-Sum-Exp approximation to smoothly combine atomic heights.

    Args:
        positions: (N, 3) array of atomic coordinates.
        radii: (N,) array of atomic van der Waals radii.
        grid_x: (H, W) grid of X coordinates.
        grid_y: (H, W) grid of Y coordinates.
        smoothness: Controls the sharpness of the surface (lower is sharper).

    Returns:
        (H, W) height map array.
    """
    # 1. Compute horizontal distance from each atom to each grid point
    atom_x = positions[:, 0, None, None]
    atom_y = positions[:, 1, None, None]
    atom_z = positions[:, 2, None, None]

    dx = grid_x - atom_x
    dy = grid_y - atom_y
    r_sq = dx**2 + dy**2

    # 2. Compute the theoretical height of the surface above each atom
    vdw_sq = radii[:, None, None] ** 2
    heights = atom_z + jnp.sqrt(jnp.maximum(vdw_sq - r_sq, 0.0))

    # Mask out areas where the atom doesn't contribute (r > r_vdw)
    mask = (r_sq < vdw_sq).astype(jnp.float32)
    masked_heights = (heights - 1000.0) * (1.0 - mask) + heights * mask

    # 3. Combine heights using Log-Sum-Exp for differentiability
    lse_height = smoothness * jax.nn.logsumexp(masked_heights / smoothness, axis=0)

    # Floor the height map at zero (the substrate surface)
    return cast(jax.Array, jnp.maximum(lse_height, 0.0))


def tip_sample_height_map(
    positions: jax.Array,
    radii: jax.Array,
    grid_x: jax.Array,
    grid_y: jax.Array,
    tip_radius: float = 20.0,
    smoothness: float = 0.5,
) -> jax.Array:
    """
    Computes a height map where the tip radius is explicitly modeled.
    This simulates 'tip dilation' where the image is the dilation of the
    sample surface by the tip shape.

    Technically, the height at grid point (x, y) is:
        H(x, y) = max_{atom i} (z_i + sqrt((r_i + r_tip)^2 - d_xy_i^2)) - r_tip
    """
    atom_x = positions[:, 0, None, None]
    atom_y = positions[:, 1, None, None]
    atom_z = positions[:, 2, None, None]

    dx = grid_x - atom_x
    dy = grid_y - atom_y
    r_sq = dx**2 + dy**2

    # Effective radius is the sum of atom radius and tip radius
    eff_radius = radii[:, None, None] + tip_radius
    eff_radius_sq = eff_radius**2

    # Theoretical height of the center of the tip sphere
    tip_center_z = atom_z + jnp.sqrt(jnp.maximum(eff_radius_sq - r_sq, 0.0))

    # Mask out areas where the tip doesn't touch the atom
    mask = (r_sq < eff_radius_sq).astype(jnp.float32)
    masked_heights = (tip_center_z - 1000.0) * (1.0 - mask) + tip_center_z * mask

    # Smooth max across atoms
    max_tip_z = smoothness * jax.nn.logsumexp(masked_heights / smoothness, axis=0)

    # The actual height measured is the tip center minus the tip radius
    measured_height = max_tip_z - tip_radius

    return cast(jax.Array, jnp.maximum(measured_height, 0.0))
