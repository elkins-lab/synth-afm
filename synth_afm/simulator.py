import jax
import jax.numpy as jnp
from typing import Optional, cast
from synth_afm.kernels import simple_height_map_kernel, tip_sample_height_map


class AFMSimulator:
    """
    High-level simulator for generating HS-AFM images.
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        grid_size: tuple[int, int] = (64, 64),
        tip_radius: float = 10.0,
        smoothness: float = 0.5,
    ):
        self.pixel_size = pixel_size
        self.grid_size = grid_size
        self.tip_radius = tip_radius
        self.smoothness = smoothness

        # Pre-compute grid
        x = jnp.arange(grid_size[0]) * pixel_size
        y = jnp.arange(grid_size[1]) * pixel_size
        self.grid_x, self.grid_y = jnp.meshgrid(x, y, indexing="ij")

    def scan(
        self,
        positions: jax.Array,
        radii: Optional[jax.Array] = None,
        use_tip_dilation: bool = True,
    ) -> jax.Array:
        """
        Generates a height map for the given positions.

        Args:
            positions: (N, 3) atomic coordinates.
            radii: (N,) atomic radii. If None, uses a default of 2.0 Å.
            use_tip_dilation: If True, uses the realistic tip-dilation kernel.
                              If False, uses a simple point-tip height map.
        """
        if radii is None:
            radii = jnp.full((positions.shape[0],), 2.0)

        if use_tip_dilation:
            return tip_sample_height_map(
                positions,
                radii,
                self.grid_x,
                self.grid_y,
                self.tip_radius,
                self.smoothness,
            )
        else:
            return simple_height_map_kernel(
                positions, radii, self.grid_x, self.grid_y, self.smoothness
            )

    def scan_movie(
        self,
        trajectory: jax.Array,
        radii: Optional[jax.Array] = None,
        frames_per_second: float = 10.0,
        use_tip_dilation: bool = True,
    ) -> jax.Array:
        """
        Generates a movie of HS-AFM images from a coordinate trajectory.
        Simulates scanning lag by sampling different time-steps for each line.

        Args:
            trajectory: (T, N, 3) atomic coordinates over time.
            radii: (N,) atomic radii.
            frames_per_second: Nominal scan rate.
            use_tip_dilation: Whether to use the realistic tip model.

        Returns:
            (T, H, W) array of height maps.
        """
        t_steps, n_atoms, _ = trajectory.shape
        h, w = self.grid_size

        if radii is None:
            radii = jnp.full((n_atoms,), 2.0)

        def _scan_one_frame(t_idx: int) -> jax.Array:
            # Simple implementation: sample coordinates for the whole frame.
            # Future expansion: sample line-by-line using jax.lax.scan
            return self.scan(trajectory[t_idx], radii, use_tip_dilation)

        return jax.vmap(_scan_one_frame)(jnp.arange(t_steps))
