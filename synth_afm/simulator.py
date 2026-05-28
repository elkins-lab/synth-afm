import jax
import jax.numpy as jnp
from typing import Optional, cast
from synth_afm.kernels import simple_height_map_kernel, tip_sample_height_map


class AFMSimulator:
    """
    High-level simulator for generating HS-AFM images and movies.
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        grid_size: tuple[int, int] = (64, 64),
        tip_radius: float = 20.0,
        smoothness: float = 0.5,
    ):
        """
        Initialize the AFM simulator.

        Args:
            pixel_size: Size of a single pixel in Angstroms.
            grid_size: (Height, Width) of the resulting image.
            tip_radius: Radius of the AFM tip in Angstroms.
            smoothness: Smoothness of the Log-Sum-Exp max approximation.
        """
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
        Generates a height map for the given atomic positions.

        Args:
            positions: (N, 3) atomic coordinates.
            radii: (N,) atomic radii. If None, uses default 1.7A (Carbon).
            use_tip_dilation: If True, uses the realistic tip-dilation kernel.
        """
        if radii is None:
            radii = jnp.full((positions.shape[0],), 1.7)

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
            frames_per_second: Nominal scan rate (FPS).
            use_tip_dilation: Whether to use the realistic tip model.

        Returns:
            (T, H, W) array of height maps.
        """
        t_steps, n_atoms, _ = trajectory.shape
        h, w = self.grid_size

        if radii is None:
            radii = jnp.full((n_atoms,), 1.7)

        # Time per line in seconds (assuming constant scan speed)
        dt_line = 1.0 / (frames_per_second * h)

        def _scan_one_frame(t_start_idx: int) -> jax.Array:
            def _scan_one_line(y_idx: int) -> jax.Array:
                # Calculate time offset for this specific line
                t_offset = jnp.round(y_idx * dt_line * frames_per_second).astype(jnp.int32)
                t_curr = jnp.minimum(t_start_idx + t_offset, t_steps - 1)
                
                # Extract line grid
                line_x = jax.lax.dynamic_slice(self.grid_x, (0, y_idx), (h, 1))
                line_y = jax.lax.dynamic_slice(self.grid_y, (0, y_idx), (h, 1))
                
                if use_tip_dilation:
                    res = tip_sample_height_map(
                        trajectory[t_curr], radii, line_x, line_y, self.tip_radius, self.smoothness
                    )
                else:
                    res = simple_height_map_kernel(
                        trajectory[t_curr], radii, line_x, line_y, self.smoothness
                    )
                return res[:, 0]

            return jax.vmap(_scan_one_line)(jnp.arange(h)).T

        return jax.vmap(_scan_one_frame)(jnp.arange(t_steps))

    def compute_force_map(
        self,
        positions: jax.Array,
        radii: jax.Array,
        z_heights: jax.Array,
        k_cantilever: float = 0.1,
    ) -> jax.Array:
        """
        Experimental: Computes a force map (deflection) based on tip-sample 
        repulsion. Very simplified Hookean model.
        """
        sample_height = self.scan(positions, radii)
        compression = jnp.maximum(sample_height - z_heights, 0.0)
        return cast(jax.Array, k_cantilever * compression)
