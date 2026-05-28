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
        Simulates scanning lag by sampling different time-steps for each line
        within a single frame.

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
            radii = jnp.full((n_atoms,), 2.0)

        # Time per line in seconds (assuming constant scan speed)
        dt_line = 1.0 / (frames_per_second * h)
        
        # We'll use vmap for frames, and a nested logic for lines.
        # For simplicity in this first version, we map each line to the nearest 
        # integer time step in the trajectory. 
        # Future: Use jax.numpy.interp for sub-step resolution.

        def _scan_one_frame(t_start_idx: int) -> jax.Array:
            def _scan_one_line(y_idx: int) -> jax.Array:
                # Calculate time offset for this specific line
                # Clamp to avoid going out of trajectory bounds
                t_offset = jnp.round(y_idx * dt_line * frames_per_second).astype(jnp.int32)
                t_curr = jnp.minimum(t_start_idx + t_offset, t_steps - 1)
                
                # Use dynamic_slice to extract just one line from the grid
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

            # Collect all lines to form a frame
            return jax.vmap(_scan_one_line)(jnp.arange(h)).T

        return jax.vmap(_scan_one_frame)(jnp.arange(t_steps))
