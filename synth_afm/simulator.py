import jax
import jax.numpy as jnp
from typing import Optional, cast, Tuple
from synth_afm.kernels import simple_height_map_kernel, tip_sample_height_map


class AFMSimulator:
    """
    High-level simulator for generating HS-AFM images and movies.
    """

    def __init__(
        self,
        pixel_size: float = 1.0,
        grid_size: Tuple[int, int] = (64, 64),
        tip_radius: float = 20.0,
        smoothness: float = 0.5,
        noise_level: float = 0.0,
        substrate_tilt: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Initialize the AFM simulator.

        Args:
            pixel_size: Size of a single pixel in Angstroms.
            grid_size: (Height, Width) of the resulting image.
            tip_radius: Radius of the AFM tip in Angstroms.
            smoothness: Smoothness of the Log-Sum-Exp max approximation.
            noise_level: Standard deviation of Gaussian noise added to heights.
            substrate_tilt: (tilt_x, tilt_y) slope of the substrate.
        """
        self.pixel_size = pixel_size
        self.grid_size = grid_size
        self.tip_radius = tip_radius
        self.smoothness = smoothness
        self.noise_level = noise_level
        self.substrate_tilt = substrate_tilt

        # Pre-compute grid
        h, w = grid_size
        x = jnp.arange(h) * pixel_size
        y = jnp.arange(w) * pixel_size
        self.grid_x, self.grid_y = jnp.meshgrid(x, y, indexing="ij")

        # Pre-compute tilt background
        self.tilt_bg = self.substrate_tilt[0] * self.grid_x + self.substrate_tilt[1] * self.grid_y

    def scan(
        self,
        positions: jax.Array,
        radii: Optional[jax.Array] = None,
        use_tip_dilation: bool = True,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Generates a height map for the given atomic positions.

        Args:
            positions: (N, 3) atomic coordinates.
            radii: (N,) atomic radii. If None, uses default 1.7A (Carbon).
            use_tip_dilation: If True, uses the realistic tip-dilation kernel.
            key: PRNG key for noise generation.
        """
        if radii is None:
            radii = jnp.full((positions.shape[0],), 1.7)

        if use_tip_dilation:
            img = tip_sample_height_map(
                positions,
                radii,
                self.grid_x,
                self.grid_y,
                self.tip_radius,
                self.smoothness,
            )
        else:
            img = simple_height_map_kernel(
                positions, radii, self.grid_x, self.grid_y, self.smoothness
            )
        
        # Apply tilt
        img = img + self.tilt_bg

        # Apply noise
        if self.noise_level > 0.0:
            if key is None:
                key = jax.random.PRNGKey(0)
            img = img + jax.random.normal(key, img.shape) * self.noise_level

        return cast(jax.Array, jnp.maximum(img, 0.0))

    def scan_movie(
        self,
        trajectory: jax.Array,
        radii: Optional[jax.Array] = None,
        frames_per_second: float = 10.0,
        trajectory_dt: Optional[float] = None,
        use_tip_dilation: bool = True,
        key: Optional[jax.Array] = None,
    ) -> jax.Array:
        """
        Generates a movie of HS-AFM images from a coordinate trajectory.
        Simulates scanning lag by sampling different time-steps for each line.
        Uses jax.lax.scan for memory-efficient iteration over frames.

        Args:
            trajectory: (T, N, 3) atomic coordinates over time.
            radii: (N,) atomic radii.
            frames_per_second: Nominal scan rate (FPS).
            trajectory_dt: Time between trajectory steps (seconds).
                If None, assumes 1.0 / FPS.
            use_tip_dilation: Whether to use the realistic tip model.
            key: PRNG key for noise generation.

        Returns:
            (T, H, W) array of height maps.
        """
        t_steps, n_atoms, _ = trajectory.shape
        h, w = self.grid_size

        if radii is None:
            radii = jnp.full((n_atoms,), 1.7)

        # Time per line in seconds. Scan is along y-axis (columns).
        dt_line = 1.0 / (frames_per_second * w)

        # If trajectory_dt is not provided, we assume one trajectory step per frame duration
        if trajectory_dt is None:
            trajectory_dt = 1.0 / frames_per_second

        def _scan_one_frame(frame_idx: jax.Array, frame_key: Optional[jax.Array]) -> jax.Array:
            t_start_idx = frame_idx
            
            def _scan_one_column(y_idx: jax.Array) -> jax.Array:
                # Calculate time offset in units of trajectory steps
                t_offset_exact = (y_idx * dt_line) / trajectory_dt
                t_offset = jnp.round(t_offset_exact).astype(jnp.int32)
                t_curr = jnp.minimum(t_start_idx + t_offset, t_steps - 1)

                # Extract column grid
                col_x = jax.lax.dynamic_slice(self.grid_x, (0, y_idx), (h, 1))
                col_y = jax.lax.dynamic_slice(self.grid_y, (0, y_idx), (h, 1))

                if use_tip_dilation:
                    res = tip_sample_height_map(
                        trajectory[t_curr], radii, col_x, col_y, self.tip_radius, self.smoothness
                    )
                else:
                    res = simple_height_map_kernel(
                        trajectory[t_curr], radii, col_x, col_y, self.smoothness
                    )
                return res[:, 0]

            img = jax.vmap(_scan_one_column)(jnp.arange(w)).T
            
            # Apply tilt
            img = img + self.tilt_bg
            
            # Apply noise
            if self.noise_level > 0.0 and frame_key is not None:
                img = img + jax.random.normal(frame_key, img.shape) * self.noise_level
                
            return jnp.maximum(img, 0.0)

        if key is None and self.noise_level > 0.0:
            key = jax.random.PRNGKey(0)

        if key is not None:
            keys = jax.random.split(key, t_steps)
        else:
            keys = jnp.zeros((t_steps, 2), dtype=jnp.uint32) # Dummy keys

        def _body_fn(carry, x):
            idx, k = x
            # Use key if it's not a dummy (meaning key was provided or defaulted)
            actual_key = k if key is not None else None
            return carry, _scan_one_frame(idx, actual_key)

        _, movie = jax.lax.scan(
            _body_fn, 
            None, 
            (jnp.arange(t_steps), keys)
        )
        
        return movie

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
