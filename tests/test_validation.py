import jax
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator


def test_tip_dilation_geometry() -> None:
    """
    Verify that tip dilation follows the analytical formula for a spherical tip:
    H(d) = z + sqrt((r + R)^2 - d^2) - R
    where R is tip radius, r is atom radius, d is horizontal distance.
    """
    R = 50.0  # 5nm tip
    r = 2.0  # 0.2nm atom
    z = 10.0
    pixel_size = 1.0
    sim = AFMSimulator(pixel_size=pixel_size, grid_size=(101, 101), tip_radius=R, smoothness=0.01)

    # Atom at center (50, 50)
    coords = jnp.array([[50.0, 50.0, z]])
    radii = jnp.array([r])

    img = sim.scan(coords, radii, use_tip_dilation=True)

    # Check at distance d = 10.0 from center
    # Distance in pixels is 10.0
    h_measured = img[60, 50]

    d = 10.0
    eff_r = r + R
    expected_h = z + jnp.sqrt(eff_r**2 - d**2) - R

    # Smoothness=0.01 should be very accurate
    assert jnp.isclose(h_measured, expected_h, atol=0.05)


def test_lse_error_quantification() -> None:
    """
    Quantify the overestimation error of Log-Sum-Exp.
    For N identical heights H, LSE(H, s) = H + s * log(N).
    """
    s = 0.5
    sim = AFMSimulator(grid_size=(10, 10), smoothness=s)

    # 10 atoms at the same position and height
    coords = jnp.tile(jnp.array([[5.0, 5.0, 10.0]]), (10, 1))
    radii = jnp.full((10,), 2.0)

    img = sim.scan(coords, radii)
    peak = img[5, 5]

    H = 12.0  # z + r
    N = 10
    expected_peak = H + s * jnp.log(N)

    assert jnp.isclose(peak, expected_peak, atol=0.01)


def test_scanning_lag_stroboscopic_effect() -> None:
    """
    A moving atom should appear 'sheared' in a scan-movie.
    """
    h, w = 64, 64
    sim = AFMSimulator(grid_size=(h, w))

    # Atom moving horizontally across frames
    # Frame 0: (32, 32)
    # Frame 1: (34, 32)
    trajectory = jnp.array([[[32.0, 32.0, 10.0]], [[34.0, 32.0, 10.0]]])

    # We want t_offset to be 0 for y=0 and 1 for y=63
    # t_offset = (y * dt_line) / traj_dt
    # For y=63, we want t_offset = 1.
    # traj_dt = 63 * dt_line = 63 / (2.0 * 64) = 0.4921875
    traj_dt = 0.5
    movie = sim.scan_movie(trajectory, frames_per_second=2.0, trajectory_dt=traj_dt)

    frame0 = movie[0]

    # y=0: t_offset = 0. Peak at 32.
    # y=63: t_offset = (63 * 0.0078) / 0.5 = 0.98. Round to 1. Peak at 34.

    assert (
        jnp.argmax(frame0[:, 32]) == 32
    )  # Column 32, t_offset = 32*0.0078/0.5 = 0.5 -> round to 0 or 1?
    # Actually 32/64 = 0.5. round(0.5) is 0.
    # So column 32 should still be 32.
    # Column 40: 40/64 = 0.625 -> round to 1.
    assert jnp.argmax(frame0[:, 40]) == 34


def test_gradient_wrt_tip_radius() -> None:
    """Gradients should flow back to the tip_radius parameter."""
    sim = AFMSimulator(grid_size=(16, 16))
    coords = jnp.array([[8.0, 8.0, 5.0]])
    radii = jnp.array([2.0])

    def loss_fn(r_tip: float) -> jax.Array:
        # We need to pass tip_radius to the kernel or use a fresh simulator
        # The kernels are what's actually differentiable
        from synth_afm.kernels import tip_sample_height_map

        img = tip_sample_height_map(coords, radii, sim.grid_x, sim.grid_y, tip_radius=r_tip)
        return jnp.sum(img)

    grad = jax.grad(loss_fn)(20.0)
    assert grad > 0.0  # Increasing tip radius should increase total height (dilation)


def test_noise_application() -> None:
    """Verify that noise_level adds randomness to the scan."""
    sim = AFMSimulator(grid_size=(32, 32), noise_level=1.0)
    coords = jnp.array([[16.0, 16.0, 10.0]])

    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(2)

    img1 = sim.scan(coords, key=key1)
    img2 = sim.scan(coords, key=key2)

    assert not jnp.allclose(img1, img2)
    assert jnp.std(img1) > 0.5  # Substantial noise


def test_substrate_tilt() -> None:
    """Verify that substrate_tilt adds a linear gradient."""
    # 0.1A per Angstrom tilt along X
    sim = AFMSimulator(grid_size=(10, 10), pixel_size=1.0, substrate_tilt=(0.1, 0.0))

    # Flat surface (no atoms)
    coords = jnp.array([[0.0, 0.0, -100.0]])  # Far away
    img = sim.scan(coords)

    # At x=0, tilt=0. At x=9, tilt=0.9
    assert jnp.isclose(img[0, 0], 0.0)
    assert jnp.isclose(img[9, 0], 0.9)


def test_movie_memory_efficiency() -> None:
    """Scan a long trajectory to ensure lax.scan works as expected."""
    T = 100
    sim = AFMSimulator(grid_size=(16, 16))
    trajectory = jnp.zeros((T, 1, 3))

    movie = sim.scan_movie(trajectory)
    assert movie.shape == (T, 16, 16)


def test_movie_default_noise() -> None:
    """Verify that noise is applied in scan_movie even if no key is passed."""
    sim = AFMSimulator(grid_size=(16, 16), noise_level=1.0)
    trajectory = jnp.zeros((2, 1, 3))

    movie = sim.scan_movie(trajectory)  # No key passed
    assert jnp.std(movie[0]) > 0.5
    assert not jnp.allclose(movie[0], movie[1])
