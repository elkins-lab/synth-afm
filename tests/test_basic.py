import jax
import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator


def test_simulator_output_shape() -> None:
    """The simulator should return an image of the requested grid size."""
    sim = AFMSimulator(grid_size=(32, 32))
    coords = jnp.array([[16.0, 16.0, 5.0]])  # One atom in the middle
    image = sim.scan(coords)
    assert image.shape == (32, 32)


def test_height_accuracy() -> None:
    """A single atom at a grid point should produce the correct height."""
    pixel_size = 1.0
    sim = AFMSimulator(pixel_size=pixel_size, grid_size=(10, 10))

    # Place atom exactly at grid index (5, 5) with z=10.0 and radius=2.0
    z_pos = 10.0
    radius = 2.0
    coords = jnp.array([[5.0, 5.0, z_pos]])
    radii = jnp.array([radius])

    image = sim.scan(coords, radii)

    # Peak height should be z_pos + radius
    peak_height = image[5, 5]
    expected = z_pos + radius
    assert jnp.isclose(peak_height, expected, atol=0.2)  # Smoothness affects exact match


def test_gradient_flow() -> None:
    """Gradients should flow from the image back to coordinates."""
    sim = AFMSimulator(grid_size=(16, 16))
    coords = jnp.array([[8.0, 8.0, 5.0]])

    def loss_fn(c: jax.Array) -> jax.Array:
        img = sim.scan(c)
        return jnp.sum(img**2)

    grad = jax.grad(loss_fn)(coords)
    assert grad.shape == coords.shape
    assert not jnp.all(grad == 0.0)


def test_tip_dilation_effect() -> None:
    """Tip dilation should produce a wider profile than a point tip."""
    sim = AFMSimulator(pixel_size=0.5, grid_size=(20, 20), tip_radius=5.0)
    coords = jnp.array([[5.0, 5.0, 2.0]])

    img_point = sim.scan(coords, use_tip_dilation=False)
    img_dilated = sim.scan(coords, use_tip_dilation=True)

    # Sum of heights should be much larger for dilated image
    assert jnp.sum(img_dilated) > jnp.sum(img_point)

    # Dilated image should have non-zero values further from the center
    # Center is at index (10, 10) for pixel_size 0.5
    assert img_dilated[15, 10] > 0.0
    assert img_point[15, 10] == 0.0


def test_scan_movie_shape() -> None:
    """Scanning a trajectory should return a 3D volume of images."""
    sim = AFMSimulator(grid_size=(16, 16))
    trajectory = jnp.zeros((5, 10, 3))  # 5 frames, 10 atoms
    movie = sim.scan_movie(trajectory)
    assert movie.shape == (5, 16, 16)


def test_scan_movie_no_tip_dilation() -> None:
    """Scanning a movie without tip dilation should also work."""
    sim = AFMSimulator(grid_size=(16, 16))
    trajectory = jnp.zeros((2, 5, 3))
    movie = sim.scan_movie(trajectory, use_tip_dilation=False)
    assert movie.shape == (2, 16, 16)
