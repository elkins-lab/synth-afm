"""
Example: Generating Synthetic HS-AFM data from PDB structures.

This script demonstrates how to:
1. Generate a protein model using synth-pdb.
2. Simulate a single high-resolution AFM scan with tip dilation.
3. Simulate a movie with "Scanning Lag" (temporal distortion).
"""

import jax.numpy as jnp
from synth_afm.simulator import AFMSimulator

# 1. Generate / Setup Coordinates
# For this example, we'll create a simple 'synthetic' protein (a set of atoms)
# representing a 10nm long alpha helix.
n_atoms = 100
t = jnp.linspace(0, 10.0, n_atoms)
x = 5.0 + 2.0 * jnp.cos(t)
y = 5.0 + 2.0 * jnp.sin(t)
z = t
coords = jnp.stack([x, y, z], axis=1) # (100, 3)

# 2. Initialize the HS-AFM Simulator
# We'll use 0.5 Angstrom pixels and a standard 10nm tip radius.
sim = AFMSimulator(
    pixel_size=0.5, 
    grid_size=(64, 64), 
    tip_radius=20.0, 
    smoothness=0.2
)

# 3. Perform a Static Scan
print("Simulating static AFM scan...")
static_img = sim.scan(coords, use_tip_dilation=True)
print(f"Static image shape: {static_img.shape}")

# 4. Simulate HS-AFM Scanning Lag
# We'll create a 10-frame trajectory where the molecule rotates.
print("Generating trajectory...")
trajectory_frames = []
for angle in jnp.linspace(0, jnp.pi, 10):
    # Rotate around Z axis
    rot = jnp.array([
        [jnp.cos(angle), -jnp.sin(angle), 0],
        [jnp.sin(angle),  jnp.cos(angle), 0],
        [0, 0, 1]
    ])
    trajectory_frames.append(coords @ rot.T)

trajectory = jnp.stack(trajectory_frames) # (10, 100, 3)

print("Simulating HS-AFM movie with scanning lag...")
# We use a slow FPS to make the lag effects obvious
movie = sim.scan_movie(trajectory, frames_per_second=2.0)
print(f"Movie shape: {movie.shape} (Frames, Height, Width)")

# Note: In a real environment, you would use matplotlib to visualize
# plt.imshow(static_img)
# plt.show()

print("\nSuccess! Synthetic AFM data generated.")
