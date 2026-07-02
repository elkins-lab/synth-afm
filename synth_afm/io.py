import biotite.structure as struc
import biotite.structure.io as strucio
import jax.numpy as jnp
from typing import Tuple
from synth_afm.constants import VDW_RADII, DEFAULT_ATOM_RADIUS


def load_coords_and_radii(
    file_path: str, filter_protein: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Loads coordinates and assigns vdW radii from a PDB/mmCIF file.

    Args:
        file_path: Path to the structure file.
        filter_protein: If True, filters only amino acid atoms.

    Returns:
        tuple: (coords (N, 3), radii (N,)) as JAX arrays.
    """
    atom_array = strucio.load_structure(file_path)

    if filter_protein:
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]

    coords = jnp.array(atom_array.coord)

    # Assign radii based on element
    elements = atom_array.element
    radii_list = []
    for elem in elements:
        radii_list.append(VDW_RADII.get(elem.upper(), DEFAULT_ATOM_RADIUS))

    radii = jnp.array(radii_list)

    return coords, radii
