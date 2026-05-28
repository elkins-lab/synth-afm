"""
Standardized atomic and biophysical constants for HS-AFM simulation.
Radii values based on standard van der Waals parameters (Bondi, 1964).
"""

# Default van der Waals radii (in Angstroms)
VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "P": 1.80,
    "S": 1.80,
    "CA": 1.70, # Alpha carbon specifically
}

DEFAULT_ATOM_RADIUS = 1.70
DEFAULT_TIP_RADIUS = 20.0 # 2nm is standard for sharp HS-AFM tips
