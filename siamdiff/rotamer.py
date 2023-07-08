import math
import numpy as np

import torch


atom_name_vocab = {
    "C": 0, "CA": 1, "CB": 2, "CD": 3, "CD1": 4, "CD2": 5, "CE": 6, "CE1": 7, "CE2": 8,
    "CE3": 9, "CG": 10, "CG1": 11, "CG2": 12, "CH2": 13, "CZ": 14, "CZ2": 15, "CZ3": 16,
    "N": 17, "ND1": 18, "ND2": 19, "NE": 20, "NE1": 21, "NE2": 22, "NH1": 23, "NH2": 24,
    "NZ": 25, "O": 26, "OD1": 27, "OD2": 28, "OE1": 29, "OE2": 30, "OG": 31, "OG1": 32,
    "OH": 33, "OXT": 34, "SD": 35, "SG": 36
}
residue_list = [
    "GLY", "ALA", "SER", "PRO", "VAL", "THR", "CYS", "ILE", "LEU", "ASN",
    "ASP", "GLN", "LYS", "GLU", "MET", "HIS", "PHE", "ARG", "TYR", "TRP"
]
residue_vocab = {r: i for i, r in enumerate(residue_list)}
three_to_one = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H",
    "ILE": "I", "LYS": "K", "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", 
    "ARG": "R", "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"
}
one_to_three = {v: k for k, v in three_to_one.items()}
# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "NE",
        "CZ",
        "NH1",
        "NH2",
        "",
        "",
        "",
    ],
    "ASN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "ND2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "ASP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "OD1",
        "OD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "NE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "OE1",
        "OE2",
        "",
        "",
        "",
        "",
        "",
    ],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "ND1",
        "CD2",
        "CE1",
        "NE2",
        "",
        "",
        "",
        "",
    ],
    "ILE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "CD1",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LEU": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "LYS": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD",
        "CE",
        "NZ",
        "",
        "",
        "",
        "",
        "",
    ],
    "MET": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "SD",
        "CE",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "PHE": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "",
        "",
        "",
    ],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "OG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "TRP": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    ],
    "TYR": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "CE1",
        "CE2",
        "CZ",
        "OH",
        "",
        "",
    ],
    "VAL": [
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG1",
        "CG2",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
    ],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}
"""
restype_atomname_index_map[i_resi][j_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_atom: atom name, 0-36
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
restype_atom14_index_map = -torch.ones((len(residue_list), 37), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    for value, name in enumerate(restype_name_to_atom14_names[resi_name3]):
        if name in atom_name_vocab:
            restype_atom14_index_map[i_resi][atom_name_vocab[name]] = value

chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
}

"""
chi_atom_index_map[i_resi][j_chi][k_atom]:
  i_resi: index of residue_list, specifies resi_type
  j_chi: chi number, 0-3
  k_atom: k-th atom in the torsion angle, 0-3
  value: index of atom_names, specifies atom_type, -1 means no such torsion
chi_atom14_index_map[i_resi][j_chi][k_atom]:
  value: index in residue type, 0-13, specifies atom_type, -1 means no atoms
"""
chi_atom_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
chi_atom14_index_map = -torch.ones((len(residue_list), 4, 4), dtype=torch.long)
for i_resi, resi_name3 in enumerate(residue_list):
    chi_angles_atoms_i = chi_angles_atoms[resi_name3]
    for j_chi, atoms in enumerate(chi_angles_atoms_i):
        for k_atom, atom in enumerate(atoms):
            chi_atom_index_map[i_resi][j_chi][k_atom] = atom_name_vocab[atom]
            chi_atom14_index_map[i_resi][j_chi][k_atom] = restype_atom14_index_map[i_resi][atom_name_vocab[atom]]
# Masks out non-existent torsions.
chi_masks = chi_atom_index_map != -1


@torch.no_grad()
def rotate_side_chain(protein, rotate_angles):
    assert rotate_angles.shape[0] == protein.num_residue
    assert rotate_angles.shape[1] == 4
    node_position = torch.zeros((protein.num_residue, 14, 3), dtype=torch.float, device=protein.device)
    atom14index = restype_atom14_index_map[protein.residue_type[protein.atom2residue], protein.atom_name]
    mask = atom14index != -1
    node_position[protein.atom2residue[mask], atom14index[mask], :] = protein.node_position[mask]

    chi_atom14_index = chi_atom14_index_map[protein.residue_type].to(protein.device)    # (num_residue, 4, 4) 0~13
    chi_atom14_mask = chi_atom14_index != -1
    chi_atom14_index[~chi_atom14_mask] = 0
    for i in range(4):
        atom_1, atom_2, atom_3, atom_4 = chi_atom14_index[:, i, :].unbind(-1)   # (num_residue, )
        atom_2_position = torch.gather(node_position, -2, atom_2[:, None, None].expand(-1, -1, 3))    # (num_residue, 1, 3)
        atom_3_position = torch.gather(node_position, -2, atom_3[:, None, None].expand(-1, -1, 3))    # (num_residue, 1, 3)
        axis = atom_3_position - atom_2_position
        axis_normalize = axis / (axis.norm(dim=-1, keepdim=True) + 1e-10)
        rotate_angle = rotate_angles[:, i, None, None]

        # Rotate all subsequent atoms by the rotation angle
        rotate_atoms_position = node_position - atom_2_position  # (num_residue, 14, 3)
        parallel_component = (rotate_atoms_position * axis_normalize).sum(dim=-1, keepdim=True) \
                                * axis_normalize
        perpendicular_component = rotate_atoms_position - parallel_component
        perpendicular_component_norm = perpendicular_component.norm(dim=-1, keepdim=True) + 1e-10
        perpendicular_component_normalize = perpendicular_component / perpendicular_component_norm
        normal_vector = torch.cross(axis_normalize.expand(-1, 14, -1), perpendicular_component_normalize, dim=-1)
        transformed_atoms_position = perpendicular_component * rotate_angle.cos() + \
                                normal_vector * perpendicular_component_norm * rotate_angle.sin() + \
                                parallel_component + atom_2_position    # (num_residue, 14, 3)
        assert not transformed_atoms_position.isnan().any()
        chi_mask = chi_atom14_mask[:, i, :].all(dim=-1, keepdim=True)  # (num_residue, 1)
        atom_mask = torch.arange(14, device=protein.device)[None, :] >= atom_4[:, None] # (num_residue, 14)
        mask = (atom_mask & chi_mask).unsqueeze(-1).expand_as(node_position)
        node_position[mask] = transformed_atoms_position[mask]

    mask = atom14index != -1
    protein.node_position[mask] = node_position[protein.atom2residue[mask], atom14index[mask]]
    return chi_atom14_mask.all(dim=-1)
