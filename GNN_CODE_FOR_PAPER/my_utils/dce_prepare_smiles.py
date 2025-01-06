#%%
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
# %%
def bool_to_int(array):
    return [1 if cond else 0 for cond in array]

# %%
def atom_features(atom):
    return one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                           'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                           'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
                                           'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                           'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) + \
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) + \
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) + \
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) + \
                    bool_to_int([atom.GetIsAromatic()])

# %%
def bond_features(bond):
    bt = bond.GetBondType()
    return bool_to_int([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

# %%
def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

# %%
def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

# %%
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: 1 if x == s else 0, allowable_set))

# %%
def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: 1 if x == s else 0, allowable_set))


# %%
def smiles_to_dce_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Atom features
    atom_list = []
    for atom in mol.GetAtoms():
        atom_list.append(torch.tensor(atom_features(atom), dtype=torch.float))
    node_attr = torch.stack(atom_list, dim=0)
    store_list = []
    for bond in mol.GetBonds():
        src_node = bond.GetBeginAtom().GetIdx()
        tgt_node =  bond.GetEndAtom().GetIdx()
        bond_node =  bond_features(bond)
        store_list.append([src_node, tgt_node] + bond_node)
        store_list.append([tgt_node, src_node] + bond_node)
    edge_pd = pd.DataFrame(store_list, columns=['src', 'tgt'] +\
        ['bond_' + str(i) for i in ["SINGLE","DOUBLE",
                                    "TRIPLE","AROMATIC",
                                    "IsConjugated","IsInRing"]]) 
    edge_pd = edge_pd.sort_values(by=['src', 'tgt'])
    edge_index = torch.tensor(
        edge_pd.loc[:, ['src', 'tgt']].values.T.reshape(2, -1))
    edge_attr = torch.tensor(
        edge_pd.loc[:, edge_pd.columns.str.startswith('bond')].values
    )
    return node_attr, edge_index, edge_attr

    # Edge index and features

# %%