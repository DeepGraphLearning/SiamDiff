# Copied from https://github.com/drorlab/atom3d/blob/v0.2.1/atom3d/datasets/ppi/neighbors.py
"""Methods to extract protein interface labels pair."""
import click
import numpy as np
import pandas as pd
import scipy.spatial as spa

import atom3d.util.log as log

logger = log.get_logger('neighbors')


index_columns = \
    ['ensemble', 'subunit', 'structure', 'model', 'chain', 'residue']


def neighbors_from_ensemble(ensemble, cutoff, cutoff_type):
    _, (bdf0, bdf1, udf0, udf1) = get_subunits(ensemble)
    neighbors = get_neighbors(bdf0, bdf1, cutoff, cutoff_type)
    if udf0 is not None and udf1 is not None:
        # Map to unbound.
        neighbors['subunit0'] = neighbors['subunit0'].apply(
            lambda x: x.replace('bound', 'unbound'))
        neighbors['subunit1'] = neighbors['subunit1'].apply(
            lambda x: x.replace('bound', 'unbound'))
        neighbors['structure0'] = neighbors['structure0'].apply(
            lambda x: x.replace('_b_', '_u_'))
        neighbors['structure1'] = neighbors['structure1'].apply(
            lambda x: x.replace('_b_', '_u_'))
        neighbors = remove_unmatching(neighbors, udf0, udf1)

    return neighbors


def get_subunits(ensemble):
    subunits = ensemble['subunit'].unique()

    if len(subunits) == 4:
        lb = [x for x in subunits if x.endswith('ligand_bound')][0]
        lu = [x for x in subunits if x.endswith('ligand_unbound')][0]
        rb = [x for x in subunits if x.endswith('receptor_bound')][0]
        ru = [x for x in subunits if x.endswith('receptor_unbound')][0]
        bdf0 = ensemble[ensemble['subunit'] == lb]
        bdf1 = ensemble[ensemble['subunit'] == rb]
        udf0 = ensemble[ensemble['subunit'] == lu]
        udf1 = ensemble[ensemble['subunit'] == ru]
        names = (lb, rb, lu, ru)
    elif len(subunits) == 2:
        udf0, udf1 = None, None
        bdf0 = ensemble[ensemble['subunit'] == subunits[0]]
        bdf1 = ensemble[ensemble['subunit'] == subunits[1]]
        names = (subunits[0], subunits[1], None, None)
    else:
        raise RuntimeError('Incorrect number of subunits for pair')
    return names, (bdf0, bdf1, udf0, udf1)


def remove_unmatching(neighbors, df0, df1):
    # Remove entries that are not present in input structures.
    _, res_to_idx = _get_idx_to_res_mapping(
        pd.concat([df0, df1]))
    to_drop = []
    for i, neighbor in neighbors.iterrows():
        res0 = tuple(neighbor[['ensemble0', 'subunit0', 'structure0', 'model0',
                               'chain0', 'residue0']])
        res1 = tuple(neighbor[['ensemble1', 'subunit1', 'structure1', 'model1',
                               'chain1', 'residue1']])
        if res0 not in res_to_idx or res1 not in res_to_idx:
            to_drop.append(i)
    logger.info(
        f'Removing {len(to_drop):} / {len(neighbors):} due to no matching '
        f'residue in unbound.')
    neighbors = neighbors.drop(to_drop).reset_index(drop=True)
    return neighbors


def get_neighbors(df0, df1, cutoff, cutoff_type):
    """Given pair of structures, generate neighbors."""
    if cutoff_type == 'CA':
        neighbors = _get_ca_neighbors(df0, df1, cutoff)
    else:
        neighbors = _get_heavy_neighbors(df0, df1, cutoff)
    neighbors['label'] = 1
    return neighbors


def get_res(df):
    """Get all residues."""
    return df[index_columns].drop_duplicates()


def get_negatives(neighbors, df0, df1):
    """Get negative pairs, given positives."""
    idx_to_res0, res_to_idx0 = _get_idx_to_res_mapping(df0)
    idx_to_res1, res_to_idx1 = _get_idx_to_res_mapping(df1)
    all_pairs = np.zeros((len(idx_to_res0.index), len(idx_to_res1.index)))
    for i, neighbor in neighbors.iterrows():
        res0 = tuple(neighbor[['ensemble0', 'subunit0', 'structure0', 'model0',
                               'chain0', 'residue0']])
        res1 = tuple(neighbor[['ensemble1', 'subunit1', 'structure1', 'model1',
                               'chain1', 'residue1']])
        idx0 = res_to_idx0[res0]
        idx1 = res_to_idx1[res1]
        all_pairs[idx0, idx1] = 1
    pairs = np.array(np.where(all_pairs == 0)).T
    res0 = idx_to_res0.iloc[pairs[:, 0]][index_columns]
    res1 = idx_to_res1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    res = pd.concat((res0, res1), axis=1)
    return res


def _get_idx_to_res_mapping(df):
    """Define mapping from residue index to single id number."""
    idx_to_res = get_res(df).reset_index(drop=True)
    res_to_idx = idx_to_res.reset_index().set_index(index_columns)['index']
    return idx_to_res, res_to_idx


def _get_ca_neighbors(df0, df1, cutoff):
    """Get neighbors for alpha-carbon based distance."""
    ca0 = df0[df0['name'] == 'CA']
    ca1 = df1[df1['name'] == 'CA']

    dist = spa.distance.cdist(ca0[['x', 'y', 'z']], ca1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    res0 = ca0.iloc[pairs[:, 0]][index_columns]
    res1 = ca1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    res = pd.concat((res0, res1), axis=1)
    return res


def _get_heavy_neighbors(df0, df1, cutoff):
    """Get neighbors for heavy atom based distance."""
    heavy0 = df0[df0['element'] != 'H']
    heavy1 = df1[df1['element'] != 'H']

    dist = spa.distance.cdist(heavy0[['x', 'y', 'z']], heavy1[['x', 'y', 'z']])
    pairs = np.array(np.where(dist < cutoff)).T
    res0 = heavy0.iloc[pairs[:, 0]][index_columns]
    res1 = heavy1.iloc[pairs[:, 1]][index_columns]
    res0 = res0.reset_index(drop=True).add_suffix('0')
    res1 = res1.reset_index(drop=True).add_suffix('1')
    # We concatenate and find unique _pairs_.
    res = pd.concat((res0, res1), axis=1)
    res = res.drop_duplicates()
    return res