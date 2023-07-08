import os
import math
import random
import logging
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy

import torch
from torch.utils import data as torch_data
from torch.utils.data import IterableDataset

from torchdrug import data, utils, core, datasets
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from atom3d import datasets as da
from atom3d.datasets import LMDBDataset

import siamdiff.neighbors as nb

logger = logging.getLogger(__name__)


class Atom3DDataset:

    def protein_from_data_frame(self, df, atom_feature=None, bond_feature=None, 
                                residue_feature="default", mol_feature=None):  
        assert bond_feature is None
        assert mol_feature is None
        atom_feature = data.Protein._standarize_option(atom_feature)
        bond_feature = data.Protein._standarize_option(bond_feature)
        mol_feature = data.Protein._standarize_option(mol_feature)
        residue_feature = data.Protein._standarize_option(residue_feature)
        
        atom2residue = []
        atom_type = []
        residue_type = []
        atom_name = []
        is_hetero_atom = []
        residue_number = []
        occupancy = []
        b_factor = []
        insertion_code = []
        chain_id = []
        node_position = []
        _residue_feature = []
        _atom_feature = []
        last_residue = None
        for i, atom in df.iterrows():
            atom_type.append(data.feature.atom_vocab.get(atom['element'], 0))
            type = atom['resname']
            number = atom['residue']
            code = atom['insertion_code']
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in data.Protein.residue2id:
                    warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                    type = "GLY"
                residue_type.append(data.Protein.residue2id[type])
                residue_number.append(number)
                insertion_code.append(data.Protein.alphabet2id.get(code, 0))
                chain_id.append(data.Protein.alphabet2id.get(atom['chain'], 0))
                feature = []
                for name in residue_feature:
                    if name == "default":
                        feature = data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                    else:
                        raise ValueError('Feature %s not included' % name)
                _residue_feature.append(feature)
            name = atom['name']
            if name not in data.Protein.atom_name2id:
                name = "UNK"
            atom_name.append(data.Protein.atom_name2id[name])
            is_hetero_atom.append(atom['hetero'] != ' ')
            occupancy.append(atom['occupancy'])
            b_factor.append(atom['bfactor'])
            node_position.append([atom['x'], atom['y'], atom['z']])
            atom2residue.append(len(residue_type) - 1)
            feature = []
            for name in atom_feature:
                if name == "residue_symbol":
                    feature += \
                        data.feature.onehot(atom['element'], data.feature.atom_vocab, allow_unknown=True) + \
                        data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                else:
                    raise ValueError('Feature %s not included' % name)
            _atom_feature.append(feature)
        
        atom_type = torch.tensor(atom_type)
        residue_type = torch.tensor(residue_type)
        atom_name = torch.tensor(atom_name)
        is_hetero_atom = torch.tensor(is_hetero_atom)
        occupancy = torch.tensor(occupancy)
        b_factor = torch.tensor(b_factor)
        atom2residue = torch.tensor(atom2residue)
        residue_number = torch.tensor(residue_number)
        insertion_code = torch.tensor(insertion_code)
        chain_id = torch.tensor(chain_id)
        node_position = torch.tensor(node_position)
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)
        else:
            _residue_feature = None
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)
        else:
            _atom_feature = None

        return data.Protein(edge_list=None, num_node=len(atom_type), atom_type=atom_type, bond_type=[], 
                    residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue, 
                    is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                    residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id, 
                    node_position=node_position, atom_feature=_atom_feature, residue_feature=_residue_feature)

    @torch.no_grad()
    def construct_graph(self, data_list, model=None, batch_size=1, gpus=None, verbose=True):
        protein_list = []
        if gpus is None:
            device = torch.device("cpu")
        else:
            device = torch.device(gpus[comm.get_rank() % len(gpus)])
        model = model.to(device)
        t = range(0, len(data_list), batch_size)
        if verbose:
            t = tqdm(t, desc="Constructing graphs for training")
        for start in t:
            end = start + batch_size
            batch = data_list[start:end]
            proteins = data.Protein.pack(batch).to(device)
            if gpus and hasattr(proteins, "residue_feature"):
                with proteins.residue():
                    proteins.residue_feature = proteins.residue_feature.to_dense()
            proteins = model(proteins).cpu()
            for protein in proteins:
                if gpus and hasattr(protein, "residue_feature"):
                    with protein.residue():
                        protein.residue_feature = protein.residue_feature.to_sparse()
                protein_list.append(protein)
        return protein_list

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits


@R.register("datasets.RESDataset")
class RESDataset(data.ProteinDataset, Atom3DDataset):

    url = "https://zenodo.org/record/5026743/files/RES-split-by-cath-topology.tar.gz"
    dir_name = "RES-split-by-cath-topology"
    md5 = "c93125eb93d89e3b5898d5781c538662"
    processed_file = "RES-split-by-cath-topology.pkl.gz"

    def __init__(self, path, transform=None, verbose=1, **kwargs):
        path = os.path.join(os.path.expanduser(path), self.dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        tar_file = utils.download(self.url, path, md5=self.md5)
        utils.extract(tar_file)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.data = []
            self.sequences = []
            self.pdb_files = []
            self.kwargs = kwargs
            for split in ['train', 'val', 'test']:
                self.load_lmdb(os.path.join(path, 'split-by-cath-topology', 'data', split), verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [
            sum([self.data[i][1].batch_size for i, cur_split in enumerate(splits) if cur_split == split])
                for split in ["train", "val", "test"]
        ]

        self.protein = [item[0] for item in self.data]
        data_list = []
        for i, item in enumerate(tqdm(self.data, "Unpacking subunits")):
            for subunit in item[1].unpack():
                data_list.append((subunit, i))
        self.data = data_list
        
    def load_lmdb(self, lmdb_path, verbose, **kwargs):
        dataset = da.load_dataset(lmdb_path, "lmdb")
        if verbose:
            dataset = tqdm(dataset, "Constructing proteins from data frames")
        for i, item in enumerate(dataset):
            protein = self.protein_from_data_frame(item["atoms"], **kwargs)
            if not protein:
                logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % item["id"])
                continue
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            subunits = []
            for sub in item['labels'].itertuples():
                _, num, aa = sub.subunit.split('_')
                if aa not in data.Protein.residue2id: continue
                label = data.Protein.residue2id[aa]
                num = int(num)
                node_index = item['subunit_indices'][sub.Index]
                subprotein = protein.subgraph(node_index)
                node_mask = (subprotein.residue_number[subprotein.atom2residue] == num) & \
                            (subprotein.atom_name == data.Protein.atom_name2id["CA"])
                if (node_mask.sum() > 1).all(): continue

                subunit = data.Graph(edge_list=[], num_node=len(node_index))
                with subunit.node():
                    subunit.node_index = torch.tensor(node_index)
                with subunit.graph():
                    subunit.residue_num = torch.tensor(num)
                    subunit.label = torch.tensor(label)
                subunits.append(subunit)
            if len(subunits) == 0: continue
            subunits = data.Graph.pack(subunits)
            self.data.append((protein, subunits))
            self.sequences.append(protein.to_sequence())
            self.pdb_files.append(os.path.join(lmdb_path, str(i)))

    def get_item(self, index):
        subunit, protein_index = self.data[index]
        protein = self.protein[protein_index]
        protein = protein.subgraph(subunit.node_index)
        residue_mask = torch.zeros((protein.num_residue,), dtype=torch.bool)
        residue_mask[protein.atom2residue] = 1
        protein = protein.subresidue(residue_mask)
        node_mask = (protein.residue_number[protein.atom2residue] == subunit.residue_num) & \
                    (protein.atom_name == data.Protein.atom_name2id["CA"])
        with protein.node():
            protein.ca_mask = node_mask
        with protein.graph():
            protein.label = torch.as_tensor(subunit.label)
        
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
            protein.residue_feature[protein.residue_number == subunit.residue_num, :] = 0
            protein.residue_type[protein.residue_number == subunit.residue_num] = 0
        if not hasattr(protein, "atom_feature"):
            with protein.atom():
                protein.atom_feature = torch.zeros((protein.num_atom, 1))

        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["residue type"]

    @property
    def node_feature_dim(self):
        return self.protein[0].node_feature.shape[-1]

    @property
    def edge_feature_dim(self):
        return self.protein[0].edge_feature.shape[-1]
    
    @property
    def residue_feature_dim(self):
        return self.protein[0].residue_feature.shape[-1]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: residue type",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))


@R.register("datasets.PSRDataset")
class PSRDataset(data.ProteinDataset, Atom3DDataset):

    url = "https://zenodo.org/record/4915648/files/PSR-split-by-year.tar.gz"
    dir_name = "PSR-split-by-year"
    md5 = "8647b9d10d0a79dff81d1d83c825e74c"
    processed_file = "PSR-split-by-year.pkl.gz"

    def __init__(self, path, transform=None, verbose=1, **kwargs):
        path = os.path.join(os.path.expanduser(path), self.dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        tar_file = utils.download(self.url, path, md5=self.md5)
        utils.extract(tar_file)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.data = []
            self.sequences = []
            self.pdb_files = []
            self.kwargs = kwargs
            for split in ['train', 'val', 'test']:
                self.load_lmdb(os.path.join(path, 'split-by-year', 'data', split), verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        
        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("val"), splits.count("test")]
        
    def load_lmdb(self, lmdb_path, verbose, **kwargs):
        dataset = da.load_dataset(lmdb_path, "lmdb")
        if verbose:
            dataset = tqdm(dataset, "Constructing proteins from data frames")
        for i, data in enumerate(dataset):
            protein = self.protein_from_data_frame(data["atoms"], **kwargs)
            if not protein:
                logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % data["id"])
                continue
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            with protein.graph():
                protein.gdt_ts = torch.tensor(data["scores"]["gdt_ts"])
            self.data.append(protein)
            self.sequences.append(protein.to_sequence())
            self.pdb_files.append(os.path.join(lmdb_path, str(i)))

    def get_item(self, index):
        protein = self.data[index]
        protein = protein.subgraph(protein.atom_type != 0)
        
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        
        item = {"graph": protein}
        item["gdt_ts"] = protein.gdt_ts
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["gdt_ts"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: gdt_ts",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
    

@R.register("datasets.PIPDataset")
class PIPDataset(IterableDataset, Atom3DDataset, core.Configurable):

    def __init__(self, path, transform=None, graph_construction_model=None, **kwargs):
        path = os.path.expanduser(path)
        self.dataset = LMDBDataset(path)
        self.graph_construction_model = graph_construction_model
        self.transform = transform
        self.kwargs = kwargs
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            gen = self._dataset_generator(list(range(len(self.dataset))), shuffle=True)
        else:  
            per_worker = int(math.ceil(len(self.dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.dataset))
            gen = self._dataset_generator(
                list(range(len(self.dataset)))[iter_start:iter_end],
                shuffle=True)
        return gen

    @torch.no_grad()
    def _df_to_graph(self, struct_df, chain_res):
        struct_df = struct_df[struct_df.element != 'H'].reset_index(drop=True)

        chain, resnum = chain_res
        res_df = struct_df[(struct_df.chain == chain) & (struct_df.residue == resnum)]
        if 'CA' not in res_df.name.tolist():
            return None
        ca_pos = res_df[res_df['name']=='CA'][['x', 'y', 'z']].astype(np.float32).to_numpy()[0]

        kd_tree = scipy.spatial.KDTree(struct_df[['x','y','z']].to_numpy())
        graph_pt_idx = kd_tree.query_ball_point(ca_pos, r=30.0, p=2.0)
        graph_df = struct_df.iloc[graph_pt_idx].reset_index(drop=True)
        
        ca_idx = np.where((graph_df.chain == chain) & (graph_df.residue == resnum) & (graph_df.name == 'CA'))[0]
        if len(ca_idx) != 1:
            return None

        protein = self.protein_from_data_frame(graph_df, **self.kwargs)
        node_mask = torch.zeros((protein.num_node, ), dtype=torch.bool)
        node_mask[ca_idx] = 1
        with protein.node():
            protein.ca_idx = node_mask

        protein = self.construct_graph([protein], model=self.graph_construction_model, verbose=0)[0]

        return protein

    def _dataset_generator(self, indices, shuffle=True):
        if shuffle: random.shuffle(indices)
        with torch.no_grad():
            for idx in indices:
                data = self.dataset[idx]

                neighbors = data['atoms_neighbors']
                pairs = data['atoms_pairs']
                
                for i, (ensemble_name, target_df) in enumerate(pairs.groupby(['ensemble'])):
                    sub_names, (bound1, bound2, _, _) = nb.get_subunits(target_df)
                    positives = neighbors[neighbors.ensemble0 == ensemble_name]
                    negatives = nb.get_negatives(positives, bound1, bound2)
                    negatives['label'] = 0
                    labels = self._create_labels(positives, negatives, num_pos=10, neg_pos_ratio=1)
                    
                    for index, row in labels.iterrows():
                        label = float(row['label'])
                        chain_res1 = row[['chain0', 'residue0']].values
                        chain_res2 = row[['chain1', 'residue1']].values
                        graph1 = self._df_to_graph(bound1, chain_res1)
                        graph2 = self._df_to_graph(bound2, chain_res2)
                        if (graph1 is None) or (graph2 is None):
                            continue
                        item = {
                            "graph1": graph1,
                            "graph2": graph2,
                            "interaction": label
                        }
                        if self.transform:
                            item = self.transform(item)
                        yield item

    def _create_labels(self, positives, negatives, num_pos, neg_pos_ratio):
        frac = min(1, num_pos / positives.shape[0])
        positives = positives.sample(frac=frac)
        n = positives.shape[0] * neg_pos_ratio
        n = min(negatives.shape[0], n)
        negatives = negatives.sample(n, random_state=0, axis=0)
        labels = pd.concat([positives, negatives])[['chain0', 'residue0', 'chain1', 'residue1', 'label']]
        return labels

    @property
    def tasks(self):
        """List of tasks."""
        return ["interaction"]

    def __repr__(self):
        lines = [
            "#task: interaction",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))

    
@R.register("datasets.MSPDataset")
class MSPDataset(data.ProteinDataset, Atom3DDataset):

    url = "https://zenodo.org/record/4962515/files/MSP-split-by-sequence-identity-30.tar.gz"
    dir_name = "MSP-split-by-sequence-identity-30"
    md5 = "6628e8efac12648d3b78bb0fc0d8860c"
    processed_file = "MSP-split-by-sequence-identity-30.pkl.gz"

    def __init__(self, path, transform=None, verbose=1, **kwargs):
        path = os.path.join(os.path.expanduser(path), self.dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        tar_file = utils.download(self.url, path, md5=self.md5)
        utils.extract(tar_file)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.data = []
            self.sequences = []
            self.pdb_files = []
            self.kwargs = kwargs
            for split in ['train', 'val', 'test']:
                self.load_lmdb(os.path.join(path, 'split-by-sequence-identity-30', 'data', split), verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("val"), splits.count("test")]

    def load_lmdb(self, lmdb_path, verbose, **kwargs):
        dataset = da.load_dataset(lmdb_path, "lmdb")
        if verbose:
            dataset = tqdm(dataset, "Constructing proteins from data frames")
        for i, data in enumerate(dataset):
            wt = self.protein_from_data_frame(data["original_atoms"], **kwargs)
            mt = self.protein_from_data_frame(data["mutated_atoms"], **kwargs)
            if not wt or not mt:
                logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % data["id"])
                continue
            if hasattr(wt, "residue_feature"):
                with wt.residue():
                    wt.residue_feature = wt.residue_feature.to_sparse()
                with mt.residue():
                    mt.residue_feature = mt.residue_feature.to_sparse()
            with wt.graph():
                wt.label = torch.tensor(data["label"] == '1')
            self.data.append((wt, mt))
            self.sequences.append((wt.to_sequence(), mt.to_sequence()))
            self.pdb_files.append(os.path.join(lmdb_path, str(i)))

    def get_item(self, index):
        wt = self.data[index][0].clone()
        wt = wt.subgraph(wt.atom_type != 0)
        mt = self.data[index][1].clone()
        mt = mt.subgraph(mt.atom_type != 0)

        if hasattr(wt, "residue_feature"):
            with wt.residue():
                wt.residue_feature = wt.residue_feature.to_dense()
            with mt.residue():
                mt.residue_feature = mt.residue_feature.to_dense()

        item = {"graph1": wt, "graph2": mt}
        item["label"] = wt.label
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["label"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: label",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
    

@R.register("datasets.ECDataset")
class ECDataset(datasets.EnzymeCommission):

    def __init__(self, **kwargs):
        super(ECDataset, self).__init__(**kwargs)