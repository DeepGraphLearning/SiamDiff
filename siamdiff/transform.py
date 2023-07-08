import os
import glob
import math
import numpy as np

import torch

from torchdrug import core, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from siamdiff import rotamer


@R.register("transforms.NoiseTransform")
class NoiseTransform(core.Configurable):

    def __init__(self, noise_type="gaussian", sigma=0.3):
        assert noise_type in ["gaussian", "torsion"]
        self.noise_type = noise_type
        self.sigma = sigma

    def __call__(self, item):
        graph = item["graph"].clone()
        if self.noise_type == "gaussian":
            perturb_noise = torch.randn_like(graph.node_position)
            graph.node_position = graph.node_position + perturb_noise * self.sigma
        elif self.noise_type == "torsion":
            torsion_updates = torch.randn((graph.num_residue, 4), device=graph.device) * self.sigma * np.pi
            rotamer.rotate_side_chain(graph, torsion_updates)
        item["graph2"] = graph
        return item


@R.register("transfroms.AtomFeature")
class AtomFeature(core.Configurable):

    def __init__(self, atom_feature=None, keys="graph"):
        self.atom_feature = atom_feature
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def __call__(self, item):
        for key in self.keys:
            graph = item[key]
            graph = graph.subgraph(graph.atom_type != 0)
            graph = graph.subgraph(graph.atom_type < 18)
            residue2num_atom = graph.atom2residue.bincount(minlength=graph.num_residue)
            graph = graph.subresidue(residue2num_atom > 0)
            if self.atom_feature == "residue_symbol":
                atom_feature = torch.cat([
                    functional.one_hot(graph.atom_type.clamp(max=17), 18),
                    functional.one_hot(graph.residue_type[graph.atom2residue], 21)
                ], dim=-1)
            else:
                raise ValueError
            with graph.atom():
                graph.atom_feature = atom_feature
            item[key] = graph
        return item


@R.register("transforms.TruncateProteinPair")
class TruncateProteinPair(core.Configurable):

    def __init__(self, max_length=None, random=False):
        self.truncate_length = max_length
        self.random = random

    def __call__(self, item):
        new_item = item.copy()
        graph1 = item["graph1"]
        graph2 = item["graph2"]
        length = graph1.num_residue
        if length <= self.truncate_length:
            return item
        residue_mask = graph1.residue_type != graph2.residue_type
        index = residue_mask.nonzero()[:, 0]
        if self.random:
            start = math.randint(index, min(index + self.truncate_length, length)) - self.truncate_length
        else:
            start = min(index - self.truncate_length // 2, length - self.truncate_length)
        start = max(start, 0)
        end = start + self.truncate_length
        mask = torch.zeros(length, dtype=torch.bool, device=graph1.device)
        mask[start:end] = True
        new_item["graph1"] = graph1.subresidue(mask)
        new_item["graph2"] = graph2.subresidue(mask)

        return new_item
