import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils, datasets, models, tasks
from torchdrug.utils import comm


logger = logging.getLogger(__file__)


def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg, dirname=None):
    file_name = "working_dir.tmp" # % os.environ["SLURM_JOB_ID"]
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    if "dataset" in cfg:
        dataset_class = cfg.dataset["class"]
    else:
        dataset_class = cfg.train_set["class"]
    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], dataset_class, cfg.task.model["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))
    if dirname is not None:
        working_dir = os.path.join(working_dir, dirname)

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=0)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def build_atom3d_solver(cfg, train_set, valid_set, test_set, use_solver=False):
    if cfg.task['class'] == "EC":
        cfg.task.task = [_ for _ in range(len(train_set.dataset.tasks))]
    task = core.Configurable.load_config_dict(cfg.task)
    if use_solver:
        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

        # Need to define a solver for preprocessing
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)
    else:
        task.preprocess(train_set, valid_set, test_set)

        cfg.optimizer.params = task.parameters()
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        if use_solver:
            solver.scheduler = scheduler

    if use_solver and cfg.get("checkpoint") is not None:
        solver.load(cfg.checkpoint)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        task.model.load_state_dict(model_dict)

    if use_solver:
        return solver, scheduler
    else:
        return task, optimizer, scheduler
    

def build_pretrain_solver(cfg, dataset):
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#dataset: %d" % (len(dataset)))

    task = core.Configurable.load_config_dict(cfg.task)
    cfg.optimizer.params = task.parameters()
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)
    solver = core.Engine(task, dataset, None, None, optimizer, **cfg.engine)

    if cfg.get("model_checkpoint") is not None:
        if comm.get_rank() == 0:
            logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
        cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
        model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
        model_dict.pop("alphas")
        task.load_state_dict(model_dict, strict=False)
    
    return solver