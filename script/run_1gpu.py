import os
import sys
import math
import time
import tqdm
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler

from torchdrug import core, models, tasks, datasets, utils, data
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util

from siamdiff import dataset, model, task, transform

import warnings
warnings.simplefilter("ignore")


def train(cfg, model, optimizer, scheduler, train_set, valid_set, test_set, device):    
    best_epoch, best_val = None, -1e9
    
    for epoch in range(cfg.num_epoch):
        model.train()
        loss = loop(train_set, model, optimizer=optimizer, 
                    max_time=cfg.get("train_time"), device=device)
        torch.save(model.state_dict(), "model_epoch_%d.pth" % epoch)
        print("\nEPOCH %d TRAIN loss: %.8f" % (epoch, loss))

        model.eval()
        with torch.no_grad():
            metric = test(valid_set, model, max_time=cfg.get("val_time"), device=device)
        print("\nEPOCH %d" % epoch, "VAL metric:", metric)

        if metric[cfg.eval_metric] > best_val:
            best_epoch, best_val = epoch, metric[cfg.eval_metric]
            with torch.no_grad():
                best_test_metric = test(test_set, task, max_time=cfg.test_time, device=device)
            print("\nEPOCH %d" % epoch, "TEST metric:", best_test_metric)
        print("BEST %d VAL %s: %.8f TEST %s: %.8f" % 
                (best_epoch, cfg.eval_metric, best_val, cfg.eval_metric, best_test_metric[cfg.eval_metric]))

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif scheduler is not None:
            scheduler.step()

    return best_epoch, best_val
    
    
def loop(dataset, model, optimizer=None, max_time=None, device=None):
    start = time.time()
    
    t = tqdm.tqdm(dataset)
    total_loss, total_count = 0, 0
    
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        if optimizer: optimizer.zero_grad()
        try:
            batch = utils.cuda(batch, device=device)
            loss, metric = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        
        total_loss += float(loss)
        total_count += 1
        
        if optimizer:
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e): raise(e)
                torch.cuda.empty_cache()
                print('Skipped batch due to OOM', flush=True)
                continue
            
        t.set_description(f"{total_loss/total_count:.8f}")
    
    return total_loss / total_count


def test(dataset, model, max_time=None, device=None):
    start = time.time()
    t = tqdm.tqdm(dataset)
    
    preds, targets = [], []
    for batch in t:
        if max_time and (time.time() - start) > 60 * max_time: break
        try:
            batch = utils.cuda(batch, device=device)
            pred, target = model.predict_and_target(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e): raise(e)
            torch.cuda.empty_cache()
            print('Skipped batch due to OOM', flush=True)
            continue
        
        preds.append(pred)
        targets.append(target)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    metric = model.evaluate(pred, target)

    return metric


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    dirname = os.path.basename(args.config) + "_" + str(args.seed)
    working_dir = util.create_working_directory(cfg, dirname=dirname)

    seed = args.seed
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    if cfg.dataset["class"] in ["MSPDataset", "PSRDataset", "RESDataset"]:
        _dataset = core.Configurable.load_config_dict(cfg.dataset)
        train_set, valid_set, test_set = _dataset.split()
    elif cfg.dataset["class"] in ["PIPDataset"]:
        datasets = []
        for i in range(3): 
            dataset_cfg = cfg.dataset.copy()
            dataset_cfg["path"] = dataset_cfg["path"][i]
            if "split_path" in dataset_cfg:
                dataset_cfg["split_path"] = dataset_cfg["split_path"][i]
            datasets.append(core.Configurable.load_config_dict(dataset_cfg))
        train_set, valid_set, test_set = datasets

    task, optimizer, scheduler = util.build_atom3d_solver(cfg, train_set, valid_set, test_set)

    # Only support single GPU now
    device = torch.device(cfg.engine.gpus[0]) if cfg.engine.gpus else torch.device("cpu")
    if device.type == "cuda":
        task = task.cuda(device)

    train_loader, valid_loader, test_loader = [
        data.DataLoader(dataset, cfg.engine.batch_size, shuffle=(cfg.dataset["class"] != "PIPDataset"), num_workers=cfg.engine.num_worker)
            for dataset in [train_set, valid_set, test_set]
    ]

    best_epoch, best_val = train(cfg.train, task, optimizer, scheduler, train_loader, valid_loader, test_loader, device)
    task.load_state_dict("model_epoch_%d.pth" % best_epoch)
    task.eval()
    with torch.no_grad():
        metric = test(test_loader, task, max_time=None, device=device)
    print("TEST metric", metric)
