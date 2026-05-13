#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import random
import socket
import sys
import time
from os.path import dirname, realpath

import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict
from timm.optim import optim_factory
import wandb
import yaml

# add repo root to path
sys.path.append(dirname(dirname(realpath(__file__))))

from sandstone import datasets, engines, models, optimizers, schedulers
from sandstone.utils.loading import get_eval_dataset_loader, get_train_dataset_loader
from sandstone.models.attention import init_attn_impl
from sandstone.utils import misc
from sandstone.utils.misc import logger, set_loglevel, get_augmentations, set_all_seeds
from sandstone.utils.optim import NativeScalerWithGradNormCount as NativeScaler

# constants
NUM_DEBUG_BATCHES = 2
ATTN_IMPL = os.environ.get("ATTN_IMPL", "flash_attention2")
init_attn_impl(ATTN_IMPL)

# ----------------- config helpers -----------------

class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

def merge_dict(a, b, path=None, allow_replace=False):
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)], allow_replace=allow_replace)
            elif a[key] == b[key]:
                pass
            else:
                if allow_replace:
                    logger.info(f"Replacing key at {'.'.join(path + [str(key)])} with {b[key]}")
                    a[key] = b[key]
                else:
                    raise ValueError("Conflict at {'.'.join(path + [str(key)])}")
        else:
            a[key] = b[key]
    return a

def merge_cli_opt(config, key, value):
    key_hierarchy = key.split(".")
    item_container = config
    for hierarchy in key_hierarchy[:-1]:
        if isinstance(item_container, list):
            hierarchy = int(hierarchy)
        item_container = item_container[hierarchy]

    try:
        original_value = item_container[key_hierarchy[-1]]
    except KeyError as e:
        raise KeyError(f"KeyError: {e}, the current parent structure: {item_container}")

    # basic type conversions
    if isinstance(original_value, bool):
        if value.lower() in ("true", "1"):
            value = True
        elif value.lower() in ("false", "0"):
            value = False
        else:
            raise ValueError(f"Value {value} is not a boolean value")
    elif isinstance(original_value, int):
        value = int(value)
    elif isinstance(original_value, float):
        value = float(value)
    elif isinstance(original_value, list):
        value = json.loads(value)
        if len(original_value) > 0:
            assert type(original_value[0]) == type(value[0])
            assert all([type(v) == type(value[0]) for v in value])

    assert original_value is None or type(original_value) == type(value), f"{type(original_value)} != {type(value)}"

    logger.info(f"Overriding {key} with {value} (original value: {original_value})")
    item_container[key_hierarchy[-1]] = value

def merge_cli_opts(config, cli_opts):
    assert len(cli_opts) % 2 == 0, f"{len(cli_opts)} should be even"
    for key, value in zip(cli_opts[::2], cli_opts[1::2]):
        merge_cli_opt(config, key, value)

def load_config(config_path, easydict=True, cli_opts=None, override_base_config=None, return_base_config=True):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=UniqueKeyLoader)
    if override_base_config is not None:
        config["base_config"] = override_base_config

    if "base_config" not in config:
        if cli_opts is not None:
            merge_cli_opts(config, cli_opts)
        return EasyDict(config) if easydict else config

    base_path = os.path.join(os.path.dirname(config_path), config["base_config"])
    base_config = load_config(base_path, easydict=False)
    merged_config = merge_dict(base_config, config, allow_replace=True)
    if cli_opts is not None:
        merge_cli_opts(merged_config, cli_opts)
    merged_config = EasyDict(merged_config) if easydict else merged_config
    if not return_base_config:
        del merged_config["base_config"]
    return merged_config

def parse_args(args_strings=None):
    parser = argparse.ArgumentParser(description="Sandstone research repo.")
    parser.add_argument("config", metavar="C", type=str, nargs="?", help="path for config",
                        default="configs/classif/mnist_demo.yaml")
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--resume", type=str, required=False, help="resume from checkpoint")
    parser.add_argument("--exp_name", type=str, required=False, help="experiment name override")
    parser.add_argument("--evaluate", action="store_true", help="evaluation only")
    parser.add_argument("--opts", default=[], nargs=argparse.REMAINDER, help="KEY VALUE pairs to override config")

    cli_args = parser.parse_args(args_strings) if args_strings is None else parser.parse_args(args_strings)
    config_path = cli_args.config
    logger.info(f"Loading config from {config_path}")
    args = load_config(config_path, cli_opts=cli_args.opts)
    args.config_path = config_path

    if cli_args.no_wandb:
        args.main.disable_wandb = True
    if cli_args.resume:
        args.engine.kwargs.resume = cli_args.resume
    if cli_args.evaluate:
        args.main.disable_wandb = True
        args.main.phases.train = False
        args.main.phases.dev = False
        args.main.phases.test = True
        args.main.force_loading_train_dataloader = True
    if cli_args.exp_name:
        args.main.exp_name = cli_args.exp_name
    if cli_args.debug:
        args.main.disable_wandb = True
        args.main.seed = 42
        args.dataloader.num_workers = 0
        args.engine.kwargs.limit_num_batches = NUM_DEBUG_BATCHES

    return args

# ----------------- distributed init -----------------

def init_distributed_mode(args):
    """Init distributed training environment. Mutates args to set
    .distributed, .global_rank, .local_rank, .world_size, .gpu
    """
    # default single process
    args.distributed = False
    args.global_rank = -1
    args.local_rank = -1
    args.world_size = 1
    args.gpu = None

    # torchrun / torch.distributed.launch sets RANK/WORLD_SIZE/LOCAL_RANK
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.global_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.gpu = args.local_rank
        args.distributed = True

    # SLURM common envs: SLURM_PROCID, SLURM_NTASKS, SLURM_NNODES, SLURM_TASKS_PER_NODE
    elif "SLURM_PROCID" in os.environ:
        args.global_rank = int(os.environ["SLURM_PROCID"])
        # try to infer world size
        if "SLURM_NTASKS" in os.environ:
            args.world_size = int(os.environ["SLURM_NTASKS"])
        elif "SLURM_NNODES" in os.environ and "SLURM_TASKS_PER_NODE" in os.environ:
            args.world_size = int(os.environ["SLURM_NNODES"]) * int(os.environ["SLURM_TASKS_PER_NODE"])
        else:
            # fallback: assume one task per GPU on node
            gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            args.world_size = int(os.environ.get("SLURM_NNODES", 1)) * gpus

        # local GPU index
        if torch.cuda.is_available():
            args.gpu = args.global_rank % torch.cuda.device_count()
            args.local_rank = args.gpu
        else:
            args.gpu = None
            args.local_rank = 0
        args.distributed = True

    else:
        # Not distributed: single-process
        logger.info("Not using distributed mode")
        misc.is_master = True
        return args

    # At this point distributed=True
    # Set device if GPU available
    if args.gpu is not None:
        try:
            torch.cuda.set_device(args.gpu)
        except Exception as e:
            logger.warning(f"Failed to set CUDA device {args.gpu}: {e}")

    args.dist_backend = "nccl" if torch.cuda.is_available() else "gloo"
    args.dist_url = os.environ.get("DIST_URL", "env://")

    logger.info(f"| distributed init (global rank {args.global_rank}): {args.dist_url}, gpu {args.gpu} / world_size {args.world_size}")

    # IMPORTANT: rank must be global rank across all processes
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.global_rank)
    dist.barrier()

    # mark master
    misc.setup_for_distributed(args.global_rank == 0)
    misc.setup_dirs(args, args.global_rank == 0)
    logger.info(f"| initialized host {socket.gethostname()} as rank {args.global_rank}")

    return args

# ----------------- main experiment builder -----------------

def build_experiment(args):
    # top-level main args are stored in args.main in your config
    main_args = args.main
    exp_name = main_args.exp_name
    main_args.experiment_checkpoints_dir = os.path.join(main_args.checkpoints_dir, exp_name)
    os.makedirs(main_args.experiment_checkpoints_dir, exist_ok=True)

    # init distributed mode (mutates main_args)
    init_distributed_mode(main_args)

    # set seeds
    seed = main_args.seed if hasattr(main_args, "seed") else 42
    # incorporate rank in seed for determinism across processes
    rank = main_args.global_rank if hasattr(main_args, "global_rank") else -1
    set_all_seeds(seed + (rank if rank >= 0 else 0))

    main_args.multi_gpu = getattr(main_args, "global_rank", -1) > -1
    logger.info(f"Multi-GPU: {main_args.multi_gpu}")

    # allow only master to overwrite/checkpoint dir
    if main_args.global_rank <= 0:
        allow_overwriting = main_args.allow_overwriting_experiment_checkpoints or not os.path.exists(main_args.experiment_checkpoints_dir)
        assert allow_overwriting, f"Checkpoint path ({main_args.experiment_checkpoints_dir}) already exists. Set main.allow_overwriting_experiment_checkpoints to True to overwrite."
    else:
        main_args.disable_wandb = True

    # set NODE_RANK if PMIX_RANK is present
    if "PMIX_RANK" in os.environ:
        os.environ["NODE_RANK"] = os.environ["PMIX_RANK"]

    # wandb mode for non-master processes
    if main_args.disable_wandb:
        wandb_mode = "disabled"
    elif os.environ.get("SGE_IN_USE", False):
        wandb_mode = "offline"
    else:
        wandb_mode = None

    # init wandb only on master (or disabled)
    if main_args.global_rank <= 0:
        wandb.init(
            project=main_args.wandb_project,
            entity=main_args.wandb_entity,
            name=exp_name,
            dir=main_args.experiment_checkpoints_dir,
            mode=wandb_mode,
            tags=args.main.tags if hasattr(args.main, "tags") else None,
            settings=wandb.Settings(start_method="thread")
        )
        if wandb.run is not None:
            wandb.run.log_code(main_args.experiment_checkpoints_dir)
    else:
        # ensure wandb disabled for non-master
        wandb.init(mode="disabled")

    logger.info(f"Checkpoint directory: {main_args.experiment_checkpoints_dir}")

    main_args.callbacks = None  # remove callbacks for pickling / distributed

    # augmentations / datasets
    augmentations = get_augmentations(args.dataset.image_augmentations, args)
    test_augmentations = get_augmentations(args.dataset.test_image_augmentations, args)

    dataset_cls = datasets.__dict__[args.dataset.type]
    dataset_info = {}

    if main_args.phases.train or main_args.force_loading_train_dataloader:
        train_dataset = dataset_cls(args, augmentations=augmentations, split_group="train", **{**args.dataset.shared_dataset_kwargs, **args.dataset.dataset_train_kwargs})
        train_dataloader = get_train_dataset_loader(args, train_dataset)
        dataset_info["train"] = getattr(train_dataset, "info", None)

    if main_args.phases.train or main_args.phases.dev or (args.main.use_val_as_test and main_args.phases.test):
        dev_dataset = dataset_cls(args, augmentations=test_augmentations, split_group="dev", **{**args.dataset.shared_dataset_kwargs, **args.dataset.dataset_dev_kwargs})
        multi_gpu_eval = main_args.multi_gpu and args.dataloader.multi_gpu_eval
        eval_dataloader = get_eval_dataset_loader(args, dev_dataset, shuffle=False, multi_gpu_eval=multi_gpu_eval)
        dataset_info["dev"] = getattr(dev_dataset, "info", None)

    if main_args.phases.test:
        if args.main.use_val_as_test:
            test_dataset = dev_dataset
        else:
            test_dataset = dataset_cls(args, augmentations=test_augmentations, split_group="test", **{**args.dataset.shared_dataset_kwargs, **args.dataset.dataset_test_kwargs})
        test_dataloader = get_eval_dataset_loader(args, test_dataset, shuffle=False, multi_gpu_eval=False)
        dataset_info["test"] = getattr(test_dataset, "info", None)

    # engine
    engine_args = args.engine.kwargs
    engine_full_kwargs = dict(args=args, dataset_info=dataset_info, **args.engine.kwargs)
    engine: engines.Engine = engines.__dict__[args.engine.type](**engine_full_kwargs)
    logger.info("Engine built!")

    # model
    model = models.__dict__[args.model.type](args=args, **args.model.kwargs)
    model_summary = misc.get_model_summary(model, keys=["params"])
    misc.log_dict(model_summary)

    # device selection
    device = torch.device(main_args.gpu if (main_args.gpu is not None and torch.cuda.is_available()) else "cpu")
    model.to(device, non_blocking=True)
    model_without_ddp = model

    logger.info(f"Model: {model_without_ddp}")


    # DDP wrap (after moving to device)
    if main_args.multi_gpu:
        # convert syncbn if requested before DDP? Usually convert after model creation but before DDP
        if main_args.get("sync_bn", False):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[main_args.gpu] if main_args.gpu is not None else None,
            output_device=main_args.gpu,
            broadcast_buffers=True,
            find_unused_parameters=main_args.get("find_unused_parameters", False),
        )

    elif main_args.get("sync_bn", False):
        # sync_bn is only meaningful in DDP; warn otherwise
        logger.warning("sync_bn requested but not running distributed; skipping SyncBatchNorm.")

    if main_args.get("compile", False):
        model = torch.compile(model)

    # optimizer and param groups
    if args.optimizer.get('layer_decay', None) is not None:
        param_groups = model_without_ddp.param_groups_lrd(args.optimizer.timm_weight_decay, no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.optimizer.layer_decay)
        logger.info(f"Layer decay: {args.optimizer.get('layer_decay', None)}, weight decay: {args.optimizer.get('timm_weight_decay', None)}")
    elif args.optimizer.get('timm_weight_decay', None) is not None:
        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.optimizer.timm_weight_decay)
    else:
        param_groups = model.parameters()

    optimizer: optimizers.Optimizer = optimizers.__dict__[args.optimizer.type](param_groups, **args.optimizer.kwargs)

    scheduler_interval = args.optimizer.scheduler.interval
    if scheduler_interval == "epoch":
        lr_scheduler = schedulers.__dict__[args.optimizer.scheduler.type](optimizer, **args.optimizer.scheduler.kwargs)
    else:
        lr_scheduler = schedulers.__dict__[args.optimizer.scheduler.type](optimizer, lr=args.optimizer.kwargs.lr, **args.optimizer.scheduler.kwargs)
        assert scheduler_interval == "step", f"scheduler type is not epoch or step: {scheduler_interval}"
    lr_scheduler.interval = scheduler_interval

    loss_scaler = NativeScaler()

    # ensure all processes wait before starting heavy I/O
    if main_args.multi_gpu:
        dist.barrier()

    # resume if required
    if engine_args.resume:
        checkpoint = torch.load(engine_args.resume, map_location="cpu")
        # handle wrapped checkpoint
        if list(checkpoint["model"].keys())[0].startswith("module."):
            checkpoint["model"] = {k[len("module."):]: v for k, v in checkpoint["model"].items()}
        model_without_ddp.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint and "lr_scheduler" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            engine_args.start_epoch = checkpoint["epoch"] + 1
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])
            if "global_step" in checkpoint:
                engine.global_step = checkpoint["global_step"]
    else:
        engine_args.start_epoch = 0

    # training loop
    logger.info(f"Start training for {engine_args.max_epochs} epochs")
    start_time = time.time()

    if main_args.phases.train:
        for epoch in range(engine_args.start_epoch, engine_args.max_epochs):
            if main_args.multi_gpu:
                # set epoch on distributed sampler for shuffling
                train_dataloader.sampler.set_epoch(epoch)

            train_stats = engine.train_one_epoch(
                model,
                train_dataloader,
                optimizer,
                device,
                epoch,
                loss_scaler,
                lr_scheduler=None if scheduler_interval == "epoch" else lr_scheduler,
                args=engine_args,
                clip_grad=engine_args.get("clip_grad", None),
                log_grad_norm=engine_args.get("log_grad_norm", None),
                log_interval=engine_args.get("log_interval", 200),
            )

            # only master saves checkpoints
            if misc.get_is_master():
                state_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                    "scaler": loss_scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": engine.global_step,
                    "args": args,
                }
                engine.save_on_master(ckpt_dir=main_args.experiment_checkpoints_dir, epoch=-1, state=state_dict)
                if epoch % main_args.ckpt_freq == 0 or epoch == engine_args.max_epochs - 1:
                    engine.save_on_master(ckpt_dir=main_args.experiment_checkpoints_dir, epoch=epoch, state=state_dict)

            # evaluation
            if main_args.phases.dev:
                eval_stats = engine.evaluate(model, eval_dataloader, device, epoch, test=False, gather_predictions=multi_gpu_eval)

            # scheduler step (epoch based)
            if scheduler_interval == "epoch":
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(eval_stats['val_loss'])
                else:
                    lr_scheduler.step()

    # testing
    if main_args.phases.test:
        test_stats = engine.evaluate(model, test_dataloader, device, epoch=None, test=True, gather_predictions=False)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Experiment time {total_time_str}")

# ----------------- entrypoint -----------------

if __name__ == "__main__":
    __spec__ = None
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    logger.info(f"Local rank: {local_rank}")
    set_loglevel(debug=(local_rank <= 0))

    config_args = parse_args()
    build_experiment(config_args)
