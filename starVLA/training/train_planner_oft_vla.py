"""VLA trainer entrypoint for the planner_oft project."""

import argparse

import torch.distributed as dist
from omegaconf import OmegaConf

from starVLA.dataloader.planner_oft_datasets import make_planner_oft_vla_dataloader
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training import train_starvla as base
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def prepare_data(cfg, accelerator, output_dir):
    base.logger.info(f"Creating planner_oft VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = make_planner_oft_vla_dataloader(cfg)
    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


base.prepare_data = prepare_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/ActiveArena_Astribot/train_files/managed_runs/planner_oft/config.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    base.main(cfg)
