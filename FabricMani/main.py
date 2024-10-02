import os.path as osp
import json
import hydra
from omegaconf import DictConfig, OmegaConf
from chester import logger

from FabricMani.utils.utils import set_resource, configure_logger, configure_seed
from FabricMani.task.task_loader import task_loader

## OmegaConf & Hydra Config

# Resolvers used in hydra configs (see https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#resolvers)
OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for num_envs
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)

@hydra.main(config_path="cfg", config_name="config")
def main(args: DictConfig) -> None:

    # # Get the directory of the current script file
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # # Change the working directory to the script directory
    # os.chdir(script_dir)

    set_resource()  # To avoid pin_memory issue
    configure_logger(args.log_dir, args.exp_name)
    configure_seed(args.seed)

    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(OmegaConf.to_container(args, resolve=True), f, indent=2, sort_keys=True)
    # flatten cfg
    args = flatten_cfg(args)

    task_fuc = task_loader(args.task_name, args.real_robot)
    task_fuc(args)

def flatten_dict(d, prefix=''):
    flattened = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, prefix=f"{prefix}"))
        else:
            flattened[f"{prefix}{key}"] = value
    return flattened

def flatten_cfg(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    flattened_dict = flatten_dict(cfg_dict)
    flattened_cfg = OmegaConf.create(flattened_dict)
    return flattened_cfg


if __name__ == "__main__":
    main()
