from FabricMani.utils.utils import vv_to_args
from FabricMani.module.dynamics import Dynamic
from FabricMani.module.edge import Edge
from FabricMani.utils.env_utils import create_env

import json
import os.path as osp

def train_dy(args):

    if not args.real_robot:
        env = create_env(args, render_mode=args.render_mode)
    else:
        env = None

    # load vcd_edge
    if args.edge_model_path is not None:
        edge_model_vv = json.load(open(osp.join(args.edge_model_path, 'variant.json')))
        edge_model_args = vv_to_args(edge_model_vv)
        dia_edge = Edge(edge_model_args, env=env)
        dia_edge.load_model(args.edge_model_path)
        print('EdgeGNN successfully loaded from ', args.edge_model_path, flush=True)
    else:
        dia_edge = None

    dyn_model = Dynamic(args, env, dia_edge)

    dyn_model.train()