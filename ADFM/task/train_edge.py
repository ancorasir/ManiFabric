from ADFM.module.edge import Edge
from ADFM.utils.env_utils import create_env

def train_edge(args):

    if args.local:
        env = create_env(args.env)
    else:
        env = None

    edge = Edge(args, env)

    edge.train()