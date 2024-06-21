from ADFM.module.critic import Critic
from ADFM.utils.env_utils import create_env


def train_critic(args):

    if args.local:
        env = create_env(args.env)
    else:
        env = None

    critic = Critic(args, env)

    #critic.train()

    if args.visualize:
        critic.visualize()