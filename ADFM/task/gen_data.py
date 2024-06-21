from ADFM.module.gen_dataset import DataCollector
from ADFM.utils.env_utils import create_env

def gen_data(args):

    env = create_env(args, render_mode='cloth')

    collectors = {phase: DataCollector(args, phase, env) for phase in ['train', 'valid']}

    for phase in ['train', 'valid']:
        collectors[phase].gen_dataset()

    print('Dataset generated in', args.dataf)