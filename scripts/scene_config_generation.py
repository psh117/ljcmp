import numpy as np
import torch
import torch.utils.data
"""pytorch lightning"""

import argparse

from ljcmp.utils.model_utils import generate_scene_config, load_model
from ljcmp.utils.generate_environment import generate_environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=500)
parser.add_argument('--config_size', type=int, default=100)
parser.add_argument('--seed', type=int, default=1107)
parser.add_argument('--exp_name', '-E', type=str, default='panda_triple')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed) 

constraint, model_info, condition, update_scene_from_yaml, set_constraint, _ = generate_environment(args.exp_name)
constraint.set_early_stopping(True)
constraint_model, _ = load_model(args.exp_name, model_info, False)

generate_scene_config(constraint, constraint_model, model_info=model_info,
                      condition=condition, update_scene_from_yaml=update_scene_from_yaml, 
                      start=args.start, end=args.end, config_size=args.config_size)