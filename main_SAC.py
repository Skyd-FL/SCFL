import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
# from torch.utils.tensorboard import SummaryWriter
from agents.sac.agent_sac import *
from agents.sac.modules.relay_memory import *
from envs_DM.environment import *
from envs.env_utils import *
from envs.env_agent_utils import *
from utils.setting_setup import *
from utils.result_utils import *



args = get_arguments()

env = DRGO_env(args)

agent = SAC(
    args,
    env
)
agent.train(args)